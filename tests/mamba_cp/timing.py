import argparse
import datetime
import os

import torch
from torch.distributed.fsdp import MixedPrecision


import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import Mamba2CP


class CUDATimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._end_events: list[torch.cuda.Event] = []

    def __enter__(self) -> "CUDATimer":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._end_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._end_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(end) / 1e3
            for start, end in zip(self._start_events, self._end_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def reset(self) -> None:
        self._start_events.clear()
        self._end_events.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--d_state", type=int, default=256)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not rank:
        print(f"{args=}")

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=30),
    )
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
    mamba_kwargs = {
        "d_state": args.d_state,
        "headdim": args.headdim,
        "d_model": args.d_model,
        "chunk_size": args.chunk_size,
        "device": device,
    }
    if args.cp:
        model = nn.Sequential(
            *[Mamba2CP(mesh=mesh, **mamba_kwargs) for _ in range(args.n_layer)]
        )
    else:
        model = nn.Sequential(*[Mamba2(**mamba_kwargs) for _ in range(args.n_layer)])
    model_fsdp = FSDP(
        model,
        process_group=mesh.get_group(),
        auto_wrap_policy=ModuleWrapPolicy([Mamba2, Mamba2CP]),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        ),
    )
    if not rank:
        print(f"{model_fsdp=}")

    inputs = torch.randn(
        args.batch_size, args.seq_len, args.d_model, device=device, dtype=dtype
    )

    for _ in range(args.warmups):
        outputs = model_fsdp(inputs)
        outputs.mean().backward()
        model_fsdp.zero_grad()

    with CUDATimer() as timer:
        for _ in range(args.iters):
            outputs = model_fsdp(inputs)
            outputs.sum().backward()
            model_fsdp.zero_grad()
        dist.barrier()

    total_toks = args.batch_size * args.seq_len * world_size * args.iters
    secs = timer.get_total_time_s()
    toks_per_sec = total_toks / secs
    if not rank:
        print(f"Total tokens: {total_toks}")
        print(f"Total Secs: {secs}")
        print(f"Tok/sec {toks_per_sec}")
