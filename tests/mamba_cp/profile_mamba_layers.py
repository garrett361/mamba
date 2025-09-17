from contextlib import nullcontext
import json
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from mamba_ssm.modules.mamba2_cp import Mamba2CP


class SequentialRecorded(nn.Module):
    def __init__(self, mod_list: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(mod_list)

    def forward(self, inputs) -> torch.Tensor:
        outputs = inputs
        for idx, layer in enumerate(self.layers):
            with record_function(f"layer_{idx}"):
                outputs = layer(outputs)
        return outputs


def trace_handler(prof):
    global impl
    global args
    global model
    global rank
    global world_size

    trace_ranks = {int(r) % world_size for r in args.trace_ranks.split(",")}
    if rank not in trace_ranks:
        return

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
            header=f"{impl=}",
        )
    )
    trace_dir = Path(args.trace_dir)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    subdir = trace_dir.joinpath(f"world_{world_size}/{timestamp}/")
    subdir.mkdir(parents=True, exist_ok=True)
    seq = args.seq_len_per_gpu * world_size
    filename = f"trace_{impl}_step_{prof.step_num}_rank_{rank}_bsz_{args.batch_size}_seq_{seq}_layers_{args.n_layers}.json"

    prof.export_chrome_trace(str(subdir.joinpath(f"{filename}")))
    if not rank:
        with open(subdir.joinpath(f"args_{impl}.json"), "w") as fp:
            json.dump(vars(args), fp)
        short_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with open(subdir.joinpath(short_hash), "w") as _:
            pass


if __name__ == "__main__":
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    # some setups
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl", timeout=timedelta(seconds=60), device_id=device
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        parser = ArgumentParser()
        parser.add_argument("--active", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--d_model", type=int, default=4096)  # bamba 9.8b default
        parser.add_argument("--impls", type=str, default="allgather,serial")
        parser.add_argument("--iters", type=int, default=50)
        parser.add_argument("--repeat", type=int, default=2)
        parser.add_argument("--n_layers", type=int, default=10)
        parser.add_argument("--seq_len_per_gpu", type=int, default=65536)
        parser.add_argument("--trace_dir", default="/dev/null")
        parser.add_argument("--trace_ranks", type=str, default="0,-1")
        parser.add_argument("--wait", type=int, default=3)
        parser.add_argument("--warmup", type=int, default=3)
        parser.add_argument("--bwd", type=bool, default=False) 

        args = parser.parse_args()
        impls = args.impls.split(",")
        iters_per_impl = args.repeat * (args.active + args.wait + args.warmup)

        if rank == 0:
            print(f"--> running with these configs {args}")

        inputs = torch.randn(
            args.batch_size,
            args.seq_len_per_gpu,
            args.d_model,
            device=device,
            dtype=dtype,
        )

        for impl in impls:
            mamba_stack = SequentialRecorded(
                [
                    Mamba2CP(
                        d_model=args.d_model,
                        cp_mesh=mesh,
                        cp_mamba_impl=impl,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(args.n_layers)
                ]
            )

            ctx = nullcontext if args.bwd else torch.no_grad
            with (
                profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    # record_shapes=True,
                    # with_stack=True,
                    # profile_memory=True,
                    schedule=torch.profiler.schedule(
                        wait=args.wait,
                        warmup=args.warmup,
                        active=args.active,
                        repeat=args.repeat,
                    ),
                    on_trace_ready=trace_handler,
                ) as prof,
                ctx(),
            ):
                for _ in range(iters_per_impl):
                    mamba_stack(inputs)
                    prof.step()
            del mamba_stack

    finally:
        dist.destroy_process_group()
