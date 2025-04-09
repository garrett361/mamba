import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy


class CUDATimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._stop_events: list[torch.cuda.Event] = []

    def __enter__(self) -> "CUDATimer":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._stop_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._stop_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(stop) / 1e3
            for start, stop in zip(self._start_events, self._stop_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def get_std_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return torch.tensor(time_list_s).std().item()

    def reset(self) -> None:
        self._start_events.clear()
        self._stop_events.clear()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=16)

    args = parser.parse_args()
    dist.init_process_group("nccl")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        # Init NCCL - NOTE: @goon - might be needed for multi-node to work correctly? Was getting
        # odd NCCL errors due to all_to_alls without it.
        dist.barrier()

        if not rank:
            print(f"{args=})")

        dtype = torch.bfloat16

        # Clear cache, like triton do_bench
        cache_size_bytes = 512 * 2**20  # 512k
        cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        results = defaultdict(list)
        # 128MiB to 8 GiB
        for exp in range(27, 34):
            num_bytes = 2**exp
            if not rank:
                print(f"Launching: 2^{exp} elements")
            numel = num_bytes // dtype.itemsize
            inputs = torch.randn(
                numel,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            outputs = torch.empty_like(inputs)

            for _ in range(args.warmups):
                dist.all_to_all_single(outputs, inputs)
                cache.zero_()

            timer = CUDATimer()
            for _ in range(args.reps):
                with timer:
                    dist.all_to_all_single(outputs, inputs)
                cache.zero_()
            time_s = timer.get_mean_time_s()
            time_std_s = timer.get_std_time_s()
            gib = num_bytes / 2**30
            results["GiB"].append(gib)
            results["numel"].append(numel)
            results["time_s"].append(time_s)
            results["time_std_s"].append(time_std_s)
            results["GiB/s"].append(gib / time_s)

        if not rank:
            import pandas as pd

            df = pd.DataFrame.from_dict(results).reset_index(drop=True)
            print("\nRESULTS\n")
            print(df)
    finally:
        dist.destroy_process_group()
