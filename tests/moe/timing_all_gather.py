import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.distributed as dist
from timing_utils import CUDATimer
from torch.distributed.fsdp import MixedPrecisionPolicy

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
        # 128MiB to 8 GiB all-gathered size
        for exp in range(27, 34):
            # Gathered num_bytes
            num_bytes = 2**exp
            if not rank:
                print(f"Launching: 2^{exp} elements")
            numel = num_bytes // world_size // dtype.itemsize
            inputs = torch.randn(
                numel,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            outputs = torch.stack(
                [torch.empty_like(inputs) for _ in range(world_size)], dim=0
            )

            for _ in range(args.warmups):
                dist.all_to_all_single(outputs, inputs)
                cache.zero_()

            timer = CUDATimer()
            for _ in range(args.reps):
                with timer:
                    dist.all_gather_into_tensor(outputs, inputs)
                cache.zero_()
            time_s = timer.get_mean_time_s()
            time_std_s = timer.get_std_time_s()
            gib = num_bytes / 2**30
            # Report the gathered sizes
            results["GiB"].append(gib)
            # results["numel"].append(numel)
            results["time_s"].append(time_s)
            # results["time_std_s"].append(time_std_s)
            # Use the standard algo factor when reporting bandwidth:
            # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allgather
            results["GiB/s"].append(gib * (world_size - 1) / world_size / time_s)

        if not rank:
            import pandas as pd

            df = pd.DataFrame.from_dict(results).reset_index(drop=True)
            print("\nRESULTS\n")
            print(df)
    finally:
        dist.destroy_process_group()
