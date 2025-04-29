import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from timing_utils import CUDATimer
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from mamba_ssm.modules.moe import MoE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_routed_experts", type=int, default=64)
    parser.add_argument("--n_activated_experts", type=int, default=8)
    parser.add_argument("--in_features", type=int, default=3072)
    parser.add_argument("--d_intermediate", type=int, default=1344)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--n_layers_max", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--force_equal_loads", action="store_true")
    parser.add_argument("--compute_tok_stats", action="store_true")

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
        inputs = torch.randn(
            args.batch_size,
            args.seqlen,
            args.in_features,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        # Clear cache, like triton do_bench
        cache_size_bytes = 512 * 2**20  # 512k
        cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        ep_mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
        results = defaultdict(list)
        for n_layers in range(1, args.n_layers_max + 1):
            results["n_layers"].append(n_layers)
            for ep in (True, False):
                if not rank:
                    print(f"\nRunning {ep=}, {n_layers=}")
                torch.manual_seed(42 + (rank if ep else 0))
                model = nn.Sequential(
                    *[
                        MoE(
                            in_features=args.in_features,
                            d_intermediate=args.d_intermediate,
                            n_routed_experts=args.n_routed_experts,
                            n_activated_experts=args.n_activated_experts,
                            n_shared_experts=0,
                            ep_mesh=ep_mesh if ep else None,
                            device="cuda",
                            dtype=dtype,
                        )
                        for _ in range(n_layers)
                    ]
                )

                if not ep and n_layers > 2:
                    for moe in model:
                        fully_shard(moe, mp_policy=mp_policy)
                    fully_shard(model, mp_policy=mp_policy)

                for _ in range(args.warmups):
                    model(inputs).sum().backward()
                    cache.zero_()
                timer = CUDATimer()
                for _ in range(args.reps):
                    with timer:
                        model(inputs).sum().backward()
                    cache.zero_()
                time_s = timer.get_mean_time_s()
                time_std_s = timer.get_std_time_s()
                results["time_s" + (" (EP)" if ep else " (FSDP)")].append(time_s)
                results["time_std_s" + (" (EP)" if ep else " (FSDP)")].append(
                    time_std_s
                )
                if ep and args.compute_tok_stats:
                    # Collecting stats for the number of tok per expert
                    tok_counts = [layer._tok_count for layer in model]
                    tok_counts_t = torch.tensor(
                        tok_counts, device="cuda", dtype=torch.bfloat16
                    )
                    tok_counts_gathered = (
                        [torch.empty_like(tok_counts_t) for _ in range(world_size)]
                        if not rank
                        else None
                    )
                    dist.gather(tok_counts_t, tok_counts_gathered, dst=0)
                    if not rank:
                        # Sanity check
                        tok_counts_gathered_t = torch.stack(tok_counts_gathered, dim=0)
                        print(
                            f"Tok per rank (row) per layer (col): {tok_counts_gathered_t}"
                        )
                        print(f"Tok sum over ranks: {tok_counts_gathered_t.sum(dim=0)}")
                        print(f"Tok std over ranks: {tok_counts_gathered_t.std(dim=0)}")

        if not rank:
            import pandas as pd

            df = pd.DataFrame.from_dict(results).reset_index(drop=True)
            print("\nRESULTS\n")
            print(df)
    finally:
        dist.destroy_process_group()
