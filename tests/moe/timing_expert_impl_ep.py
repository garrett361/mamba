import os
from argparse import ArgumentParser
from datetime import timedelta

import torch
from timing_utils import (
    CUDATimer,
    SequentialProfileModule,
    get_ep_mesh,
    shard_sequential_model,
)
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy

from mamba_ssm.modules.moe import EP_EXPERT_CLASSES, MoE

if __name__ == "__main__":
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    parser = ArgumentParser()
    parser.add_argument("--sharding_strategy", choices=["fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--act_ckpt", action="store_true")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--d_intermediate", type=int, default=1344)
    parser.add_argument(
        "--ep_degrees", type=str, default=f"1,{torch.cuda.device_count()}"
    )
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--n_activated_experts", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_routed_experts", type=int, default=64)
    parser.add_argument("--n_shared_experts", type=int, default=0)
    parser.add_argument("--no_bwd", action="store_true")
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--impls", type=str, default=",".join(EP_EXPERT_CLASSES))

    args = parser.parse_args()

    impls = args.impls.split(",")
    ep_degrees = [int(e) for e in args.ep_degrees.split(",")]
    if not rank:
        print(f"--> running with these configs {args}")

    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    # Clear cache, like triton do_bench
    cache_size_bytes = 512 * 2**20  # 512k
    cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

    iters_per_impl = args.warmups + args.reps
    torch.manual_seed(42 + rank)
    inputs = torch.randn(
        iters_per_impl, args.bsz * args.seqlen, args.in_features, **factory_kwargs
    )
    results = {}

    # some setups
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        if args.sharding_strategy == "hsdp":
            fsdp_mesh = init_device_mesh(
                "cuda",
                (
                    world_size // torch.cuda.device_count(),
                    torch.cuda.device_count(),
                ),
                mesh_dim_names=("outer", "inner"),
            )
        else:
            fsdp_mesh = init_device_mesh(
                "cuda", (world_size,), mesh_dim_names=("fsdp",)
            )
        for ep_degree in ep_degrees:
            ep_mesh = get_ep_mesh(ep_degree, world_size)
            for impl in impls:
                model = SequentialProfileModule(
                    [
                        MoE(
                            in_features=args.in_features,
                            d_intermediate=args.d_intermediate,
                            n_routed_experts=args.n_routed_experts,
                            n_shared_experts=args.n_shared_experts,
                            n_activated_experts=args.n_activated_experts,
                            ep_mesh=None if ep_mesh is None else ep_mesh["inner"],
                            moe_impl=impl,
                        )
                        for _ in range(args.n_layer)
                    ]
                )

                if args.act_ckpt:
                    if not rank:
                        print("Applying activation checkpointing")
                    for layer_idx, moe in enumerate(model):
                        model[layer_idx] = checkpoint_wrapper(
                            moe, preserve_rng_state=False
                        )

                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
                )
                shard_sequential_model(
                    seq_model=model,
                    ep_degree=ep_degree,
                    world_size=world_size,
                    ep_mesh=ep_mesh,
                    fsdp_mesh=fsdp_mesh,
                    mp_policy=mp_policy,
                )

                idx = 0
                for _ in range(args.warmups):
                    out = model(inputs[idx])
                    if not args.no_bwd:
                        out.pow(2).sum().backward()
                    cache.zero_()
                    idx += 1

                timer = CUDATimer()
                for _ in range(args.reps):
                    with timer:
                        out = model(inputs[idx])
                        if not args.no_bwd:
                            out.pow(2).sum().backward()
                    cache.zero_()
                    idx += 1
                time_s = timer.get_mean_time_s()
                time_std_s = timer.get_std_time_s()
                impl_name = f"{impl}_ep_{ep_degree}"
                if not rank:
                    print(
                        f"{impl_name}: {time_s=:.2e}, {time_std_s=:.2e}, {time_std_s/time_s=:.2e}"
                    )
                    results[impl_name] = time_s
                del model

    finally:
        if not rank:
            print("Relative times:")
            min_time = min(results.values())
            for impl_name, time_s in results.items():
                print(f"\t{impl_name}: {time_s / min_time:.2e}")
        dist.destroy_process_group()
