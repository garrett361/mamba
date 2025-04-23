from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import (
    RoutedExpertsNoEPForLoopAlt,
    RoutedExpertsNoEPGroupedMM,
    RoutedExpertsNoEPForLoop,
    _SimpleRoutedExperts,
    _get_single_exp_output
)





EXP_CLASSES = {
    "simple": _SimpleRoutedExperts,
    "torch": RoutedExpertsNoEPForLoop,
    "torch_alt": RoutedExpertsNoEPForLoopAlt,
    "torch_gemm": RoutedExpertsNoEPGroupedMM,
}


if __name__ == "__main__":
    # some setups
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--in_features", type=int, default=3072)
    parser.add_argument("--d_intermediate", type=int, default=1344)
    parser.add_argument("--n_routed_experts", type=int, default=64)
    parser.add_argument("--n_activated_experts", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trace_dir", default=None)
    parser.add_argument(
        "--impl", choices=["torch", "torch_gemm", "simple"], default="torch"
    )
    parser.add_argument("--bwd", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    torch.cuda.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    model = EXP_CLASSES[args.impl](
        in_features=args.in_features,
        d_intermediate=args.d_intermediate,
        n_routed_experts=args.n_routed_experts,
        **factory_kwargs,
    )
    print(f"{model=}")
    print(f"{sum(p.numel() for p in model.parameters())/2**30=}")

    inputs = torch.randn(
        args.batch_size * args.seqlen, args.in_features, **factory_kwargs
    )
    weights = torch.randn(
        args.batch_size * args.seqlen,
        args.n_activated_experts,
        **factory_kwargs,
    )
    indices = (
        torch.randn(
            args.batch_size * args.seqlen,
            args.n_routed_experts,
            device=device,
        )
        .topk(args.n_activated_experts, dim=-1)
        .indices
    )

    for _ in range(args.warmups):
        out = model(inputs, weights, indices)
        if args.bwd:
            out.pow(2).sum().backward()
            model.zero_grad()

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("fwd"):
            out = model(inputs, weights, indices)
        if args.bwd:
            with record_function("bwd"):
                out.pow(2).sum().backward()
            model.zero_grad()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    if args.trace_dir is not None:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        import json

        with open(trace_dir.joinpath(f"args_{args.impl}.json"), "w") as fp:
            json.dump(vars(args), fp)
        prof.export_chrome_trace(str(trace_dir.joinpath(f"trace_{args.impl}.json")))
