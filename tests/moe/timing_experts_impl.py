from argparse import ArgumentParser

import torch
from timing_utils import CUDATimer

from mamba_ssm.modules.moe import RoutedExpertsNoEPGroupedMM, RoutedExpertsNoEPTorch

expert_clases = (RoutedExpertsNoEPTorch, RoutedExpertsNoEPGroupedMM)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--in_features", type=int, default=3072)
    parser.add_argument("--d_intermediate", type=int, default=1344)
    parser.add_argument("--n_routed_experts", type=int, default=64)
    parser.add_argument("--n_activated_experts", type=int, default=8)
    parser.add_argument("--n_shared_experts", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--bwd", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    # Clear cache, like triton do_bench
    cache_size_bytes = 512 * 2**20  # 512k
    cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

    inputs = torch.randn(args.bsz * args.seqlen, args.in_features, **factory_kwargs)
    weights = torch.randn(
        args.bsz * args.seqlen,
        args.n_activated_experts,
        **factory_kwargs,
    )
    # Even spread of indices:
    indices = (
        torch.randn(
            args.bsz * args.seqlen,
            args.n_routed_experts,
            device=device,
        )
        .topk(args.n_activated_experts, dim=-1)
        .indices
    )
    results = {}
    for model_cls in (RoutedExpertsNoEPTorch, RoutedExpertsNoEPGroupedMM):
        model = model_cls(
            in_features=args.in_features,
            d_intermediate=args.d_intermediate,
            n_routed_experts=args.n_routed_experts,
            **factory_kwargs,
        )

        for _ in range(args.warmups):
            out = model(inputs, weights, indices)
            if args.bwd:
                out.pow(2).sum().backward()
            cache.zero_()

        timer = CUDATimer()
        for _ in range(args.reps):
            with timer:
                out = model(inputs, weights, indices)
                if args.bwd:
                    out.pow(2).sum().backward()
            cache.zero_()
        time_s = timer.get_mean_time_s()
        time_std_s = timer.get_std_time_s()
        print(f"{model_cls.__name__}: {time_s=}, {time_std_s=}")
        results[model_cls.__name__] = time_s

    print("Relative times:")
    min_time = min(results.values())
    for cls, time_s in results.items():
        print(f"\t{cls}: {time_s / min_time}")
