from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from timing_utils import CUDATimer

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import RoutedExpertsNoEPGroupedMM, RoutedExpertsNoEPTorch


class SimpleRoutedExperts(nn.Module):
    """
    Simple routed experts class mirroring the public DeepSeekv3 impl.
    """

    def __init__(
        self,
        in_features: int,
        d_intermediate: int,
        n_routed_experts: int,
        multiple_of: int = 1,
        activation=F.silu,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.experts = nn.ModuleList(
            [
                GatedMLP(
                    in_features=in_features,
                    hidden_features=d_intermediate,
                    multiple_of=multiple_of,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_routed_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y


expert_classes = (
    SimpleRoutedExperts,
    RoutedExpertsNoEPTorch,
    RoutedExpertsNoEPGroupedMM,
)


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
    for model_cls in expert_classes:
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
        print(f"{model_cls.__name__}: {time_s=:.2e}, {time_std_s=:.2e}, {time_std_s/time_s=:.2e}")
        results[model_cls.__name__] = time_s

    print("Relative times:")
    min_time = min(results.values())
    for cls, time_s in results.items():
        print(f"\t{cls}: {time_s / min_time:.2e}")
