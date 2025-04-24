from argparse import ArgumentParser

import torch
from timing_utils import CUDATimer

from mamba_ssm.modules.moe import _get_counts
from mamba_ssm.ops.triton.moe import pad_sorted_idxs


def fn(
    counts: torch.Tensor, n_toks: int, alignment: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Torch impl:
    counts_aligned = ((counts + alignment - 1) // alignment) * alignment
    offs = counts_aligned.cumsum(dim=0, dtype=torch.int32)

    # Build the aligned index map
    idxs_offs_align = (counts_aligned - counts).roll(1)
    idxs_offs_align[0] = 0
    idxs_offs_align = idxs_offs_align.cumsum(dim=0)
    idxs = torch.arange(counts.sum(), device=counts.device)
    idxs = idxs + idxs_offs_align.repeat_interleave(counts)
    return idxs, offs


impls = {"torch": fn, "torch_compile": torch.compile(fn), "triton": pad_sorted_idxs}


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
    parser.add_argument("--no_bwd", action="store_true")

    parser.add_argument("--alignment", type=int, default=4096)
    args = parser.parse_args()
    print(f"{args=}")

    indices = (
        torch.randn(
            args.seqlen,
            args.n_routed_experts,
            device="cuda",
        )
        .topk(args.n_activated_experts, dim=-1)
        .indices
    )
    counts = _get_counts(indices, args.n_routed_experts)
    n_toks = args.bsz * args.seqlen

    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    # Clear cache, like triton do_bench
    cache_size_bytes = 512 * 2**20  # 512k
    cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

    results = {}
    for name, impl in impls.items():
        for _ in range(args.warmups):
            impl(counts, n_toks, args.alignment)
            cache.zero_()

        timer = CUDATimer()
        for _ in range(args.reps):
            with timer:
                impl(counts, n_toks, args.alignment)
            cache.zero_()
        time_s = timer.get_mean_time_s()
        time_std_s = timer.get_std_time_s()
        print(f"{name}: {time_s=:.2e}, {time_std_s=:.2e}, {time_std_s/time_s=:.2e}")
        results[name] = time_s

    print("Relative times:")
    min_time = min(results.values())
    for cls, time_s in results.items():
        print(f"\t{cls}: {time_s / min_time:.2e}")
