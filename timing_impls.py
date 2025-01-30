from typing import Optional
import argparse

import torch
import torch.nn.functional as F

from mamba_ssm.modules.ssd_minimal import (
    ssd_minimal_discrete,
    ssd_minimal_discrete_alt,
    ssd_minimal_no_chunk_linear,
)
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from triton.testing import do_bench


def get_xdtABC(
    seqlen=4096,
    d_model=2560,
    expand=2,
    headdim=32,
    ngroups=1,
    dstate=8,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad: bool = False,
    batch_size: int = 1,
    nheads: Optional[int] = None,
):
    nheads = nheads or (d_model * expand) // headdim
    x = torch.randn(
        batch_size,
        seqlen,
        nheads,
        headdim,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    dt = F.softplus(
        torch.randn(
            batch_size,
            seqlen,
            nheads,
            dtype=dtype,
            device=device,
        )
        - 4
    )
    A = -torch.exp(
        torch.rand(
            nheads,
            dtype=dtype,
            device=device,
        )
    )
    if requires_grad:
        # Set dt and A as requires_grad, and not the tensors they're built from, so that they
        # are leaf tensors which accumulate gradients.
        dt.requires_grad_()
        A.requires_grad_()
    B = torch.randn(
        batch_size,
        seqlen,
        ngroups,
        dstate,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    C = torch.randn(
        batch_size,
        seqlen,
        ngroups,
        dstate,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return (
        x.contiguous(),
        dt.contiguous(),
        A.contiguous(),
        B.contiguous(),
        C.contiguous(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_mamba", action="store_true")

    args = parser.parse_args()
    x, dt, A, B, C = get_xdtABC(seqlen=args.seqlen, batch_size=args.batch_size)

    kernel_args = (x, dt, A, B, C, args.chunk_size)

    def kernel_args_wrapper(x, dt, A, B, C, chunk_size):
        return (x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    impl_dict = {
        "discrete": lambda *kernel_args: ssd_minimal_discrete(
            *kernel_args_wrapper(*kernel_args)
        ),
        "discrete_alt": lambda *kernel_args: ssd_minimal_discrete_alt(
            *kernel_args_wrapper(*kernel_args)
        ),
        "no_chunk_linear": lambda *kernel_args: ssd_minimal_no_chunk_linear(
            *kernel_args_wrapper(*kernel_args)[:-1]
        ),
    }
    if not args.no_mamba:
        impl_dict["mamba_ssm"] = mamba_chunk_scan_combined

    results: dict[str, float] = {}
    for name, impl in impl_dict.items():
        # Don't compile the mamba impl
        if name != "mamba_ssm" and args.compile:
            impl = torch.compile(impl, fullgraph=True, mode="max-autotune")
            name += "_compile"

        bench_fn = lambda: impl(*kernel_args)
        if name != "mamba_ssm" and args.compile:
            # Compile
            bench_fn()

        mean_time_ms = do_bench(bench_fn, warmup=args.warmup, rep=args.rep)
        print(f"Mean time {name}: {mean_time_ms:.2f} ms")
        results[name] = mean_time_ms

    min_time = min(results.values())
    print("\nRelative times:")
    for name, r in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {r / min_time}")
