import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--row_limit", type=int, default=30)

    args = parser.parse_args()
    x, dt, A, B, C = get_xdtABC(seqlen=args.seqlen, batch_size=args.batch_size)

    kernel_args = (x, dt, A, B, C, args.chunk_size)
    impl_dict = {
        # "discrete": lambda *kernel_args: ssd_minimal_discrete(
        #     *kernel_args_wrapper(*kernel_args)
        # ),
        # "discrete_alt": lambda *kernel_args: ssd_minimal_discrete_alt(
        #     *kernel_args_wrapper(*kernel_args)
        # ),
        # "no_chunk_linear": lambda *kernel_args: ssd_minimal_no_chunk_linear(
        #     *kernel_args_wrapper(*kernel_args)[:-1]
        # ),
        "triton": mamba_chunk_scan_combined,
    }

    def kernel_args_wrapper(x, dt, A, B, C, chunk_size):
        return (x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    for name, impl in impl_dict.items():
        for warmup in range(5):
            impl(*kernel_args)
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            impl(*kernel_args)
        print(
            f"\n############# {name} #############\n",
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=args.row_limit
            ),
        )
        prof.export_chrome_trace(f"traces/{name}_trace.json")
