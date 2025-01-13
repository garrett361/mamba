from typing import Optional

import torch
import torch.nn.functional as F

from mamba_ssm.modules.ssd_minimal import (
    ssd_minimal_discrete,
    ssd_minimal_discrete_alt,
    ssd_minimal_discrete_alt_slow,
)
from torch.profiler import profile, ProfilerActivity


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
    return x, dt, A, B, C


if __name__ == "__main__":
    seqlen = 4096
    chunk_size = 32
    x, dt, A, B, C = get_xdtABC(seqlen=seqlen)
    y_alt = ssd_minimal_discrete_alt(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
    y_discrete, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    args = (x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
    for impl in (
        ssd_minimal_discrete_alt_slow,
        ssd_minimal_discrete_alt,
        ssd_minimal_discrete,
    ):
        for warmup in range(5):
            impl(*args)
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            impl(*args)
        print(
            f"{impl.__name__}: ",
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
        )
        prof.export_chrome_trace(f"traces/{impl.__name__}_trace.json")
