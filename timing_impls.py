from typing import Optional

import torch
import torch.nn.functional as F

from mamba_ssm.modules.ssd_minimal import (
    ssd_minimal_discrete_alt,
)
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class CudaTimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._end_events: list[torch.cuda.Event] = []

    def __enter__(self) -> None:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._end_events.append(stop)

    def __exit__(self, *args, **kwargs) -> None:
        self._end_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(end) / 1e3
            for start, end in zip(self._start_events, self._end_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def reset(self) -> None:
        self._start_events.clear()
        self._end_events.clear()


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
    seqlen = 8192
    batch_size = 8
    chunk_size = 8
    warmups = 10
    iters = 25
    x, dt, A, B, C = get_xdtABC(seqlen=seqlen, batch_size=batch_size)

    args = (x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
    results = {}
    for impl in (ssd_minimal_discrete_alt,):
        for warmup in range(warmups):
            impl(*args)
        timer = CudaTimer()
        with timer:
            for iteration in range(iters):
                impl(*args)
        print(f"{impl.__name__}: {timer.get_mean_time_s()=}")
        results[impl.__name__] = timer.get_mean_time_s()

    impl = mamba_chunk_scan_combined
    args = (x, dt, A, B, C, chunk_size)
    for warmup in range(warmups):
        impl(*args)
    timer = CudaTimer()
    with timer:
        for iteration in range(iters):
            impl(*args)
    print(f"{impl.__name__}: {timer.get_mean_time_s()=}")
    results[impl.__name__] = timer.get_mean_time_s()

    min_time = min(results.values())
    for name, r in results.items():
        results[name] = r / min_time
    print(f"Relative times: {results}")
