from argparse import ArgumentParser
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import Gate


class MoEBase(nn.Module):
    """
    Base impl from Deepseek (more or less)
    """

    def __init__(
        self,
        in_features: int,
        d_intermediate: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        multiple_of: int = 1,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        if ep_mesh is not None and n_routed_experts % ep_mesh.size():
            # TODO: @goon - shouldn't be a necessary constraint. Move to torch.chunk semantics for
            # placements.
            raise ValueError(
                f"{self.n_routed_experts=} must be divisible by {ep_mesh.size()=}"
            )
        if ep_mesh is not None and ep_mesh.ndim != 1:
            raise ValueError(
                f"The expert parallel mesh must be one-dimensional: {ep_mesh.ndim=}"
            )

        super().__init__()
        self.in_features = in_features
        self.d_intermediate = d_intermediate
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.multiple_of = multiple_of
        self.score_func = score_func
        self.route_scale = route_scale
        self.ep_mesh = ep_mesh
        self.n_activated_experts = n_activated_experts

        factory_kwargs = {"device": device, "dtype": dtype}

        self.ep_mesh_size = 1 if ep_mesh is None else ep_mesh.size()
        self.n_local_experts = self.n_routed_experts // (
            self.ep_mesh.size() if self.ep_mesh is not None else 1
        )

        self.experts_start_idx = (
            0 if ep_mesh is None else ep_mesh.get_local_rank() * self.n_local_experts
        )
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func=self.score_func,
            route_scale=self.route_scale,
            **factory_kwargs,
        )
        self.experts = nn.ModuleDict(
            {
                str(i): GatedMLP(
                    self.in_features,
                    self.d_intermediate,
                    multiple_of=self.multiple_of,
                    **factory_kwargs,
                )
                for i in range(self.experts_start_idx, self.experts_end_idx)
            }
        )
        self.shared_experts = (
            GatedMLP(
                self.in_features,
                self.n_shared_experts * self.d_intermediate,
                multiple_of=self.multiple_of,
                **factory_kwargs,
            )
            if self.n_shared_experts
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        x_shape = x.size()
        x = x.view(-1, self.in_features)

        weights, indices = self.gate(x)

        z = torch.zeros_like(x)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # TODO: @goon - handle no-tokens edge case
            expert = self.experts[str(i)]
            idx, top = torch.where(indices == i)
            z[idx] += expert(x[idx]) * weights[idx, top, None]

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)


class MoEAlt(MoEBase):
    """
    Alt fwd impl
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        x_shape = x.size()
        x = x.view(-1, self.in_features)

        weights, indices = self.gate(x)

        z = torch.empty(
            x.shape[0] * self.n_activated_experts,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

        for exp_idx in range(self.n_routed_experts):
            idxs = indices == exp_idx
            # TODO: @goon - handle no-tokens edge case
            z[idxs.flatten()] = self.experts[str(exp_idx)](x[idxs.any(dim=-1)])

        # Store the unsorted results back in x_by_expert
        z = z.reshape(*(weights.shape + z.shape[-1:]))
        z = torch.bmm(weights[:, None], z).squeeze(1)

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)


class CUDATimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._stop_events: list[torch.cuda.Event] = []

    def __enter__(self) -> "CUDATimer":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._stop_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._stop_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(stop) / 1e3
            for start, stop in zip(self._start_events, self._stop_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def get_std_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return torch.tensor(time_list_s).std().item()

    def reset(self) -> None:
        self._start_events.clear()
        self._stop_events.clear()


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

    # Clear cache, like triton do_bench
    cache_size_bytes = 512 * 2**20  # 512k
    cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

    for model_cls in (MoEBase, MoEAlt):
        model = model_cls(
            in_features=args.in_features,
            d_intermediate=args.d_intermediate,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
            n_activated_experts=args.n_activated_experts,
            dtype=dtype,
            device=device,
        )
        inputs = torch.randn(
            args.bsz, args.seqlen, args.in_features, dtype=dtype, device=device
        )

        for _ in range(args.warmups):
            out = model(inputs)
            if args.bwd:
                out.sum().backward()
            cache.zero_()

        timer = CUDATimer()
        for _ in range(args.reps):
            with timer:
                out = model(inputs)
                if args.bwd:
                    out.sum().backward()
            cache.zero_()
        time_s = timer.get_mean_time_s()
        time_std_s = timer.get_std_time_s()
        print(f"{model_cls.__name__}: {time_s=}, {time_std_s=}")
