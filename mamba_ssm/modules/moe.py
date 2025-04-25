from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional
from torch.profiler import ProfilerActivity, profile, record_function

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.moe import pad_sorted_idxs, pad_sorted_idxs_torch

_GROUPED_MM_ALIGNMENT = 16


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        in_features (int): Dimensionality of input features.
        n_routed_experts (int): Number of experts available for token routing.
        n_activated_experts (int): Number of top experts activated for each input.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(
        self,
        in_features: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        factory_kwargs = {"device": device, "dtype": dtype}

        self.lin = nn.Linear(
            self.in_features, self.n_routed_experts, bias=False, **factory_kwargs
        )
        # Fix bias usage
        self.bias = (
            nn.Parameter(torch.empty(self.n_routed_experts, **factory_kwargs))
            if self.in_features == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = self.lin(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise ValueError(f"Unexpected {self.score_func=} not in (softmax, sigmoid)")
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_expert_groups > 1:
            scores = scores.view(x.size(0), self.n_expert_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                # NOTE: @goon - this line seems odd, but is standard: sum the top-2 scores from each
                # group to create the group score.
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.n_limited_groups, dim=-1)[1]
            mask = scores.new_ones(
                x.size(0), self.n_expert_groups, dtype=bool
            ).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features},"
            f" n_routed_experts={self.n_routed_experts},"
            f" n_activated_experts={self.n_activated_experts},"
            f" n_expert_groups={self.n_expert_groups},"
            f" n_limited_groups={self.n_limited_groups},"
            f" score_func={self.score_func})"
        )


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        in_features (int): Dimensionality of input features.
        d_intermediate (int): Dimensionality of hidden features of each expert.
        n_routed_experts (int): Total number of experts in the model.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
        ep_mesh (Optional[DeviceMesh]): 1D device mesh for expert parallel, if desired.
    """

    def __init__(
        self,
        in_features: int,
        d_intermediate: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        multiple_of: int = 1,
        activation=F.silu,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
        moe_impl: Literal[
            "torch", "torch_gemm", "torch_gemm_triton", "_simple"
        ] = "torch",
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
        self.activation = activation
        self.score_func = score_func
        self.route_scale = route_scale
        self.ep_mesh = ep_mesh
        self.n_activated_experts = n_activated_experts
        self._tok_count = 0

        self.ep_mesh_size = self.ep_mesh.size() if self.ep_mesh is not None else 1
        self.n_local_experts = self.n_routed_experts // (self.ep_mesh_size)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.gate = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func=self.score_func,
            route_scale=self.route_scale,
            **factory_kwargs,
        )

        # TODO: @goon - better config for which routed experts impl to use.

        routed_experts_cls = (
            NON_EP_EXPERT_CLASSES[moe_impl]
            if ep_mesh is None
            else EP_EXPERT_CLASSES[moe_impl]
        )
        self.experts = routed_experts_cls(
            in_features=self.in_features,
            d_intermediate=self.d_intermediate,
            n_routed_experts=self.n_routed_experts,
            multiple_of=self.multiple_of,
            activation=self.activation,
            ep_mesh=self.ep_mesh,
            **factory_kwargs,
        )
        self.shared_experts = (
            GatedMLP(
                self.in_features,
                self.n_shared_experts * self.d_intermediate,
                multiple_of=self.multiple_of,
                activation=self.activation,
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
        z = self.experts(x, weights, indices)

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)


def _get_counts(indices: torch.LongTensor, n_routed_experts: int) -> torch.LongTensor:
    """
    Turns the `indices` tensor into a sorted count of the number of tokens per expert.
    """
    with torch.no_grad():
        counts = indices.new_zeros((indices.shape[0], n_routed_experts))
        counts.scatter_(1, indices, 1)
        counts = counts.sum(dim=0)
    return counts


def _get_single_exp_output(
    x: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Compute the outputs from a single expert.
    """
    y = F.linear(x, fc1_weights)
    y, gate = y.chunk(2, dim=-1)
    y = y * activation(gate)
    y = F.linear(y, fc2_weights)
    return y


def _get_exp_outputs_grouped_mm(
    x: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    offs: torch.IntTensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Compute the outputs from all experts using torch._grouped_mm
    """
    y = torch._grouped_mm(
        x, fc1_weights.transpose(-2, -1), offs=offs, out_dtype=x.dtype
    )
    y, gate = y.chunk(2, dim=-1)
    y = y * activation(gate)
    y = torch._grouped_mm(
        y, fc2_weights.transpose(-2, -1), offs=offs, out_dtype=x.dtype
    )
    return y


class _RoutedExperts(nn.Module, ABC):
    def __init__(
        self,
        in_features: int,
        d_intermediate: int,
        n_routed_experts: int,
        multiple_of: int = 1,
        activation=F.silu,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ):
        """
        Analogous to a set of GatedMLP layers, but with single combined weights.

        TODO: @goon - state dict hooks so that the expert weights are individually saved and loadable?
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.d_intermediate = (
            (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        )
        self.n_routed_experts = n_routed_experts
        self.multiple_of = multiple_of
        self.activation = activation
        self.ep_mesh = ep_mesh

        self.ep_mesh_size = self.ep_mesh.size() if self.ep_mesh is not None else 1
        self.n_local_experts = self.n_routed_experts // (self.ep_mesh_size)
        self.experts_start_idx = (
            0 if ep_mesh is None else ep_mesh.get_local_rank() * self.n_local_experts
        )
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.fc1_weights = nn.Parameter(
            torch.randn(
                self.n_local_experts,
                2 * d_intermediate,
                in_features,
                **factory_kwargs,
            )
            / self.in_features ** (0.5)
        )
        self.fc2_weights = nn.Parameter(
            torch.randn(
                self.n_local_experts,
                self.in_features,
                self.d_intermediate,
                **factory_kwargs,
            )
            / self.d_intermediate ** (0.5)
        )

    @abstractmethod
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features},"
            f" d_intermediate={self.d_intermediate},"
            f" n_routed_experts={self.n_routed_experts},"
            f" n_local_experts={self.n_local_experts})"
        )


class _RoutedExpertsNoEP(_RoutedExperts):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.ep_mesh is None


class _RoutedExpertsTorchEP(_RoutedExperts):
    """
    Adds torch all-to-all EP routing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.ep_mesh is not None
        self._tok_count = 0  # for testing

    def _get_ep_toks_and_routing_data(
        self, x_by_expert: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        # [EP Routing and Indexing]
        # To perform EP routing with torch comms primitives, each EP rank needs to know how many
        # tokens EP rank `r` will be sending to its local expert `l`. This can be done by exchanging
        # information about each rank's `counts`.
        #
        # Each rank then sorts their tokens in order of global expert idx (x_by_expert) and sends as
        # appropriate. The received tokens are **not** in expert order: they are first ordered by
        # the sending rank, and then by local order. Schematically: recv ~ cat([recv_from_rank_0] +
        # [recv_from_rank_1] + ...) where recv_from_rank_r will contain the tokens for each of the L
        # local experts sorted by local expert order.
        #
        # In order to use GEMM kernels, the received tokens must be re-sorted in local expert
        # order, so that tokens belonging to the same local expert are all contiguous. This is a
        # data-dependent resorting and it does not appear possible to implement this with torch
        # primitives without incurring a CUDA sync.

        # Get counts of incoming tensors. tokens_per_expert_group.reshape(self.ep_mesh.size(),
        # self.n_local_experts)[r, l] = num tokens rank r sent to local expert l

        counts = _get_counts(indices, self.n_routed_experts)
        assert self.ep_mesh is not None  # mypy
        with record_function("all2all::tok_per_exp_grp"):
            tokens_per_expert_group = funcol.all_to_all_single(
                counts, None, None, group=self.ep_mesh
            )

        # We need the list version of the counts due to NCCL signatures. This incurs a CUDA sync.
        # TODO: avoid https://github.com/NVIDIA/nccl/issues/1648
        send_counts = (
            counts.reshape(self.ep_mesh_size, self.n_local_experts).sum(dim=1).tolist()
        )
        recv_counts = (
            tokens_per_expert_group.reshape(self.ep_mesh_size, self.n_local_experts)
            .sum(dim=1)
            .tolist()
        )
        self._tok_count += sum(recv_counts)  # testing

        # Receive toks from other workers
        import torch.distributed as dist

        dist.barrier()
        with record_function("all2all::send0"):
            x_recv = funcol.all_to_all_single_autograd(
                x_by_expert, recv_counts, send_counts, group=self.ep_mesh
            )
        return x_recv, send_counts, recv_counts, tokens_per_expert_group


class RoutedExpertsNoEPForLoop(_RoutedExpertsNoEP):
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        z = torch.zeros_like(x)
        # Note: for some reason, getting the weights with CPU integer indexing, like fc1 =
        # self.fc1_weights[exp_idx], results in super-slow CUDA syncs during the backwards pass.
        for exp_idx, (fc1, fc2) in enumerate(zip(self.fc1_weights, self.fc2_weights)):
            # TODO: @goon - handle no-tokens edge case
            # NOTE: @goon - torch.where incurs a CUDA sync.
            tok_idx, act_exp_idx = torch.where(indices == exp_idx)
            z[tok_idx] += (
                _get_single_exp_output(x[tok_idx], fc1, fc2, self.activation)
                * weights[tok_idx, act_exp_idx, None]
            )
        return z


class RoutedExpertsNoEPGroupedMM(_RoutedExpertsNoEP):
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        # Sort tokens by the expert they are indexed to.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        n_activated_experts = indices.shape[-1]
        x_by_expert = x[flat_sorted_indices // n_activated_experts]

        counts = _get_counts(indices, self.n_routed_experts)

        idxs_align, offs = pad_sorted_idxs_torch(
            counts, indices.numel(), _GROUPED_MM_ALIGNMENT
        )
        # NOTE: @goon - offs[-1] induces a very long CUDA sync. Can avoid it by letting z be a fixed
        # sized, empirically tuned to be large enough to handle the routed tokens.
        z = torch.empty(offs[-1], x.shape[-1], dtype=x.dtype, device=x.device)
        z[idxs_align] = x_by_expert
        z = _get_exp_outputs_grouped_mm(
            z, self.fc1_weights, self.fc2_weights, offs, self.activation
        )

        # Remove the alignment and return the tokens to their original ordering
        x_by_expert[flat_sorted_indices] = z[idxs_align]

        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z


class RoutedExpertsNoEPGroupedMMTriton(_RoutedExpertsNoEP):
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        # Sort tokens by the expert they are indexed to.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        n_activated_experts = indices.shape[-1]
        x_by_expert = x[flat_sorted_indices // n_activated_experts]

        counts = _get_counts(indices, self.n_routed_experts)

        # Expand the tokens to an appropriately aligned tensor.
        idxs_align, offs = pad_sorted_idxs(
            counts, indices.numel(), _GROUPED_MM_ALIGNMENT
        )
        # NOTE: @goon - offs[-1] induces a very long CUDA sync. Can avoid it by letting z be a fixed
        # sized, empirically tuned to be large enough to handle the routed tokens.
        z = torch.empty(offs[-1], x.shape[-1], dtype=x.dtype, device=x.device)
        # z = torch.empty(
        #     ((int(1.5 * indices.numel()) + 31) // 32) * 32,
        #     x.shape[-1],
        #     dtype=x.dtype,
        #     device=x.device,
        # )

        z[idxs_align] = x_by_expert
        z = _get_exp_outputs_grouped_mm(
            z, self.fc1_weights, self.fc2_weights, offs, self.activation
        )

        # Remove the alignment and return the tokens to their original ordering
        x_by_expert[flat_sorted_indices] = z[idxs_align]

        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z


class RoutedExpertsTorchEPForLoop(_RoutedExpertsTorchEP):
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        # Sort tokens by the expert they are indexed to.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        n_activated_experts = indices.shape[-1]
        x_by_expert = x[flat_sorted_indices // n_activated_experts]

        x_recv, send_counts, recv_counts, tokens_per_expert_group = (
            self._get_ep_toks_and_routing_data(x_by_expert, indices)
        )

        # Prepare outputs
        x_send = torch.empty_like(x_recv)

        # Need to know which idxs in x_recv correspond to which local experts. Can derive from
        # tokens_per_expert_group.
        local_expert_idxs = (
            torch.arange(
                tokens_per_expert_group.numel(),
                device=tokens_per_expert_group.device,
            )
            % self.n_local_experts
        )
        # NOTE: @goon - repeat_interleave incurs a CUDA sync since it needs to wait on
        # the CUDA tensor tokens_per_expert_group to know the output shape
        local_expert_idxs = local_expert_idxs.repeat_interleave(tokens_per_expert_group)

        for exp_idx, (fc1_weight, fc2_weight) in enumerate(
            zip(self.fc1_weights, self.fc2_weights)
        ):
            idxs = local_expert_idxs == exp_idx
            # TODO: @goon - handle no-tokens edge case
            x_send[idxs] = _get_single_exp_output(
                x_recv[idxs], fc1_weight, fc2_weight, self.activation
            )

        # Send results back to original ranks (reversed send/recv count data)
        with record_function("all2all::send1"):
            x_out = funcol.all_to_all_single_autograd(
                x_send, send_counts, recv_counts, group=self.ep_mesh
            )

        # Save an allocation: store the unsorted results back in x_by_expert.
        x_by_expert[flat_sorted_indices] = x_out
        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x_by_expert.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z


class RoutedExpertsTorchEPGroupedMM(_RoutedExpertsTorchEP):
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
    ) -> torch.Tensor:
        # Sort tokens by the expert they are indexed to.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        n_activated_experts = indices.shape[-1]
        x_by_expert = x[flat_sorted_indices // n_activated_experts]

        x_recv, send_counts, recv_counts, tokens_per_expert_group = (
            self._get_ep_toks_and_routing_data(x_by_expert, indices)
        )

        # Need to know which idxs in x_recv correspond to which local experts. Can derive from
        # tokens_per_expert_group.
        local_expert_idxs = (
            torch.arange(
                tokens_per_expert_group.numel(),
                device=tokens_per_expert_group.device,
            )
            % self.n_local_experts
        )
        # NOTE: @goon - repeat_interleave incurs a CUDA sync since it needs to wait on
        # the CUDA tensor tokens_per_expert_group to know the output shape
        local_expert_idxs = local_expert_idxs.repeat_interleave(tokens_per_expert_group)

        # Sort and pad to use grouped GEMM
        recv_sorted_indices = local_expert_idxs.argsort(dim=-1)
        x_recv_sorted = x_recv[recv_sorted_indices]
        idxs_align, offs = pad_sorted_idxs(
            tokens_per_expert_group.view(-1, self.n_local_experts).sum(dim=0),
            x_recv.shape[0],
            _GROUPED_MM_ALIGNMENT,
        )
        z = torch.empty(offs[-1], x.shape[-1], dtype=x.dtype, device=x.device)
        z[idxs_align] = x_recv_sorted
        z = _get_exp_outputs_grouped_mm(
            z, self.fc1_weights, self.fc2_weights, offs, self.activation
        )
        # Remove the alignment and return the tokens to their original ordering. Re-use
        # x_recv_sorted and avoid a new allocation.
        x_recv_sorted[recv_sorted_indices] = z[idxs_align]

        # Send results back to original ranks (reversed send/recv count data)
        print(f"{self.ep_mesh.get_local_rank()=}:\t{x_recv_sorted.shape=}\n\t{send_counts=}\n\t{sum(send_counts)=}\n\t{recv_counts=}\n\t{sum(recv_counts)=}")
        with record_function("all2all::send1"):
            x_out = funcol.all_to_all_single_autograd(
                x_recv_sorted, send_counts, recv_counts, group=self.ep_mesh
            )

        # Save an allocation: store the unsorted results back in x_by_expert.
        x_by_expert[flat_sorted_indices] = x_out
        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x_by_expert.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z


class _SimpleRoutedExperts(nn.Module):
    """
    Simple routed experts class mirroring the public DeepSeekv3 impl. Just for testing.
    """

    def __init__(
        self,
        in_features: int,
        d_intermediate: int,
        n_routed_experts: int,
        multiple_of: int = 1,
        activation=F.silu,
        ep_mesh: Optional[DeviceMesh] = None,
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


NON_EP_EXPERT_CLASSES = {
    "_simple": _SimpleRoutedExperts,
    "torch": RoutedExpertsNoEPForLoop,
    "torch_gemm": RoutedExpertsNoEPGroupedMM,
    "torch_gemm_triton": RoutedExpertsNoEPGroupedMMTriton,
}
EP_EXPERT_CLASSES = {
    "torch": RoutedExpertsTorchEPForLoop,
    "torch_gemm": RoutedExpertsTorchEPGroupedMM,
}
