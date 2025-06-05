import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.profiler import record_function

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.moe import pad_sorted_idxs, pad_sorted_idxs_torch

_GROUPED_MM_ALIGNMENT = 16


class TokenCounter(nn.Module):
    """
    Turns the `indices` tensor into a sorted count of the number of tokens per expert. Implemented as
    a module so that we can easily attach hooks for token statistics inspection.
    """

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(
        self, indices: torch.LongTensor, n_routed_experts: int
    ) -> torch.IntTensor:
        assert indices.ndim == 2
        counts = indices.new_zeros((indices.shape[0], n_routed_experts))
        counts.scatter_(1, indices, 1)
        counts = counts.sum(dim=0, dtype=torch.int32)
        return counts


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
        bias: bool = False,
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
        self.register_buffer(
            "bias",
            torch.empty(self.n_routed_experts, **factory_kwargs) if bias else None,
        )

        self.tok_counter = TokenCounter()
        self.reset_parameters()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.IntTensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.LongTensor, torch.IntTensor]: Routing weights, selected expert
            indices, and the token count per expert.
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
            # NOTE: @goon -  Not sure why there is branching logic; from DSv3 github.
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                # NOTE: @goon - this line seems odd, but is standard: sum the top-2 scores from each
                # group to create the group score. Appears in both HF and DSv3 repos.
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.n_limited_groups, dim=-1)[1]
            mask = scores.new_ones(
                x.size(0), self.n_expert_groups, dtype=bool
            ).scatter_(1, indices, False)
            # NOTE: @goon -  using the in-place masked_fill_ gives in-place backwards pass errors.
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        # The bias, when it exists, is only used for routing decisions, not in the weight computation.
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return (
            weights.type_as(x),
            indices,
            self.tok_counter(indices, self.n_routed_experts),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features},"
            f" n_routed_experts={self.n_routed_experts},"
            f" n_activated_experts={self.n_activated_experts},"
            f" n_expert_groups={self.n_expert_groups},"
            f" n_limited_groups={self.n_limited_groups},"
            f" score_func={self.score_func})"
        )

    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # self.lin: nn.Linear can reset its own parameters already. Intentionally not resetting.


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
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        gate_bias: bool = False,
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
                f"{n_routed_experts=} must be divisible by {ep_mesh.size()=}"
            )
        if ep_mesh is not None and ep_mesh.ndim != 1:
            raise ValueError(f"{ep_mesh} must be one-dimensional")

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

        self.ep_mesh_size = self.ep_mesh.size() if self.ep_mesh is not None else 1
        self.n_local_experts = self.n_routed_experts // (self.ep_mesh_size)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.gate = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            score_func=self.score_func,
            route_scale=self.route_scale,
            bias=gate_bias,
            **factory_kwargs,
        )

        # TODO: @goon - better config for which routed experts impl to use.

        routed_experts_cls = (
            NON_EP_EXPERT_CLASSES_AND_SIMPLE[moe_impl]
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

        weights, indices, counts = self.gate(x)
        z = self.experts(x, weights, indices, counts)

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)


def _sort_by_exp_idx(
    x: torch.Tensor,
    indices: torch.LongTensor,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """
    Take tensor x of shape (n_tok, d_model) and corresponding expert indices of shape (n_tok,
    n_activated_experts) or (n_tok * n_activated_experts), and return a (n_tok *
    n_activated_experts, d_model) shaped tensor sorted by expert idx along with the sorted index
    tensor.
    """

    if indices.ndim == 1:
        # This path occurs in EP where indices are already flattened.
        flat_sorted_indices = indices.argsort(dim=-1)
        x_by_expert = x[flat_sorted_indices]
    else:
        # This path occurs with no EP.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        n_activated_experts = indices.shape[-1]
        x_by_expert = x[flat_sorted_indices // n_activated_experts]
    return x_by_expert, flat_sorted_indices


def _get_single_exp_output(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Compute the outputs from a single expert.
    """
    # NOTE: @goon - When the routed experts are ignored in fully_shard, their dtype doesn't get
    # converted with the fsdp mp_policy, so we ensure conversion here.
    y = F.linear(x, fc1_weight.to(x))
    y, gate = y.chunk(2, dim=-1)
    y = y * activation(gate)
    y = F.linear(y, fc2_weight.to(y))
    return y


def _get_exp_outputs_grouped_mm(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    offsets: torch.IntTensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Compute the outputs from all experts using torch._grouped_mm
    """
    # NOTE: @goon - When the routed experts are ignored in fully_shard, their dtype doesn't get
    # converted with the fsdp mp_policy, so we ensure conversion here.
    y = torch._grouped_mm(
        x, fc1_weight.to(x).transpose(-2, -1), offs=offsets, out_dtype=x.dtype
    )
    y, gate = y.chunk(2, dim=-1)
    y = y * activation(gate)
    y = torch._grouped_mm(
        y, fc2_weight.to(y).transpose(-2, -1), offs=offsets, out_dtype=x.dtype
    )
    return y


def _get_exp_outputs_titan_cg_gemm(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    expert_indices: torch.IntTensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
    group_size_m: int = 128,
):
    """
    Compute the outputs from all experts using titan's CG grouped gemm kernels.

    TODO: @goon - finish. "ModuleNotFoundError: No module named 'tma_cuda_autotune'" errors as of
    May 15
    """
    from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_backward import (
        cg_grouped_gemm,
    )

    # Only intended to operate on locally available shards for EP:
    if isinstance(fc1_weight, DTensor):
        fc1_weight = fc1_weight.to_local()
    if isinstance(fc2_weight, DTensor):
        fc2_weight = fc2_weight.to_local()

    y = cg_grouped_gemm(
        x, fc1_weight, expert_indices=expert_indices, group_size_m=group_size_m
    )
    y, gate = y.chunk(2, dim=-1)
    y = y * activation(gate)
    y = cg_grouped_gemm(
        y, fc2_weight, expert_indices=expert_indices, group_size_m=group_size_m
    )
    return y


def _get_local_indices_and_counts(
    tokens_per_expert_group: torch.Tensor, n_local_experts: int
) -> tuple[torch.IntTensor, torch.IntTensor]:
    """
    Convert the routing information in tokens_per_expert_group to a tensor which maps token position
    to local expert idx.
    """
    local_expert_idxs = (
        torch.arange(
            tokens_per_expert_group.numel(),
            device=tokens_per_expert_group.device,
            dtype=torch.int32,
        )
        % n_local_experts
    )
    # NOTE: @goon - repeat_interleave incurs a CUDA sync since it needs to wait on
    # the CUDA tensor tokens_per_expert_group to know the output shape
    local_expert_idxs = local_expert_idxs.repeat_interleave(tokens_per_expert_group)

    # Compute tok count per local expert
    local_expert_counts = tokens_per_expert_group.reshape(-1, n_local_experts).sum(
        dim=0, dtype=torch.int32
    )
    return local_expert_idxs, local_expert_counts


class RoutedExpertsWeights(nn.Module):
    """
    Container for routed expert weights. Convenient for aligning FQN's with other mamba_ssm
    conventions (thereby ensuring these weight are initalized by _init_weights.) and handling EP via
    DTensor, which is necessary for easy DCP integration.
    """

    def __init__(
        self,
        n_routed_experts: int,
        out_features: int,
        in_features: int,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.out_features = out_features
        self.in_features = in_features
        self.ep_mesh = ep_mesh
        if ep_mesh is None:
            data = torch.empty(
                n_routed_experts, out_features, in_features, device=device, dtype=dtype
            )
        else:
            data = torch.empty(
                n_routed_experts // ep_mesh.size(),
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
            data = DTensor.from_local(
                data,
                device_mesh=ep_mesh,
                placements=(Shard(0),),
            )
        self.weight = nn.Parameter(data)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Default nn.Linear init.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features},"
            f" n_routed_experts={self.n_routed_experts},"
            f" out_features={self.out_features},"
            f" in_features={self.in_features},"
            f" ep_mesh={self.ep_mesh})"
        )


class _ExpertFFNImpl(nn.Module, ABC):
    """
    Base class for different routed expert FFN implementations. Useful to create this as a separate
    module, so that we can checkpoint the FFN inputs without also checkpointing the all-to-all comms
    in EP use cases.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Inputs `x` are expected to be (n_toks, d_model)-shaped and unsorted. `indices` are either 1D
        and (n_toks) shaped or 2D and (n_toks, n_activated_experts) shaped. Outputs are (n_toks,
        d_model) or (n_toks * n_activated_experts, model) shaped in the two respective cases.
        """
        raise NotImplementedError


class ExpertFFNForLoop(_ExpertFFNImpl):
    """
    for-loop impl
    """

    def forward(
        self,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x_by_expert, flat_sorted_indices = _sort_by_exp_idx(x, indices)
        z = torch.zeros_like(x_by_expert)

        # Build the idx map
        n_local_experts = fc1_weight.shape[0]
        local_expert_idxs = torch.arange(
            n_local_experts,
            device=fc1_weight.device,
            dtype=torch.int32,
        )
        # NOTE: @goon - repeat_interleave incurs a CUDA sync since it needs to wait on
        # the CUDA tensor counts to know the output shape
        exp_idxs_per_tok = local_expert_idxs.repeat_interleave(counts)

        # Note: for some reason, getting the weights with CPU integer indexing, like fc1 =
        # self.fc1.weight[exp_idx], results in super-slow CUDA syncs during the backwards pass.
        for exp_idx, (fc1, fc2) in enumerate(zip(fc1_weight, fc2_weight)):
            tok_idx = exp_idx == exp_idxs_per_tok
            # NOTE: @goon -  also CUDA syncing here, since the slices have data-dependent shapes
            z[tok_idx] = _get_single_exp_output(
                x_by_expert[tok_idx], fc1, fc2, activation
            )

        # Save an allocation: store the unsorted results back in x_by_expert.
        x_by_expert[flat_sorted_indices] = z
        return x_by_expert


class ExpertFFNGroupedMM(_ExpertFFNImpl):
    """
    torch._grouped_mm
    """

    def forward(
        self,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x_by_expert, flat_sorted_indices = _sort_by_exp_idx(x, indices)

        # Expand the tokens to an appropriately aligned tensor.
        idxs_align, _, offsets = pad_sorted_idxs(
            counts, indices.numel(), _GROUPED_MM_ALIGNMENT
        )
        # NOTE: @goon - offsets[-1] induces a very long CUDA sync. Can avoid it by letting z be a fixed
        # sized, empirically tuned to be large enough to handle the routed tokens.
        z = torch.empty(offsets[-1], x.shape[-1], dtype=x.dtype, device=x.device)

        z[idxs_align] = x_by_expert
        z = _get_exp_outputs_grouped_mm(z, fc1_weight, fc2_weight, offsets, activation)

        # Remove the alignment and return the tokens to their original ordering
        x_by_expert[flat_sorted_indices] = z[idxs_align]
        return x_by_expert


class ExpertFFNGroupedMMTriton(_ExpertFFNImpl):
    """
    torch._grouped_mm and triton indexing kernels
    """

    def forward(
        self,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x_by_expert, flat_sorted_indices = _sort_by_exp_idx(x, indices)

        idxs_align, _, offsets = pad_sorted_idxs_torch(
            counts, indices.numel(), _GROUPED_MM_ALIGNMENT
        )
        # NOTE: @goon - offsets[-1] induces a very long CUDA sync. Can avoid it by letting z be a
        # fixed sized, empirically tuned to be large enough to handle the routed tokens.
        z = torch.empty(offsets[-1], x.shape[-1], dtype=x.dtype, device=x.device)
        z[idxs_align] = x_by_expert
        z = _get_exp_outputs_grouped_mm(z, fc1_weight, fc2_weight, offsets, activation)

        # Remove the alignment and return the tokens to their original ordering
        x_by_expert[flat_sorted_indices] = z[idxs_align]
        return x_by_expert


class ExpertFFNCGGroupedGemmTriton(_ExpertFFNImpl):
    """
    titan grouped gemm + triton indexing
    """

    def forward(
        self,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x_by_expert, flat_sorted_indices = _sort_by_exp_idx(x, indices)

        # Expand the tokens to an appropriately aligned tensor.
        idxs_align, _, offsets = pad_sorted_idxs(
            counts, indices.numel(), _GROUPED_MM_ALIGNMENT
        )
        # NOTE: @goon - offsets[-1] induces a very long CUDA sync. Can avoid it by letting z be a fixed
        # sized, empirically tuned to be large enough to handle the routed tokens.
        z = torch.empty(offsets[-1], x.shape[-1], dtype=x.dtype, device=x.device)

        z[idxs_align] = x_by_expert
        z = _get_exp_outputs_grouped_mm(z, fc1_weight, fc2_weight, offsets, activation)

        # Remove the alignment and return the tokens to their original ordering
        x_by_expert[flat_sorted_indices] = z[idxs_align]
        return x_by_expert


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

        self.fc1 = RoutedExpertsWeights(
            self.n_routed_experts,
            2 * d_intermediate,
            in_features,
            ep_mesh,
            **factory_kwargs,
        )
        self.fc2 = RoutedExpertsWeights(
            self.n_routed_experts,
            in_features,
            d_intermediate,
            ep_mesh,
            **factory_kwargs,
        )

        # To be set by subclasses. Make an __init__ arg?
        self.ffn_impl: _ExpertFFNImpl = None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
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

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
    ) -> torch.Tensor:
        z = self.ffn_impl(
            x=x,
            fc1_weight=self.fc1.weight,
            fc2_weight=self.fc2.weight,
            indices=indices,
            counts=counts,
            activation=self.activation,
        )
        # Reshape and weight
        z = z.reshape(*(weights.shape + x.shape[-1:]))
        z = torch.bmm(weights[:, None], z).squeeze(1)
        return z


class RoutedExpertsNoEPForLoop(_RoutedExpertsNoEP):
    """
    for-loop impl
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNForLoop()


class RoutedExpertsNoEPGroupedMM(_RoutedExpertsNoEP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNGroupedMM()


class RoutedExpertsNoEPGroupedMMTriton(_RoutedExpertsNoEP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNGroupedMMTriton()


class RoutedExpertsNoEPCGGroupedGemmTriton(_RoutedExpertsNoEP):
    """
    titan grouped gemm + triton indexing
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNCGGroupedGemmTriton()


### Start of EP Classes


@dataclass
class _EPData:
    """
    Small helper class for cleaning up API.
    """

    recv: Optional[torch.Tensor] = None
    send: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    counts: Optional[torch.Tensor] = None


class _RoutedExpertsTorchEP(_RoutedExperts):
    """
    Adds torch all-to-all EP routing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.ep_mesh is not None

    def _get_ep_toks_and_routing_data(
        self, x_by_expert: torch.Tensor, counts: torch.IntTensor
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

        assert self.ep_mesh is not None  # mypy
        layer_idx = self.layer_idx if hasattr(self, "layer_idx") else None
        with record_function("all2all::counts"):
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

        # Receive toks from other workers
        with record_function("all2all::send0"):
            x_recv = funcol.all_to_all_single_autograd(
                x_by_expert, recv_counts, send_counts, group=self.ep_mesh
            )
        return x_recv, send_counts, recv_counts, tokens_per_expert_group

    @contextmanager
    def _ep_context(
        self,
        x: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
    ):
        """
        Helper context manager which performs the all-to-all dispatch and combine ops. Requires
        the caller to write and read from the yielded _EPData object.
        """
        x_by_expert, flat_sorted_indices = _sort_by_exp_idx(x, indices)
        x_recv, send_counts, recv_counts, tokens_per_expert_group = (
            self._get_ep_toks_and_routing_data(x_by_expert, counts)
        )

        local_indices, local_counts = _get_local_indices_and_counts(
            tokens_per_expert_group, self.n_local_experts
        )
        # TODO: @goon - DELETE
        recv_count_per_ep_rank = tokens_per_expert_group.reshape(self.ep_mesh_size, -1).sum(
            dim=-1, dtype=torch.int32
        )
        print(f"{self.ep_mesh.get_rank()=}, {self.layer_idx=}: {local_counts=}")
        print(f"{self.ep_mesh.get_rank()=}, {self.layer_idx=}: {recv_count_per_ep_rank=}")
        data = _EPData(
            recv=x_recv,
            send=None,
            out=x_by_expert,
            indices=local_indices,
            counts=local_counts,
        )
        try:
            yield data
        finally:
            # Send results back to original ranks (reversed send/recv count data)
            with record_function("all2all::send1"):
                x_out = funcol.all_to_all_single_autograd(
                    data.send, send_counts, recv_counts, group=self.ep_mesh
                )

            # Save an allocation: store the unsorted results back in x_by_expert.
            data.out[flat_sorted_indices] = x_out

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
    ) -> torch.Tensor:
        with self._ep_context(x, indices, counts) as data:
            # Only use locally available EP weights
            data.send = self.ffn_impl(
                data.recv,
                self.fc1.weight.to_local(),
                self.fc2.weight.to_local(),
                data.indices,
                data.counts,
                self.activation,
            )

        # Reshape and weight
        x_by_expert = data.out.reshape(*(weights.shape + data.out.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z


class RoutedExpertsTorchEPForLoop(_RoutedExpertsTorchEP):
    """
    EP + for-loop
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNForLoop()


class RoutedExpertsTorchEPGroupedMM(_RoutedExpertsTorchEP):
    """
    EP + torch._grouped_mm
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn_impl = ExpertFFNGroupedMMTriton()


class RoutedExpertsTorchEPGroupedMMTriton(_RoutedExpertsTorchEP):
    """
    EP + torch._grouped_mm and triton indexing kernels.

    TODO: incomplete.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from torchtitan.experiments.kernels.moe.indices import generate_permute_indices  # noqa

        self._gen_perm_idx_fn = generate_permute_indices
        self.ffn_impl = ExpertFFNGroupedMMTriton()


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
        ep_mesh: Optional[DeviceMesh] = None,  # Intentionally unused
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.experts = nn.ModuleDict(
            {
                str(i): GatedMLP(
                    in_features=in_features,
                    hidden_features=d_intermediate,
                    multiple_of=multiple_of,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_routed_experts)
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.LongTensor,
        counts: torch.IntTensor,
    ) -> torch.Tensor:
        y = torch.zeros_like(x)
        # Use the ds code for counts, rather than the arg
        ds_counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.n_routed_experts):
            if ds_counts[i] == 0:
                continue
            expert = self.experts[str(i)]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y


NON_EP_EXPERT_CLASSES = {
    "torch": RoutedExpertsNoEPForLoop,
    "torch_gemm": RoutedExpertsNoEPGroupedMM,
    "torch_gemm_triton": RoutedExpertsNoEPGroupedMMTriton,
}
NON_EP_EXPERT_CLASSES_AND_SIMPLE = NON_EP_EXPERT_CLASSES.copy()
NON_EP_EXPERT_CLASSES_AND_SIMPLE["_simple"] = _SimpleRoutedExperts

EP_EXPERT_CLASSES = {
    "torch": RoutedExpertsTorchEPForLoop,
    "torch_gemm": RoutedExpertsTorchEPGroupedMM,
    # "torch_gemm_triton": RoutedExpertsTorchEPGroupedMMTriton,
}
