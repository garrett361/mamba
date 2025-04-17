from typing import Literal, Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from mamba_ssm.modules.mlp import GatedMLP


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        in_features (int): Dimensionality of input features.
        d_intermediate (int): Dimensionality of hidden features of each expert.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
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
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
        _force_equal_loads: bool = False,
        _moe_kernel: Literal["torch", "torch_gemm", "torchao"] = "torch",
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
        self._tok_count = 0
        self._force_equal_loads = _force_equal_loads

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
        # counts[e] = num tokens this rank sends to expert e. Tiny optimization: counts doesn't need
        # grad, as it is only used for meta-indexing.
        with torch.no_grad():
            counts = indices.new_zeros((indices.shape[0], self.n_routed_experts))
            counts.scatter_(1, indices, 1)
            counts = counts.sum(dim=0)
            if self._force_equal_loads:
                counts = torch.full_like(counts, counts.sum() // counts.numel())

        if self.ep_mesh is None:
            z = self._get_routed_expert_outputs(x, weights, indices, counts)
        else:
            z = self._get_ep_routed_expert_outputs(x, weights, indices, counts)

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)

    def _get_routed_expert_outputs(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.zeros_like(x)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # TODO: @goon - handle no-tokens edge case
            expert = self.experts[str(i)]
            # TODO: @goon - torch.where incurs a CUDA sync. Can we avoid it? Not sure how, due to
            # shape-dependent outputs.
            idx, top = torch.where(indices == i)
            z[idx] += expert(x[idx]) * weights[idx, top, None]
        return z

    def _get_ep_routed_expert_outputs(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        # [EP Routing and Indexing]
        # To perform EP routing with torch comms primitives, each EP rank needs to know how many
        # tokens EP rank `r` will be sending to its local expert `l`. This can be done by exchanging
        # information about each rank's `counts`.
        #
        # Each rank then sorts their tokens in order of their global expert idx and sends as
        # appropriate. The received tokens are **not** in expert order: they are first ordered by
        # the sending rank, and then by local order. Schematically: recv ~ cat([recv_from_rank_0] +
        # [recv_from_rank_1] + ...) where recv_from_rank_r will contain the tokens for each of the
        # L local experts sorted by local expert order.
        #
        # In order to use GEMM kernels, the received tokens must be re-sorted in local expert
        # order, so that tokens belonging to the same local expert are all contiguous. This is a
        # data-dependent resorting and not easy to rewrite without CUDA syncs.

        # Sort tokens by the expert they are indexed to. Tokens belonging to the same expert are
        # contiguous in x_by_expert and in expert order
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        x_by_expert = x[flat_sorted_indices // self.n_activated_experts]

        assert self.ep_mesh is not None  # mypy
        # Get counts of incoming tensors. tokens_per_expert_group.reshape(self.ep_mesh.size(),
        # self.n_local_experts)[r, l] = num tokens rank r sent to local expert l
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
        self._tok_count += sum(recv_counts)

        # Receive toks from other workers
        x_recv = funcol.all_to_all_single_autograd(
            x_by_expert, recv_counts, send_counts, group=self.ep_mesh
        )

        x_send = self._get_ep_send_toks_torch(x_recv, tokens_per_expert_group)

        # Send results back to original ranks (reversed send/recv count data)
        x_out = funcol.all_to_all_single_autograd(
            x_send, send_counts, recv_counts, group=self.ep_mesh
        )

        # Store the unsorted results back in x_by_expert
        x_by_expert[flat_sorted_indices] = x_out
        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x_by_expert.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z

    def _get_ep_send_toks_torch(
        self, x_recv: torch.Tensor, tokens_per_expert_group: torch.Tensor
    ) -> torch.Tensor:
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
        local_expert_idxs = (
            local_expert_idxs.repeat_interleave(tokens_per_expert_group)
            + self.experts_start_idx
        )

        for exp_idx in range(self.experts_start_idx, self.experts_end_idx):
            idxs = local_expert_idxs == exp_idx
            # TODO: @goon - handle no-tokens edge case
            x_send[idxs] = self.experts[str(exp_idx)](x_recv[idxs])
        return x_send
