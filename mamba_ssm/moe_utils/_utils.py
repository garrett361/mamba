import math
from functools import partial
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, _init_weights
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.moe import MoE


def fully_shard_moe(
    model: MambaLMHeadModel,
    ep_degree: int,
    world_size: int,
    fsdp_mesh: DeviceMesh,
    ep_mesh: Optional[DeviceMesh] = None,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    reshard_lm_head_after_fwd: bool = False,
    explicit_fwd_prefetch: bool = True,
    explicit_bwd_prefetch: bool = False,
    no_reshard: bool = False,
) -> None:
    # TODO: @goon - hsdp?
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    assert fsdp_mesh.ndim == 1, f"{fsdp_mesh.dim=}"
    fully_shard(model.backbone.embedding, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(
        model.lm_head,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_lm_head_after_fwd,
    )
    for idx, block in model.backbone.layers.items():
        # Cases:
        # 1. ep_degree = 1: full replication, fully shard with the fsdp_mesh
        # 2. ep_degree = world_size: no expert replication at all. Ignore experts in fully_shard
        # 3. world_size > ep_degree > world_size: world_size // ep_degree expert replicas.

        # The ignored_params arg requires torch nightly (> 2.6.0)
        ignored_params = set()
        if isinstance(block.mlp, MoE):
            if ep_degree == 1:
                pass
            elif ep_degree == world_size:
                # No replication in this case.
                for p in block.mlp.experts.parameters():
                    ignored_params.add(p)
            else:
                # Don't reshard due to comms costs
                outer_ep_mesh_dim = ep_mesh.mesh_dim_names[0]
                fully_shard(
                    block.mlp.experts,
                    mesh=ep_mesh[outer_ep_mesh_dim],
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
                block.mlp.experts.set_reshard_after_backward(False)
        is_not_last_block = int(idx) < len(model.backbone.layers) - 1
        fully_shard(
            block,
            mesh=fsdp_mesh,
            ignored_params=ignored_params,
            mp_policy=mp_policy,
            reshard_after_forward=is_not_last_block,
        )
    fully_shard(
        model,
        mesh=fsdp_mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
    )
    if explicit_fwd_prefetch:
        blocks = list(model.backbone.layers.values())
        blocks.append(model.lm_head)
        for b_prev, b_next in zip(blocks[:-1], blocks[1:]):
            b_prev.set_modules_to_forward_prefetch([b_next])

    if explicit_bwd_prefetch:
        blocks = [model.lm_head, *model.backbone.layers.values()]
        reversed_blocks = list(reversed(blocks))
        for b_prev, b_next in zip(reversed_blocks[:-1], reversed_blocks[1:]):
            b_prev.set_modules_to_backward_prefetch([b_next])

    if no_reshard:
        model.lm_head.set_reshard_after_backward(False)
        model.backbone.embedding.set_reshard_after_backward(False)
        for block in model.backbone.layers.values():
            block.set_reshard_after_backward(False)
        model.set_reshard_after_backward(False)


def act_ckpt_moe(model: MambaLMHeadModel, mixer_only: bool = True):
    """
    By default, only wraps the mixers to avoid repeating costly all-to-alls.
    """
    for layer_idx, block in model.backbone.layers.items():
        if mixer_only:
            model.backbone.layers[layer_idx].mixer = checkpoint_wrapper(
                block.mixer, preserve_rng_state=False
            )
        else:
            model.backbone.layers[layer_idx] = checkpoint_wrapper(
                block, preserve_rng_state=False
            )


@torch.no_grad()
def init_meta_moe(
    model: MambaLMHeadModel,
    device: Optional[torch.cuda.device] = None,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    initializer_cfg: Optional[dict[str, Any]] = None,
    A_init_range=(1, 16),
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    conv_init: Optional[float] = None,
    verbose: bool = False,
    final_init_fn: Optional[Callable[[nn.Module], None]] = None,
):
    """
    Move a meta-device moe model to a CUDA device and initialize its parameters.
    """
    # Move to cuda and initialize.
    model.to_empty(device=device or torch.cuda.current_device())

    # First, default init anything that can be default initialized
    for n, m in model.named_modules():
        if hasattr(m, "reset_parameters"):
            if verbose:
                print(f"Calling reset_parameters on {n=}")
            m.reset_parameters()

    # Then apply the default mamba_ssm init
    if verbose:
        print("Applying default mamba_ssm init")
    model.apply(
        partial(
            _init_weights,
            n_layer=model.config.n_layer,
            initializer_range=initializer_range,
            **(initializer_cfg if initializer_cfg is not None else {}),
            rescale_prenorm_residual=rescale_prenorm_residual,
            n_residuals_per_layer=1
            if model.config.d_intermediate == 0
            else 2,  # 2 if we have MLP
            verbose=verbose,
        )
    )

    # Need to reinitialize mamba2 params.
    for name, module in model.named_modules():
        if isinstance(module, Mamba2):
            if verbose:
                print(f"Re-init Mamba2: {name=}")
            _reinit_mamba2(
                module,
                A_init_range=A_init_range,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                conv_init=conv_init,
            )

    # Optional custom init at the end.
    if final_init_fn is not None:
        final_init_fn(model)


@torch.no_grad()
def _reinit_mamba2(
    mamba2: Mamba2,
    A_init_range=(1, 16),
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    conv_init: Optional[float] = None,
) -> None:
    if conv_init is not None:
        nn.init.uniform_(mamba2.conv1d.weight, -conv_init, conv_init)

    # Re-init log dt bias
    dt_seed = torch.randn_like(mamba2.dt_bias)
    dt = torch.exp(dt_seed * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    mamba2.dt_bias.data = inv_dt

    # Re-init A_log
    assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
    # NOTE: @goon - needs a float32 here?
    A = mamba2.A_log.uniform_(*A_init_range)
    A_log = torch.log(A)
    mamba2.A_log.data = A_log

    # Re-init D
    nn.init.ones_(mamba2.D)


def get_total_exp_and_active_params(model: MambaLMHeadModel) -> tuple[int, int, int]:
    """
    Utility for getting the parameter count from all params, routed experts, and number of active
    params per token.
    """
    total = sum(p.numel() for p in model.parameters())
    exp = 0
    for m in model.modules():
        if isinstance(m, MoE):
            exp += sum(p.numel() for p in m.experts.parameters())
            moe_mod = m
    if exp == 0:
        return total, 0, total

    non_exp = total - exp
    # Assumption: same active/routed ratio globally.
    active_exp = (exp * moe_mod.n_activated_experts) // moe_mod.n_routed_experts
    return total, exp, non_exp + active_exp


class MoEState(Stateful):
    def __init__(self, model, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
