import functools
import inspect
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Optional
from warnings import warn

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, _init_weights
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.moe import MoE, TokenCounter


def fully_shard_moe(
    model: MambaLMHeadModel,
    fsdp_mesh: DeviceMesh,
    ep_mesh: Optional[DeviceMesh] = None,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    reshard_lm_head_after_fwd: bool = False,
    explicit_fwd_prefetch: bool = True,
    explicit_bwd_prefetch: bool = False,
) -> None:
    # TODO: @goon - hsdp?
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    fully_shard(model.backbone.embedding, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(
        model.lm_head,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_lm_head_after_fwd,
    )
    for idx, block in model.backbone.layers.items():
        # Cases:
        # 1. ep_mesh is None: full replication, fully shard with the fsdp_mesh
        # 2. ep_mesh.ndim == 1: no expert replication at all. Ignore experts in fully_shard
        # 3. ep_mesh.ndim == 2: DP replication over outer dim, EP over inner dim

        # The ignored_params arg requires torch nightly (> 2.6.0)
        ignored_params = set()
        if isinstance(block.mlp, MoE):
            if ep_mesh is None:
                pass
            elif ep_mesh.ndim == 1:
                assert ep_mesh.ndim == 1, f"{ep_mesh.dim=}"
                # No replication in this case.
                for p in block.mlp.experts.parameters():
                    ignored_params.add(p)
            elif ep_mesh.ndim == 2:
                # Don't reshard due to comms costs
                assert ep_mesh.ndim == 2, f"{ep_mesh.dim=}"
                outer_ep_mesh_dim = ep_mesh.mesh_dim_names[0]
                # Expectation: the inner dim is EP, outer DP
                warn(
                    f"Received 2D {ep_mesh=}, applying fully_shard over the outer mesh dim: {outer_ep_mesh_dim}.",
                    stacklevel=1,
                )
                fully_shard(
                    block.mlp.experts,
                    mesh=ep_mesh[outer_ep_mesh_dim],
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
            else:
                raise ValueError(
                    f"Expected ep_mesh to be None or a 1- or 2-D DeviceMesh, got {ep_mesh=}"
                )
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


def act_ckpt_moe(model: MambaLMHeadModel, mixer_only: bool = True):
    """
    By default, only wraps the mixers to avoid repeating costly all-to-alls.
    """
    # TODO: @goon -  skip all-to-all act ckpting
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
def init_moe(
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
) -> None:
    """
    Move a meta-device moe model to a CUDA device and initialize its parameters.

    NOTE: @goon - the default init makes the MLP contributions to the residual stream much smaller
    than those from the mixer layer. Is that desired?
    """
    # Move to device and initialize, if using meta tensors:
    if any(p.is_meta for p in model.parameters()):
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


class ModelState(Stateful):
    # From torchtitan
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }


class OptimState(Stateful):
    # Modified from torchtitan
    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optimizer = (
            [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else optimizer
        )

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v for sd in map(func, self.model, self.optimizer) for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optimizer))


def get_dcp_state_dict(
    model: nn.Module | list[nn.Module],
    optimizer: Optional[torch.optim.Optimizer | list[torch.optim.Optimizer]] = None,
) -> dict[str, Stateful]:
    state_dict: dict[str, Stateful] = {}
    state_dict["model"] = ModelState(model)
    if optimizer is not None:
        state_dict["optim"] = OptimState(model, optimizer)
    return state_dict


class Hook(ABC):
    @property
    @abstractmethod
    def value(self) -> torch.Tensor: ...

    @value.setter
    @abstractmethod
    def value(self, value: torch.Tensor) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def __call__(self, module: nn.Module, args, output: torch.Tensor) -> None: ...

    @abstractmethod
    def remove(self) -> None: ...


class HookDict(dict):
    # TODO: @goon - reduce, reset, remove
    def __setitem__(self, key: str, value: Hook, /) -> None:
        if not isinstance(key, str):
            raise TypeError(f"All keys must be strings, not {key=}")
        if not isinstance(value, Hook):
            raise TypeError(f"All values must be Hook subclasses, not {value=}")
        return super().__setitem__(key, value)

    def reset(self) -> None:
        for hook in self.values():
            hook.reset()

    def remove(self) -> None:
        for hook in self.values():
            hook.remove()

    def all_reduce(
        self,
        group: Optional[dist.ProcessGroup] = None,
        **coll_kwargs,
    ) -> None:
        self._do_collective(dist.all_reduce, group=group, **coll_kwargs)

    def reduce(
        self,
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None,
        **coll_kwargs,
    ) -> None:
        # Only rank `dst` gets accurate results.
        self._do_collective(dist.reduce, group=group, dst=dst, **coll_kwargs)

    @torch.no_grad
    def _do_collective(
        self,
        in_place_coll,
        group: Optional[dist.ProcessGroup] = None,
        **coll_kwargs,
    ) -> None:
        # Concatenate and perform a single reduce for speed. Only rank `dst` gets accurate results.
        # HACK: add trailing trivial dim to avoid errors on concatenating zero-dim tensors, then
        # remove
        all_values = torch.cat([h.value[..., None] for h in self.values()])
        in_place_coll(all_values, group=group, **coll_kwargs)
        for hook, val_chunk in zip(self.values(), all_values.chunk(len(self))):
            hook.value = val_chunk[..., 0]  # Remove trailing dim, complete HACK


class TokenCounterHook(Hook):
    def __init__(self, module: nn.Module) -> None:
        counter_modules = [m for m in module.modules() if isinstance(m, TokenCounter)]
        if len(counter_modules) != 1:
            raise RuntimeError(
                f"{self.__class__.__name__} can only be attached to modules which contain exactly"
                f" one TokenCounter instance, not {module=}"
            )
        self._handle = counter_modules[0].register_forward_hook(self)
        self.reset()

    @property
    def value(self) -> torch.Tensor:
        if not isinstance(self._count, torch.Tensor):
            raise RuntimeError("No stats recorded yet.")
        return self._count

    @value.setter
    def value(self, value: torch.Tensor) -> None:
        self._count = value

    def reset(self) -> None:
        self._count = 0

    @torch.no_grad
    def __call__(self, module: nn.Module, args, output: torch.Tensor) -> None:
        self._count += output.detach().clone()

    def remove(self) -> None:
        self._handle.remove()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={self._count})"

    def __str__(self) -> str:
        return repr(self)


def attach_tok_count_hooks(
    model: MambaLMHeadModel,
) -> dict[str, TokenCounterHook]:
    hook_dict = HookDict()
    for fqn, mod in model.named_modules():
        if isinstance(mod, MoE):
            hook_dict[fqn] = TokenCounterHook(mod)
    return hook_dict


@torch.compile
def _get_mean_abs(tensor) -> torch.Tensor:
    return tensor.abs().mean()


class TensorMeanAbsHook(Hook):
    """
    Computes average per-element magnitude of the outputs tensor.
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        self.reset()
        self._handle = module.register_forward_hook(self)

    def reset(self) -> None:
        self._mag = 0
        self._iters = 0

    @property
    def value(self) -> torch.Tensor:
        if not isinstance(self._mag, torch.Tensor):
            raise RuntimeError("No stats recorded yet.")
        return self._mag

    @value.setter
    def value(self, value: torch.Tensor) -> None:
        self._mag = value

    @torch.no_grad
    def __call__(self, module: nn.Module, args, output: torch.Tensor) -> None:
        self._iters += 1
        self._mag = (
            self._mag * ((self._iters - 1) / self._iters)
            + _get_mean_abs(output.detach().clone()) / self._iters
        )

    def remove(self) -> None:
        self._handle.remove()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mag={self._mag}, iters={self._iters})"

    def __str__(self) -> str:
        return repr(self)


def attach_magnitude_hooks(
    model: MambaLMHeadModel,
    classes: nn.Module | type[nn.Module] | list[nn.Module | type[nn.Module]],
) -> dict[str, TensorMeanAbsHook]:
    """
    Attach TensorMeanAbsHook instances to every class or class instance specified.
    """
    if isinstance(classes, nn.Module):
        classes = [classes]
    hook_dict = HookDict()
    for fqn, mod in model.named_modules():
        if any(isinstance(mod, c) if inspect.isclass(c) else mod is c for c in classes):
            hook_dict[fqn] = TensorMeanAbsHook(mod)
    return hook_dict


def apply_loss_free_moe_balancing(
    lr: float,
    model: nn.Module,
    hook_dict: HookDict[str, TokenCounterHook],
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Apply loss-free moe balancing: arXiv:2408.15664.
    """
    hook_dict.all_reduce(group=group)
    for fqn, hook in hook_dict.items():
        moe = model.get_submodule(fqn)
        assert moe.gate.bias is not None
        mean_value = hook.value.mean(dtype=torch.float32)
        sign = torch.sign(hook.value - mean_value)
        moe.gate.bias -= lr * sign


@dataclass
class Meshes:
    dp: Optional[DeviceMesh] = None
    pp: Optional[DeviceMesh] = None
    ep: Optional[DeviceMesh] = None


def get_meshes(
    world_size: int,
    hsdp: bool | int = False,
    ep: bool | int = False,
    pp: bool | int = False,
) -> Meshes:
    # TODO: @goon - unify meshes more?
    # FSDP
    if hsdp:
        hsdp_degree = hsdp if isinstance(hsdp, int) else torch.cuda.device_count()
        assert not world_size % hsdp_degree, (
            f"{world_size=} must be divisible by {hsdp_degree=}"
        )

        dp_mesh = init_device_mesh(
            "cuda",
            (world_size // hsdp_degree, hsdp_degree),
            mesh_dim_names=("outer_dp", "inner_dp"),
        )
    else:
        dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("inner_dp",))

    # NOTE: @goon - In context parallel, training was much more stable when using separate meshes
    # for fsdp and CP. Trying the same thing here with EP, but not sure it matters. Don't think it
    # should, in principle. Also, we may just need separate meshes in the future for more complex
    # scenarios.
    if ep:
        ep_degree = ep if isinstance(ep, int) else world_size
        assert world_size >= ep, f"{world_size=} must be at least as large as {ep=}"
        assert not world_size % ep_degree, (
            f"{world_size=} must be divisible by {ep_degree=}"
        )

        # Cases:
        # 1. ep_degree = 1: full replication, no ep_mesh
        # 2. ep_degree = world_size: ep_mesh is the world
        # 3. world_size > ep_degree > world_size: 2D mesh with (DP, EP) dims, experts distributed along
        #    slice.
        if ep_degree == 1:
            ep_mesh = None
        elif ep_degree == world_size:
            ep_mesh = init_device_mesh(
                "cuda",
                (world_size,),
                mesh_dim_names=("inner_ep",),
            )
        else:
            ep_mesh = init_device_mesh(
                "cuda",
                (world_size // ep_degree, ep_degree),
                mesh_dim_names=("outer_ep", "inner_ep"),
            )
    else:
        ep_mesh = None

    # TODO: @goon -
    pp_mesh = None
    return Meshes(dp=dp_mesh, ep=ep_mesh, pp=pp_mesh)


@torch.compile
def _get_tensor_list_norm(
    tensor_list: list[torch.Tensor], norm_type: int
) -> torch.Tensor:
    return torch.stack(tensor_list).pow(norm_type).sum().pow(1 / norm_type)


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Similar to torchtitan's impl, but with extra handling for EP.
    """

    g_dict = defaultdict(list)
    for p in parameters:
        if p.grad is not None:
            if isinstance(p, DTensor):
                g_dict[p.grad.device_mesh].append(p.grad)
            else:
                g_dict[None].append(p.grad)

    # Need to norm grads with different meshes independently; unsupported op otherwise.
    norm_dict = {
        k: torch.nn.utils.get_total_norm(g, norm_type, error_if_nonfinite, foreach)
        for k, g in g_dict.items()
    }
    for k, v in norm_dict.items():
        if isinstance(p, DTensor):
            norm_dict[k] = v.full_tensor()

    if math.isinf(norm_type):
        total_norm = max(norm_dict.values())
    else:
        total_norm = _get_tensor_list_norm(list(norm_dict.values()), norm_type)

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def set_pp_layers(
    model: MambaLMHeadModel,
    n_stages: int,
    stage_idx: int,
) -> None:
    """
    Deletes and/or sets appropriate layers to None. Ideally acts on a meta-device model.
    """
    is_first = stage_idx == 0
    is_last = stage_idx == n_stages - 1
    if not is_first:
        model.backbone.embedding = None
    if not is_last:
        model.lm_head = None
        model.backbone.norm_f = None

    # Divide blocks among stages. Currently giving early stages extra blocks.
    n_blocks = len(model.backbone.layers)
    all_block_idxs = set(model.backbone.layers)
    this_stage_block_idxs = {
        str(n)
        for n in torch.arange(n_blocks).tensor_split(n_stages)[stage_idx].tolist()
    }
    for k in all_block_idxs - this_stage_block_idxs:
        del model.backbone.layers[k]
