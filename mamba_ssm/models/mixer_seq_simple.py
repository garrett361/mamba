# Copyright (c) 2023, Albert Gu, Tri Dao.

import copy
import json
import math
import os
from collections import namedtuple
from functools import partial
from typing import Any, Optional, Callable

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.profiler import record_function

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import MoE
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    moe_layer_idx=None,
    moe_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    ep_mesh: Optional[DeviceMesh] = None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if moe_layer_idx is None:
        moe_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    if moe_cfg is None:
        moe_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(
                f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2"
            )
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs,
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    elif layer_idx in moe_layer_idx:
        mlp_cls = partial(
            MoE,
            ep_mesh=ep_mesh,
            **moe_cfg,
            **factory_kwargs,
        )
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
    verbose: bool=False
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                if verbose:
                    print(f"Calling _init_weights on {module=}")
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        if verbose:
            print(f"Calling _init_weights on {module=}")
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if verbose:
                    print(f"Calling _init_weights on {name=}")
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        moe_layer_idx=None,
        moe_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleDict(
            {
                str(i): create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    moe_layer_idx=moe_layer_idx,
                    moe_cfg=moe_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    ep_mesh=ep_mesh,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            }
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1
                if d_intermediate == 0
                else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in self.layers.items()
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer_idx in sorted(self.layers):
            # TODO: @goon - remove record_function
            with record_function(f"{layer_idx=}"):
                layer = self.layers[layer_idx]
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    **mixer_kwargs,
                )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        moe_layer_idx = config.moe_layer_idx
        moe_cfg = config.moe_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            moe_layer_idx=moe_layer_idx,
            moe_cfg=moe_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            ep_mesh=ep_mesh,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(
            input_ids, inference_params=inference_params, **mixer_kwargs
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(
            load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        )
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

    def _get_tok_counts(self) -> int:
        """
        Get, and reset, the total token counts received by all experts.
        """
        tok_count = 0
        for mod in self.modules():
            if isinstance(mod, MoE) and hasattr(mod.experts, "_tok_count"):
                tok_count += mod.experts._tok_count
                mod.experts._tok_count = 0
        return tok_count


def fully_shard_moe(
    model: MambaLMHeadModel,
    ep_degree: int,
    world_size: int,
    fsdp_mesh: DeviceMesh,
    ep_mesh: Optional[DeviceMesh] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    reshard_lm_head_after_fwd: bool = False,
    explicit_fwd_prefetch: bool = True,
    explicit_bwd_prefetch: bool = False,
    no_reshard: bool = False,
) -> None:
    # TODO: @goon - hsdp?
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
                ignored_params.add(block.mlp.experts.parameters())
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
    dt_limit=(0.0, float("inf")),
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
            reinit_mamba2(module)

    # Optional custom init at the end.
    if final_init_fn is not None:
        final_init_fn(model)


def reinit_mamba2(
    mamba2: Mamba2,
    A_init_range=(1, 16),
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    conv_init: Optional[float] = None,
) -> None:
    with torch.no_grad():
        if conv_init is not None:
            nn.init.uniform_(mamba2.conv1d.weight, -conv_init, conv_init)

        # Re-init log dt bias
        dt_seed = torch.randn_like(mamba2.dt_bias)
        dt = torch.exp(
            dt_seed * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
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
