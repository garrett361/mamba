import tempfile
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional
from warnings import warn

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.tensor import DTensor

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.moe import (
    EP_EXPERT_CLASSES,
    MoE,
    RoutedExpertsNoEPForLoop,
    RoutedExpertsTorchEPGroupedMM,
    TokenCounter,
)
from mamba_ssm.moe_utils import (
    attach_magnitude_hooks,
    attach_tok_count_hooks,
    clip_grad_norm_,
    fully_shard_moe,
    get_dcp_state_dict,
    init_moe,
)
from mamba_ssm.moe_utils._utils import (
    act_ckpt_moe,
    apply_loss_free_moe_balancing,
    set_pp_layers,
)
from tests.moe.test_utils import (
    flattened_cross_entropy,
    mean_loss_fn,
    skip_moe_impl_if_no_h100s,
    sum_loss_fn,
)

"""
Notes:
- Safer to use SGD with momentum than Adam in tests because the former is more sensitive to
  accidentally errors in multiplicative factors of param grads, which can easily occur with EP
  sharding.
"""


def _copy_params(model: nn.Module, model_fsdp: nn.Module) -> None:
    """
    Copy prams from the sharded model to the local model.
    """
    for mod_name, mod_fsdp in model_fsdp.named_modules():
        # Act-ckpt wrapper handling
        try:
            mod = model.get_submodule(mod_name)
        except AttributeError:
            mod = model.get_submodule(
                mod_name.replace("._checkpoint_wrapped_module", "")
            )

        with torch.no_grad():
            for p_dest, p_src in zip(
                mod.parameters(recurse=False), mod_fsdp.parameters(recurse=False)
            ):
                if isinstance(p_src, DTensor):
                    p_src = p_src.full_tensor()
                p_dest.data.copy_(p_src.data)


def _test_grads(
    model: nn.Module,
    model_fsdp: nn.Module,
    tol: float,
    skip_patterns: Optional[str | list[str]] = None,
) -> None:
    if skip_patterns is None:
        skip_patterns = []
    elif isinstance(skip_patterns, str):
        skip_patterns = [skip_patterns]

    fails = {}
    with torch.no_grad():
        # NOTE: @goon - by iterating over the model_fsdp modules, and not the model modules, this
        # function also handles PP.
        for mod_name, mod_fsdp in model_fsdp.named_modules():
            # Act-ckpt wrapper handling
            try:
                mod = model.get_submodule(mod_name)
            except AttributeError:
                mod = model.get_submodule(
                    mod_name.replace("._checkpoint_wrapped_module", "")
                )

            for (n, p), (_, p_fsdp) in zip(
                mod.named_parameters(recurse=False),
                mod_fsdp.named_parameters(recurse=False),
            ):
                if any(skip_pat in n for skip_pat in skip_patterns):
                    warn(f"Skipping {n=} due to {skip_patterns=}", stacklevel=1)
                    continue
                if p.grad is None:
                    assert p_fsdp.grad is None
                grad = p.grad
                grad_fsdp = p_fsdp.grad
                if isinstance(grad_fsdp, DTensor):
                    grad_fsdp = grad_fsdp.full_tensor()
                try:
                    torch.testing.assert_close(grad_fsdp, grad, atol=tol, rtol=tol)
                except AssertionError as e:
                    fails[(mod, n)] = str(e)
    if fails:
        raise AssertionError(str(fails))


class _TestBase(DTest):
    in_features = 256
    d_intermediate = in_features // 2
    n_shared_experts = 1
    n_activated_experts = 2
    n_layer = 2
    vocab_size = 512
    tie_embeddings = False

    seqlen = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Uniform FSDP2 dtype issues with bfloat16 b/c the dt_bias doesn't respect factory_kwargs.
    dtype = torch.float32
    ssm_cfg = {"layer": "Mamba2"}
    attn_layer_idx = [n_layer - 1]
    head_dim = 64
    attn_cfg = {
        "causal": True,
        "d_conv": 0,
        "head_dim": head_dim,
        "num_heads": 8,
        "num_heads_kv": 2,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,
    }
    moe_layer_idx = list(range(1, n_layer))

    # Put the tolerances pretty high. Should still catch egregious errors, while allowing for
    # sharding inaccuracies.
    tol = 1e-1
    hi_tol = 1e-3

    # Largish lr so that stepping produced big changes
    lr = 1e-2
    momentum = 1e-2

    @property
    def n_routed_experts(self) -> int:
        return 4 * self.n_activated_experts * self.world_size

    @property
    def batch_size(self) -> int:
        return self.world_size

    @property
    def moe_cfg(self) -> dict[str, int]:
        return {
            "n_routed_experts": self.n_routed_experts,
            "n_activated_experts": 1,
            "n_shared_experts": 1,
            "d_intermediate": 64,
            "gate_bias": True,
        }

    @property
    def cfg(self) -> MambaConfig:
        return MambaConfig(
            d_model=self.in_features,
            d_intermediate=self.d_intermediate,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
            attn_layer_idx=self.attn_layer_idx,
            attn_cfg=self.attn_cfg,
            moe_layer_idx=self.moe_layer_idx,
            moe_cfg=self.moe_cfg,
            ssm_cfg=self.ssm_cfg,
        )

    @property
    def attn_only_cfg(self) -> MambaConfig:
        """
        For testing attn-only models.
        """
        return MambaConfig(
            d_model=self.in_features,
            d_intermediate=self.d_intermediate,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
            attn_layer_idx=list(range(self.n_layer)),
            attn_cfg=self.attn_cfg,
            moe_layer_idx=self.moe_layer_idx,
            moe_cfg=self.moe_cfg,
            ssm_cfg=self.ssm_cfg,
        )

    @property
    def factory_kwargs(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def get_inputs(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.batch_size, self.seqlen, self.in_features, **self.factory_kwargs
        )

    def get_input_toks(
        self, seed: int = 42, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randint(
            self.vocab_size,
            size=(batch_size or self.batch_size, self.seqlen),
            device=self.device,
        )

    def get_inputs_weights_indices_counts(
        self, seed: int = 42
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        torch.manual_seed(seed)
        inputs = torch.randn(
            self.batch_size * self.seqlen, self.in_features, **self.factory_kwargs
        )
        weights = torch.randn(
            self.batch_size * self.seqlen,
            self.n_activated_experts,
            **self.factory_kwargs,
        ).softmax(dim=-1)
        indices = (
            torch.randn(
                self.batch_size * self.seqlen,
                self.n_routed_experts,
                device=self.device,
            )
            .topk(self.n_activated_experts, dim=-1)
            .indices
        )
        return inputs, weights, indices, TokenCounter()(indices, self.n_routed_experts)

    def get_pp_input_output_args(
        self, is_first: bool, is_last: bool, batch_size: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the meta-device input/output args which allow for skipping PP shape-inference.
        """
        batch_size = batch_size or self.batch_size
        if is_first:
            input_args = self.get_input_toks(batch_size=batch_size).to(device="meta")
        else:
            input_args = self.get_inputs(batch_size=batch_size).to(device="meta")
        if is_last:
            output_args = torch.randn(
                batch_size, self.seqlen, self.vocab_size, device="meta"
            )
        else:
            output_args = self.get_inputs(batch_size=batch_size).to(device="meta")

        return input_args, output_args

    @contextmanager
    def temp_dir(self):
        """
        Create a shared temp dir for writing to.
        """
        if not self.rank:
            temp_dir = tempfile.TemporaryDirectory()
            temp_dir_name = temp_dir.name
        else:
            temp_dir_name = None
        temp_dir_name_list = [temp_dir_name]
        dist.broadcast_object_list(temp_dir_name_list, src=0)
        try:
            yield temp_dir_name_list[0]
        finally:
            # The temp dir is cleaned up once it leaves scope. The barrier ensures all procs have
            # left the ctx manager before performing this cleanup.
            dist.barrier()


class TestRoutedExperts(_TestBase):
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_fwd(self, cls) -> None:
        skip_moe_impl_if_no_h100s(cls)
        # Some classes have dtype constraints:
        if cls == RoutedExpertsTorchEPGroupedMM:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )
        model = RoutedExpertsNoEPForLoop(**model_kwargs)
        model_ep = cls(**model_kwargs, ep_mesh=ep_mesh)

        # Set weights equal
        _copy_params(model, model_ep)

        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts(
            seed=42 + self.rank
        )
        outputs = model(inputs, weights, indices, counts)
        outputs_ep = model_ep(inputs, weights, indices, counts)

        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_bwd(self, cls) -> None:
        skip_moe_impl_if_no_h100s(cls)
        # Some classes have dtype constraints:
        if cls == RoutedExpertsTorchEPGroupedMM:
            self.dtype = torch.bfloat16

        torch.manual_seed(42)
        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts()
        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        weights_ep = weights.tensor_split(self.world_size, dim=0)[self.rank]
        indices_ep = indices.tensor_split(self.world_size, dim=0)[self.rank]

        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )
        model = RoutedExpertsNoEPForLoop(**model_kwargs)
        model_ep = cls(**model_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)

        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        optim_ep = torch.optim.SGD(
            model_ep.parameters(), lr=self.lr, momentum=self.momentum
        )

        outputs = model(inputs, weights, indices, counts)

        # The counts need to be rederived from indices_ep
        counts_ep = TokenCounter()(indices_ep, self.n_routed_experts)
        outputs_ep = model_ep(inputs_ep, weights_ep, indices_ep, counts_ep)

        torch.testing.assert_close(
            outputs_ep,
            outputs.to(outputs_ep).tensor_split(self.world_size, dim=0)[self.rank],
            atol=self.hi_tol,
            rtol=self.hi_tol,
        )

        # Note: important to use a sum-type loss function because this test is not using any grad
        # averaging mechanism like fully_shard.
        sum_loss_fn(outputs).backward()
        sum_loss_fn(outputs_ep).backward()

        _test_grads(model, model_ep, tol=self.hi_tol)

        # Step and ensure outputs match again
        optim.step()
        optim.zero_grad()
        optim_ep.step()
        optim_ep.zero_grad()

        outputs = model(inputs, weights, indices, counts)
        outputs_ep = model_ep(inputs_ep, weights_ep, indices_ep, counts_ep)
        torch.testing.assert_close(
            outputs_ep, outputs.tensor_split(self.world_size, dim=0)[self.rank]
        )

    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_zero_toks(self, cls) -> None:
        """
        Verify the routed experts can handle getting zero tokens for one or more experts.
        """
        skip_moe_impl_if_no_h100s(cls)
        torch.manual_seed(42)
        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts()
        # Set all indices to 0, 1, so that other experts don't get tokens
        assert self.n_routed_experts > 2
        indices[:, 0] = 0
        indices[:, 1] = 1
        counts = torch.zeros_like(counts)
        counts[0] = indices.numel() // 2
        counts[1] = indices.numel() // 2
        kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )

        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )
        model_ep = cls(**model_kwargs, ep_mesh=ep_mesh)

        # Just test for no errors
        out = model_ep(inputs, weights, indices, counts)
        mean_loss_fn(out).backward()


class TestMoEEP(_TestBase):
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd(self, moe_impl: str) -> None:
        skip_moe_impl_if_no_h100s(EP_EXPERT_CLASSES[moe_impl])
        # Some classes have dtype constraints:
        if "gemm" in moe_impl:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_shared_experts=self.n_shared_experts,
            score_func="sigmoid",
            moe_impl=moe_impl,
            **self.factory_kwargs,
        )
        model = MoE(**model_kwargs)
        model_ep = MoE(**model_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_inputs()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)

        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_bwd(self, moe_impl: str) -> None:
        skip_moe_impl_if_no_h100s(EP_EXPERT_CLASSES[moe_impl])
        # Some classes have dtype constraints:
        if "gemm" in moe_impl:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_shared_experts=self.n_shared_experts,
            score_func="sigmoid",
            moe_impl=moe_impl,
            **self.factory_kwargs,
        )
        model = MoE(**model_kwargs).to(self.dtype)
        model_ep = MoE(**model_kwargs, ep_mesh=ep_mesh).to(self.dtype)

        # Force models equal
        _copy_params(model, model_ep)

        fully_shard(model_ep.gate, mesh=ep_mesh)
        if model_ep.shared_experts is not None:
            fully_shard(model_ep.shared_experts, mesh=ep_mesh)

        # The ignored_params arg requires torch nightly (> 2.6.0)
        fully_shard(
            model_ep, mesh=ep_mesh, ignored_params=set(model_ep.experts.parameters())
        )

        # Fully shard averages grads for all wrapped layers, as is required for mean-type losses.
        # The EP expert weights aren't wrapped and so they miss out on these averaged factors.
        # Correctness then requires tensor hooks:
        for p in model_ep.parameters():
            p.register_hook(lambda g: g / self.world_size)

        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        optim_ep = torch.optim.SGD(
            model_ep.parameters(), lr=self.lr, momentum=self.momentum
        )

        inputs = self.get_inputs()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Note: important to use an avg-type loss here.
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_ep).backward()

        _test_grads(model, model_ep, tol=self.hi_tol)

        # Step and ensure outputs match again.
        optim.step()
        optim.zero_grad()
        optim_ep.step()
        optim_ep.zero_grad()

        outputs = model(inputs)
        outputs_ep = model_ep(inputs_ep)
        torch.testing.assert_close(
            outputs_ep,
            outputs.tensor_split(self.world_size, dim=0)[self.rank],
            atol=self.hi_tol,
            rtol=self.hi_tol,
        )


class TestModelEP(_TestBase):
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd(self, moe_impl: str) -> None:
        skip_moe_impl_if_no_h100s(EP_EXPERT_CLASSES[moe_impl])
        # Some classes have dtype constraints:
        if "gemm" in moe_impl:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        moe_cfg = deepcopy(self.cfg)
        moe_cfg.moe_cfg["moe_impl"] = moe_impl
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh)

        # Verify EP
        for m, m_ep in zip(model.modules(), model_ep.modules()):
            if isinstance(m, MoE):
                assert (
                    m_ep.experts.n_local_experts
                    == m.experts.n_local_experts // self.world_size
                )

        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)

        torch.testing.assert_close(outputs_ep, outputs, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    @pytest.mark.parametrize("attn_only", [True, False])
    def test_bwd(self, moe_impl: str, attn_only: bool) -> None:
        skip_moe_impl_if_no_h100s(EP_EXPERT_CLASSES[moe_impl])
        # Some classes have dtype constraints:
        if "gemm" in moe_impl:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        cfg = self.attn_only_cfg if attn_only else self.cfg
        model = MambaLMHeadModel(cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(cfg)
        moe_cfg.moe_cfg["moe_impl"] = moe_impl
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        # Force models equal
        _copy_params(model, model_ep)

        fully_shard(model_ep.lm_head, mesh=ep_mesh)
        fully_shard(model_ep.backbone.embedding, mesh=ep_mesh)
        for block in model_ep.backbone.layers.values():
            # The ignored_params arg requires torch nightly (> 2.6.0)
            ignored_params = set()
            if isinstance(block.mlp, MoE):
                for p in block.mlp.experts.parameters():
                    ignored_params.add(p)
                    # Fully shard averages grads for all wrapped layers, as is required for
                    # mean-type losses. The EP expert weights aren't wrapped and so they miss out on
                    # these averaged factors. Correctness then requires tensor hooks:
                    # p.register_hook(lambda g: g / self.world_size)
            fully_shard(block, mesh=ep_mesh, ignored_params=ignored_params)

        fully_shard(model_ep, mesh=ep_mesh)

        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        optim_ep = torch.optim.SGD(
            model_ep.parameters(), lr=self.lr, momentum=self.momentum
        )

        inputs = self.get_input_toks()
        outputs = model(inputs).logits

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep).logits

        # Grads should match with an avg-over-batches type loss
        flattened_cross_entropy(outputs, inputs).backward()
        flattened_cross_entropy(outputs_ep, inputs_ep).backward()

        # NOTE: @goon - the D grads are non-deterministic for some triton reasons, unfortunately.
        _test_grads(model, model_ep, tol=self.hi_tol, skip_patterns="D")

        # Step and ensure outputs match again. NOTE: @goon - works for attn_only=True, fails for
        # False, probably because of D grad issues?
        optim.step()
        optim.zero_grad()
        optim_ep.step()
        optim_ep.zero_grad()

        outputs = model(inputs).logits
        outputs_ep = model_ep(inputs_ep).logits
        torch.testing.assert_close(
            outputs_ep,
            outputs.tensor_split(self.world_size, dim=0)[self.rank],
            atol=self.hi_tol,
            rtol=self.hi_tol,
        )

    @pytest.mark.skip(
        "Hitting 'Expected IntList but got GenericList', skipping for now"
    )
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd_compile(self, moe_impl: str) -> None:
        skip_moe_impl_if_no_h100s(EP_EXPERT_CLASSES[moe_impl])
        # Some classes have dtype constraints:
        if "gemm" in moe_impl:
            self.dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        moe_cfg = deepcopy(self.cfg)
        moe_cfg.moe_cfg["moe_impl"] = moe_impl
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh)

        # Verify EP
        for m, m_ep in zip(model.modules(), model_ep.modules()):
            if isinstance(m, MoE):
                assert (
                    m_ep.experts.n_local_experts
                    == m.experts.n_local_experts // self.world_size
                )

        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        model_ep = torch.compile(model_ep)
        outputs_ep = model_ep(inputs)

        torch.testing.assert_close(outputs_ep, outputs, atol=self.tol, rtol=self.tol)


class TestMoEUtils(_TestBase):
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_meta_init(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        with torch.device("meta"):
            meta_model_ep = MambaLMHeadModel(self.cfg, ep_mesh=ep_mesh)

        init_moe(meta_model_ep)
        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs_ep = meta_model_ep(inputs)

    # NOTE: @goon - currently using self.attn_only_cfg in fully_shard_moe tests to avoid
    # complications with non-deterministic mamba D grads. The fully_shard_moe util is independent of
    # having mamba layers or not.
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(
            self.dtype
        )
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        fully_shard_moe(model_ep, fsdp_mesh=ep_mesh, ep_fsdp_mesh=None)
        init_moe(model_ep)
        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_bwd_fully_shard_moe(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(
            self.dtype
        )
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_fsdp_mesh=None,
        )
        init_moe(model_ep)
        # Force models equal
        _copy_params(model, model_ep)

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an avg-over-batches type loss
        flattened_cross_entropy(outputs.logits, inputs).backward()
        flattened_cross_entropy(outputs_ep.logits, inputs_ep).backward()

        _test_grads(model, model_ep, tol=self.hi_tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe_replicated(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            "cuda",
            (2, self.world_size // 2),
            mesh_dim_names=("outer", "inner"),
        )
        fsdp_mesh = init_device_mesh(
            "cuda",
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(
            self.dtype
        )
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(
            moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh["inner"]
        ).to(self.dtype)

        fully_shard_moe(
            model_ep,
            fsdp_mesh=fsdp_mesh,
            ep_fsdp_mesh=ep_mesh["outer"],
        )
        init_moe(model_ep)
        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("mp", [False, True])
    def test_bwd_fully_shard_moe_replicated(self, mp: bool) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            "cuda",
            (2, self.world_size // 2),
            mesh_dim_names=("outer", "inner"),
        )
        fsdp_mesh = init_device_mesh(
            "cuda",
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(
            self.dtype
        )
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(
            moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh["inner"]
        ).to(self.dtype)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=fsdp_mesh,
            ep_fsdp_mesh=ep_mesh["outer"],
            mp_policy=None
            if not mp
            else MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
        )
        init_moe(model_ep)
        _copy_params(model, model_ep)

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an avg-over-batches type loss
        flattened_cross_entropy(outputs.logits, inputs).backward()
        flattened_cross_entropy(outputs_ep.logits, inputs_ep).backward()

        _test_grads(model, model_ep, tol=self.tol if mp else self.hi_tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe_hsdp(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            "cuda",
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        hsdp_mesh = init_device_mesh(
            "cuda",
            (2, self.world_size // 2),
            mesh_dim_names=("outer", "inner"),
        )

        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(
            self.dtype
        )
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )
        fully_shard_moe(
            model_ep,
            fsdp_mesh=hsdp_mesh,
            ep_fsdp_mesh=None,
        )
        init_moe(model_ep)
        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe_with_mp(self) -> None:
        # Everything in bfloat16
        dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.attn_only_cfg, **self.factory_kwargs).to(dtype)
        moe_cfg = deepcopy(self.attn_only_cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh)
        init_moe(model_ep)

        # Force models equal
        _copy_params(model, model_ep)
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_fsdp_mesh=None,
            mp_policy=mp_policy,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        outputs_ep = model_ep(inputs).logits
        # NOTE: @goon - currently failing on ~0.8% of all inputs.
        torch.testing.assert_close(
            outputs_ep, outputs, atol=self.hi_tol, rtol=self.hi_tol
        )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_dcp_save_load(self) -> None:
        with self.temp_dir() as checkpoint_id:
            torch.manual_seed(42)
            ep_mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("ep",)
            )
            model_ep = MambaLMHeadModel(
                self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh
            ).to(self.dtype)

            fully_shard_moe(
                model_ep,
                fsdp_mesh=ep_mesh,
                ep_fsdp_mesh=None,
            )
            # Large LR to create big changes:
            optim = torch.optim.SGD(
                model_ep.parameters(), lr=self.lr, momentum=self.momentum
            )
            torch.manual_seed(42 + self.rank)
            inputs = self.get_input_toks()
            pre_step_outputs = model_ep(inputs).logits
            mean_loss_fn(pre_step_outputs).backward()
            optim.step()
            optim.zero_grad()
            post_step_outputs = model_ep(inputs).logits
            # Sanity check that the model changed:
            assert not torch.allclose(pre_step_outputs, post_step_outputs)

            # Save state
            state_dict = get_dcp_state_dict(model_ep, optim)
            dcp.save(state_dict, checkpoint_id=checkpoint_id)
            # Not sure the barrier is needed, but just in case.
            dist.barrier()

            # Corrupt the state by taking another step
            mean_loss_fn(post_step_outputs).backward()
            optim.step()
            optim.zero_grad()

            # And save the new outputs
            post_second_step_outputs = model_ep(inputs).logits
            # Sanity check:
            assert not torch.allclose(post_step_outputs, post_second_step_outputs)

            # Reload and overwrite the corrupted state
            corrupted_state_dict = get_dcp_state_dict(model_ep, optim)
            dcp.load(corrupted_state_dict, checkpoint_id=checkpoint_id)

            # Check outputs agree pre- and post-taking another step
            reloaded_outputs = model_ep(inputs).logits
            torch.testing.assert_close(post_step_outputs, reloaded_outputs)

            mean_loss_fn(reloaded_outputs).backward()
            optim.step()
            optim.zero_grad()
            post_reload_step_outputs = model_ep(inputs).logits
            torch.testing.assert_close(
                post_reload_step_outputs, post_second_step_outputs
            )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_dcp_save_load_reconfigured(self) -> None:
        """
        Test reloading in a different sharded cfg.
        """
        with self.temp_dir() as checkpoint_id:
            # Run on all available ranks initially:
            seed = 42
            torch.manual_seed(seed)
            ep_mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("ep",)
            )
            model_ep = MambaLMHeadModel(
                self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh
            ).to(self.dtype)

            fully_shard_moe(
                model_ep,
                fsdp_mesh=ep_mesh,
                ep_fsdp_mesh=None,
            )
            # Large LR to create big changes:
            lr = 1e-1
            optim = torch.optim.SGD(
                model_ep.parameters(), lr=lr, momentum=self.momentum
            )
            torch.manual_seed(42 + self.rank)
            inputs = self.get_input_toks()
            pre_step_outputs = model_ep(inputs).logits
            mean_loss_fn(pre_step_outputs).backward()
            optim.step()
            optim.zero_grad()
            post_step_outputs = model_ep(inputs).logits
            # Sanity check that the model changed:
            assert not torch.allclose(pre_step_outputs, post_step_outputs)

            # Save state
            state_dict = get_dcp_state_dict(model_ep, optim)
            dcp.save(state_dict, checkpoint_id=checkpoint_id)
            # Not sure the barrier is needed, but just in case.
            dist.barrier()

            # Take another step and save the outputs
            mean_loss_fn(post_step_outputs).backward()
            optim.step()
            optim.zero_grad()
            post_second_step_outputs = model_ep(inputs).logits
            # Sanity check:
            assert not torch.allclose(post_step_outputs, post_second_step_outputs)

            # Try to reload on half the ranks.
            if self.rank < self.world_size // 2:
                torch.manual_seed(seed + 1)
                ep_mesh = init_device_mesh(
                    self.device_type, (self.world_size // 2,), mesh_dim_names=("ep",)
                )
                model_ep = MambaLMHeadModel(
                    self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh
                ).to(self.dtype)

                fully_shard_moe(
                    model_ep,
                    fsdp_mesh=ep_mesh,
                    ep_fsdp_mesh=None,
                )

                optim = torch.optim.SGD(
                    model_ep.parameters(), lr=lr, momentum=self.momentum
                )
                # Reload and overwrite the corrupted state
                reconfig_state_dict = get_dcp_state_dict(model_ep, optim)
                dcp.load(
                    reconfig_state_dict,
                    checkpoint_id=checkpoint_id,
                    process_group=ep_mesh.get_group(),  # NOTE: @goon - hangs, if not specified
                )

                # Check outputs agree pre- and post-taking another step
                reloaded_outputs = model_ep(inputs).logits
                torch.testing.assert_close(
                    post_step_outputs,
                    reloaded_outputs,
                    atol=self.hi_tol,
                    rtol=self.hi_tol,
                )

                # Re-run the backwards against the same data used originally. Requires grad-acc
                # averaging for correctness.

                (mean_loss_fn(reloaded_outputs) / 2).backward()

                # And run on the data that the missing ranks previously ran on:
                torch.manual_seed(42 + self.rank + self.world_size // 2)
                (mean_loss_fn(model_ep(self.get_input_toks()).logits) / 2).backward()

                optim.step()
                optim.zero_grad()

                # Post-step check:
                post_reload_step_outputs = model_ep(inputs).logits
                torch.testing.assert_close(
                    post_reload_step_outputs,
                    post_second_step_outputs,
                    atol=self.hi_tol,
                    rtol=self.hi_tol,
                )

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_attach_tok_count_hooks(self) -> None:
        # Test fuctionality
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        hook_dict = attach_tok_count_hooks(model)
        assert len(hook_dict) == sum(isinstance(m, MoE) for m in model.modules())
        inputs = self.get_input_toks()
        model(inputs)
        assert not hook_dict.is_reduced
        hook_dict.all_reduce()
        assert hook_dict.is_reduced
        hook_dict.reduce()
        assert hook_dict.is_reduced

        # Test collectives correctness
        for h in hook_dict.values():
            h.value = torch.ones_like(h.value)
        hook_dict.all_reduce()
        for h in hook_dict.values():
            assert torch.allclose(h.value, self.world_size * torch.ones_like(h.value))

        hook_dict.reduce()
        for h in hook_dict.values():
            if not self.rank:
                assert torch.allclose(
                    h.value, self.world_size**2 * torch.ones_like(h.value)
                )
            else:
                assert torch.allclose(
                    h.value, self.world_size * torch.ones_like(h.value)
                )

        hook_dict.reset()
        assert not hook_dict.is_reduced

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_attach_magnitude_hooks(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        init_moe(model)
        # Attach to blocks, mixers, and the lm head instance
        hook_dict = attach_magnitude_hooks(model, [Block, MHA, Mamba2, model.lm_head])
        assert not hook_dict.is_reduced
        assert len(hook_dict) == 2 * len(model.backbone.layers) + 1
        inputs = self.get_input_toks()
        model(inputs)
        hook_dict.all_reduce(op=dist.ReduceOp.AVG)
        hook_dict.reduce(op=dist.ReduceOp.AVG)
        assert hook_dict.is_reduced
        hook_dict.reset()
        assert not hook_dict.is_reduced

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_apply_loss_free_moe_balancing(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        hook_dict = attach_tok_count_hooks(model)
        inputs = self.get_input_toks()
        model(inputs)
        pre_biases = [
            m.gate.bias.detach().clone() for m in model.modules() if isinstance(m, MoE)
        ]
        hook_dict.all_reduce()
        apply_loss_free_moe_balancing(1.0, model, hook_dict)
        post_biases = [
            m.gate.bias.detach().clone() for m in model.modules() if isinstance(m, MoE)
        ]
        for b0, b1 in zip(pre_biases, post_biases):
            assert not torch.allclose(b0, b1)

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_clip_grad_norm_simple_model(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(
            self.world_size, self.in_features, device=self.device, dtype=torch.float32
        )
        world_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("world",),
        )
        split_mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("outer", "inner"),
        )

        # Test on very simple model sharded across different meshes.
        model = nn.Sequential(
            nn.Linear(
                self.in_features,
                self.in_features,
                bias=False,
                device=self.device,
            ),
            nn.Linear(
                self.in_features,
                self.in_features,
                bias=False,
                device=self.device,
            ),
        )
        for p in model.parameters():
            nn.init.normal_(p, std=1 / self.in_features**0.5)

        model_sharded = deepcopy(model)
        fully_shard(model_sharded[0], mesh=world_mesh)
        fully_shard(model_sharded[1], mesh=split_mesh)
        fully_shard(model_sharded, mesh=world_mesh)

        outputs = model(inputs)
        outputs_sharded = model_sharded(inputs.tensor_split(self.world_size)[self.rank])
        torch.testing.assert_close(
            outputs_sharded,
            outputs.tensor_split(self.world_size)[self.rank],
            atol=self.hi_tol,
            rtol=self.hi_tol,
        )
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_sharded).backward()

        # Verify correct params and  grads:
        for lin, lin_sharded in zip(model, model_sharded):
            # Sanity check
            weight = lin.weight
            weight_sharded = lin_sharded.weight.full_tensor()
            torch.testing.assert_close(
                weight_sharded, weight, atol=self.hi_tol, rtol=self.hi_tol
            )

            grad = lin.weight.grad
            grad_sharded = lin_sharded.weight.grad.full_tensor()
            torch.testing.assert_close(
                grad_sharded, grad, atol=self.hi_tol, rtol=self.hi_tol
            )

        # Norm and ensure it's non-trivial
        max_norm = 1.0
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if norm < max_norm:
            max_norm = norm / 2
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        assert norm > max_norm, "Clip was trivial"

        norm_sharded = clip_grad_norm_(model_sharded.parameters(), max_norm)
        torch.testing.assert_close(
            norm_sharded, norm, atol=self.hi_tol, rtol=self.hi_tol
        )

        # Clipping again should effectively be a no-op and return the max_norm
        assert torch.allclose(
            nn.utils.clip_grad_norm_(model.parameters(), max_norm),
            torch.tensor(max_norm, device=self.device),
        )
        assert torch.allclose(
            clip_grad_norm_(model_sharded.parameters(), max_norm),
            torch.tensor(max_norm, device=self.device),
        )

        # Verify correct grads post-clip:
        for lin, lin_sharded in zip(model, model_sharded):
            grad = lin.weight.grad
            grad_sharded = lin_sharded.weight.grad.full_tensor()
            torch.testing.assert_close(
                grad_sharded, grad, atol=self.hi_tol, rtol=self.hi_tol
            )


class TestE2E(_TestBase):
    def train_loop_ep(self: _TestBase, ep: int, attn_only: bool):
        with self.temp_dir() as checkpoint_id:
            torch.manual_seed(42)
            fsdp_mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("fsdp",)
            )
            if ep == self.world_size:
                ep_mesh = init_device_mesh(
                    self.device_type, (self.world_size,), mesh_dim_names=("ep_inner",)
                )
            else:
                ep_mesh = init_device_mesh(
                    self.device_type,
                    (self.world_size // ep, ep),
                    mesh_dim_names=("ep_outer", "ep_inner"),
                )

            cfg = self.attn_only_cfg if attn_only else self.cfg
            model = MambaLMHeadModel(cfg, **self.factory_kwargs)

            with torch.device("meta"):
                model_ep = MambaLMHeadModel(
                    cfg, **self.factory_kwargs, ep_mesh=ep_mesh["ep_inner"]
                )

            dtype = torch.bfloat16
            mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

            act_ckpt_moe(model_ep)
            fully_shard_moe(
                model_ep,
                fsdp_mesh=fsdp_mesh,
                ep_fsdp_mesh=None if ep_mesh.ndim == 1 else ep_mesh["ep_outer"],
                mp_policy=mp_policy,
            )
            init_moe(model_ep)
            _copy_params(model, model_ep)

            # For balancing loss:
            tok_hook_dict = attach_tok_count_hooks(model)
            tok_hook_dict_ep = attach_tok_count_hooks(model_ep)

            optim = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=self.momentum
            )
            optim_ep = torch.optim.SGD(
                model_ep.parameters(), lr=self.lr, momentum=self.momentum
            )

            inputs = self.get_input_toks()
            outputs = model(inputs).logits

            inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
            outputs_ep = model_ep(inputs_ep).logits
            torch.testing.assert_close(
                outputs_ep,
                outputs.to(outputs_ep).tensor_split(self.world_size, dim=0)[self.rank],
                atol=self.tol,
                rtol=self.tol,
            )

            # Grads should match with an avg-over-batches type loss.
            flattened_cross_entropy(outputs, inputs).backward()
            flattened_cross_entropy(outputs_ep, inputs_ep).backward()

            # balancing loss
            apply_loss_free_moe_balancing(
                self.lr, model, tok_hook_dict, verify_reduced=False
            )
            tok_hook_dict_ep.all_reduce()
            apply_loss_free_moe_balancing(self.lr, model_ep, tok_hook_dict_ep)

            # Clip and make sure the clip is non-trivial:
            max_norm = 1.0
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if norm.item() < max_norm:
                max_norm = norm / 2
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            norm_ep = clip_grad_norm_(model_ep.parameters(), max_norm)
            assert norm.item() > 0.0, f"{norm=}"
            assert norm_ep.item() > 0.0, f"{norm_ep=}"
            torch.testing.assert_close(norm_ep, norm, atol=self.tol, rtol=self.tol)

            _test_grads(model, model_ep, tol=self.tol)

            # Step and compare post-step outputs:
            optim.step()
            optim.zero_grad()
            optim_ep.step()
            optim_ep.zero_grad()
            with torch.no_grad():
                outputs = model(inputs).logits
                outputs_ep = model_ep(inputs_ep).logits
                torch.testing.assert_close(
                    outputs_ep,
                    outputs.to(outputs_ep).tensor_split(self.world_size, dim=0)[
                        self.rank
                    ],
                    atol=self.tol,
                    rtol=self.tol,
                )

            # Save state
            state_dict = get_dcp_state_dict(model_ep, optim_ep)
            dcp.save(state_dict, checkpoint_id=checkpoint_id)
            # Not sure the barrier is needed, but just in case.
            dist.barrier()

            # Corrupt state
            flattened_cross_entropy(model_ep(inputs_ep).logits, inputs_ep).backward()
            optim_ep.step()
            optim_ep.zero_grad()
            corrupted_outputs_ep = model_ep(inputs_ep).logits
            # Sanity check the model changed
            assert not torch.allclose(
                corrupted_outputs_ep, outputs_ep, atol=self.tol, rtol=self.tol
            )

            # Reload and check state restored
            state_dict_again = get_dcp_state_dict(model_ep, optim_ep)
            dcp.load(state_dict_again, checkpoint_id=checkpoint_id)
            reloaded_outputs_ep = model_ep(inputs_ep).logits
            torch.testing.assert_close(
                reloaded_outputs_ep, outputs_ep, atol=self.tol, rtol=self.tol
            )

    def train_loop_pp(self: _TestBase, pp: int):
        """
        PP only loop
        """
        with self.temp_dir() as checkpoint_id:
            pp_mesh = init_device_mesh(self.device_type, (pp,), mesh_dim_names=("pp",))
            # Need to avoid returning a CausalLMOutput object for PP, otherwise internal shape checks fail.
            pp_cfg = deepcopy(self.cfg)
            pp_cfg.return_logits = True

            model = MambaLMHeadModel(pp_cfg, **self.factory_kwargs)
            # Create a non-sharded version of the model with proper init to first populate the ref model's
            # weights:
            torch.manual_seed(42)
            with torch.device("meta"):
                model_pp = MambaLMHeadModel(pp_cfg, **self.factory_kwargs, ep_mesh=None)

            init_moe(model_pp)
            _copy_params(model, model_pp)

            # Then delete the unnecessary layers. We we would really delete these prior to moving weights
            # to GPU with init_moe, but this is the easiest way to ensure matching weights.
            set_pp_layers(
                model_pp, n_stages=pp_mesh.size(), stage_idx=pp_mesh.get_local_rank()
            )

            # For balancing loss:
            tok_hook_dict = attach_tok_count_hooks(model)
            tok_hook_dict_pp = attach_tok_count_hooks(model_pp)

            act_ckpt_moe(model_pp)

            lr = 1e-1
            optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.momentum)
            optim_pp = torch.optim.SGD(
                model_pp.parameters(), lr=lr, momentum=self.momentum
            )

            # Set input/output forms. Lets us avoid the stage attempting to auto-determine shapes.
            is_first = pp_mesh.get_local_rank() == 0
            is_last = pp_mesh.get_local_rank() == pp_mesh.size() - 1

            input_args, output_args = self.get_pp_input_output_args(is_first, is_last)

            stage = PipelineStage(
                model_pp,
                pp_mesh.get_local_rank(),
                pp_mesh.size(),
                self.device,
                group=pp_mesh.get_group(),
                input_args=input_args,
                output_args=output_args,
            )

            # Create pipeline schedule
            losses_pp = []
            n_microbatches = pp_mesh.size()
            inputs = self.get_input_toks(batch_size=self.batch_size * n_microbatches)
            pp_schedule = Schedule1F1B(
                stage, n_microbatches, loss_fn=flattened_cross_entropy
            )

            is_first = pp_mesh.get_local_rank() == 0
            is_last = pp_mesh.get_local_rank() == pp_mesh.size() - 1

            if is_first:
                pp_schedule.step(inputs)
                out_pp = None
            elif is_last:
                out_pp = pp_schedule.step(target=inputs, losses=losses_pp)
            else:
                pp_schedule.step()
                out_pp = None

            # Run reference model on same data.
            out = model(inputs)
            loss = flattened_cross_entropy(out, inputs)
            loss.backward()

            if is_last:
                assert len(losses_pp) == n_microbatches
                torch.testing.assert_close(out, out_pp, atol=self.tol, rtol=self.tol)
                torch.testing.assert_close(
                    torch.stack(losses_pp).mean(), loss, atol=self.tol, rtol=self.tol
                )
            else:
                assert len(losses_pp) == 0

            # balancing loss. No need to reduce since there is no EP dim
            apply_loss_free_moe_balancing(
                self.lr, model, tok_hook_dict, verify_reduced=False
            )
            apply_loss_free_moe_balancing(
                self.lr, model_pp, tok_hook_dict_pp, verify_reduced=False
            )

            # TODO: @goon - update _test_grads for pp and use here
            _test_grads(model, model_pp, tol=self.tol)

            max_norm = 1.0
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if norm.item() < max_norm:
                max_norm = norm / 2
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            norm_pp = clip_grad_norm_(model_pp.parameters(), max_norm, pp_mesh=pp_mesh)
            torch.testing.assert_close(norm, norm_pp, atol=self.tol, rtol=self.tol)

            # Steps
            optim.step()
            optim.zero_grad()
            optim_pp.step()
            optim_pp.zero_grad()

            state_dict = get_dcp_state_dict(model_pp, optim_pp)
            dcp.save(state_dict, checkpoint_id=checkpoint_id)
            # Not sure the barrier is needed, but just in case.
            dist.barrier()

            # Corrupt state
            if is_first:
                pp_schedule.step(inputs)
            elif is_last:
                out_pp_post_step = pp_schedule.step(target=inputs, losses=losses_pp)
                # Sanity check the model changed
                assert not torch.allclose(
                    out_pp_post_step, out_pp, atol=self.tol, rtol=self.tol
                )
            else:
                pp_schedule.step()
            optim_pp.step()
            optim_pp.zero_grad()

            # Verify updated outputs match ref model:
            if is_last:
                out_post_step = model(inputs)
                torch.testing.assert_close(
                    out_pp_post_step, out_post_step, atol=self.tol, rtol=self.tol
                )

            # Reload and check state restored
            state_dict_again = get_dcp_state_dict(model_pp, optim_pp)
            dcp.load(state_dict_again, checkpoint_id=checkpoint_id)

            if is_first:
                pp_schedule.step(inputs)
            elif is_last:
                out_pp_post_step_again = pp_schedule.step(
                    target=inputs, losses=losses_pp
                )
                torch.testing.assert_close(out_pp_post_step, out_pp_post_step_again)
            else:
                pp_schedule.step()

    # NOTE: @goon - currently using self.attn_only_cfg in fully_shard_moe tests to avoid
    # complications with non-deterministic mamba D grads. The fully_shard_moe util is independent of
    # having mamba layers or not.
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    # @pytest.mark.parametrize("attn_only", [True, False])
    @pytest.mark.parametrize("attn_only", [True])
    def test_ep(self, attn_only: bool) -> None:
        self.train_loop_ep(ep=self.world_size, attn_only=attn_only)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    # @pytest.mark.parametrize("attn_only", [True, False])
    @pytest.mark.parametrize("attn_only", [True])
    def test_ep_replicated(self, attn_only: bool) -> None:
        self.train_loop_ep(ep=self.world_size // 2, attn_only=attn_only)

    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_pp(self) -> None:
        self.train_loop_pp(pp=self.world_size)


def compile_breaking_fn(
    x: torch.Tensor,
    indices: torch.LongTensor,
    ep_mesh,
    n_routed_experts,
    n_local_experts: int = 8,
) -> torch.Tensor:
    # Sort tokens by the expert they are indexed to.
    flat_sorted_indices = indices.flatten().argsort(dim=-1)
    n_activated_experts = indices.shape[-1]
    x_by_expert = x[flat_sorted_indices // n_activated_experts]
    counter = TokenCounter()
    counts = counter(indices, n_routed_experts)
    assert ep_mesh is not None  # mypy
    tokens_per_expert_group = funcol.all_to_all_single(
        counts, None, None, group=ep_mesh
    )

    # We need the list version of the counts due to NCCL signatures. This incurs a CUDA sync.
    # TODO: avoid https://github.com/NVIDIA/nccl/issues/1648
    send_counts = counts.reshape(ep_mesh.size(), n_local_experts).sum(dim=1).tolist()
    recv_counts = tokens_per_expert_group.reshape(ep_mesh.size(), n_local_experts).sum(
        dim=1
    )
    recv_counts = [int(x) for x in recv_counts]

    # Receive toks from other workers
    x_recv = funcol.all_to_all_single_autograd(
        x_by_expert, recv_counts, send_counts, group=ep_mesh
    )
    return x_recv


class TestCompileBreaking(_TestBase):
    @pytest.mark.skip(reason="Failing for upstream pytorch reasons, it seems")
    @pytest.mark.world_size(2)
    @pytest.mark.gpu
    def test_fwd(self) -> None:
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        fn_compiled = torch.compile(compile_breaking_fn)
        for seed in range(3):
            inputs, _, indices, counts = self.get_inputs_weights_indices_counts(
                seed=seed
            )
            fn_compiled(inputs, indices, ep_mesh, self.n_routed_experts)
