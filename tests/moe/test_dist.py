from copy import deepcopy
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
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
    get_meshes,
    init_moe,
)
from mamba_ssm.moe_utils._utils import apply_loss_free_moe_balancing
from tests.moe.test_utils import mean_loss_fn, skip_moe_impl_if_no_h100s


def _copy_params(model: nn.Module, model_fsdp: nn.Module) -> None:
    """
    Copy prams from the sharded model to the local model.
    """
    for n, m_fsdp in model_fsdp.named_modules():
        m = model.get_submodule(n)
        with torch.no_grad():
            for p_dest, p_src in zip(
                m.parameters(recurse=False), m_fsdp.parameters(recurse=False)
            ):
                if isinstance(p_src, DTensor):
                    p_src = p_src.full_tensor()
                p_dest.data.copy_(p_src.data)


def _test_grads(model: nn.Module, model_fsdp: nn.Module, tol: float) -> None:
    with torch.no_grad():
        for n, m_fsdp in model_fsdp.named_modules():
            m = model.get_submodule(n)
            for (n, p), (_, p_fsdp) in zip(
                m.named_parameters(recurse=False),
                m_fsdp.named_parameters(recurse=False),
            ):
                if p.grad is None:
                    assert p_fsdp.grad is None
                grad = p.grad
                grad_fsdp = p_fsdp.grad
                if isinstance(grad_fsdp, DTensor):
                    grad_fsdp = grad_fsdp.full_tensor()
                try:
                    torch.testing.assert_close(grad, grad_fsdp, atol=tol, rtol=tol)
                except Exception as e:
                    raise RuntimeError(f"Failed on {m=}, {n=}") from e


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
    def factory_kwargs(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def get_inputs(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.batch_size, self.seqlen, self.in_features, **self.factory_kwargs
        )

    def get_input_toks(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
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


class TestRoutedExperts(_TestBase):
    @pytest.mark.world_size(4)
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

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_bwd(self, cls) -> None:
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

        # Force models equal
        _copy_params(model, model_ep)

        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts()
        outputs = model(inputs, weights, indices, counts)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        weights_ep = weights.tensor_split(self.world_size, dim=0)[self.rank]
        indices_ep = indices.tensor_split(self.world_size, dim=0)[self.rank]
        # The counts need to be rederived from indices_ep
        counts_ep = TokenCounter()(indices_ep, self.n_routed_experts)
        outputs_ep = model_ep(inputs_ep, weights_ep, indices_ep, counts_ep)

        # Note: important to use an avg-type loss here.
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_ep).backward()

        _test_grads(model, model_ep, tol=self.tol)

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
    @pytest.mark.world_size(4)
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

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
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

        inputs = self.get_inputs()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Note: important to use an avg-type loss here.
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_ep).backward()

        _test_grads(model, model_ep, tol=self.tol)


class TestModelEP(_TestBase):
    @pytest.mark.world_size(4)
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

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
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
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(self.cfg)
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
            ignored_params = (
                set(block.mlp.experts.parameters())
                if isinstance(block.mlp, MoE)
                else None
            )
            fully_shard(block, mesh=ep_mesh, ignored_params=ignored_params)

        fully_shard(model_ep, mesh=ep_mesh)

        inputs = self.get_input_toks()
        outputs = model(inputs).logits

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep).logits

        # Grads should match with an avg-over-batches type loss
        F.cross_entropy(
            outputs.view(-1, outputs.size(-1)), inputs.view(-1).long()
        ).backward()
        F.cross_entropy(
            outputs_ep.view(-1, outputs_ep.size(-1)), inputs_ep.view(-1).long()
        ).backward()

        _test_grads(model, model_ep, tol=self.tol)

    @pytest.mark.world_size(4)
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

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_clip_grad(self) -> None:
        # NOTE: @goon - this is only passing because of the special fsdp_mesh=ep_mesh setting.
        # Otherwise, clipping fails with different mesh errors.
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        model_ep = MambaLMHeadModel(
            self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh
        ).to(self.dtype)

        # Force models equal
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_mesh=ep_mesh,
        )

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Populate grads
        mean_loss_fn(outputs.logits).backward()
        mean_loss_fn(outputs_ep.logits).backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_ep = nn.utils.clip_grad_norm_(model_ep.parameters(), 1.0).full_tensor()
        torch.testing.assert_close(norm, norm_ep, atol=self.tol, rtol=self.tol)


class TestMoEUtils(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_meta_init(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        with torch.device("meta"):
            meta_model_ep = MambaLMHeadModel(self.cfg, ep_mesh=ep_mesh)

        init_moe(meta_model_ep, verbose=False)
        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs_ep = meta_model_ep(inputs)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(self.cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        # Force models equal
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_mesh=ep_mesh,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_bwd_fully_shard_moe(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(self.cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        # Force models equal
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_mesh=ep_mesh,
        )

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an avg-over-batches type loss
        mean_loss_fn(outputs.logits).backward()
        mean_loss_fn(outputs_ep.logits).backward()

        _test_grads(model, model_ep, tol=self.tol)

    @pytest.mark.world_size(8)
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

        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(self.cfg)
        model_ep = MambaLMHeadModel(
            moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh["inner"]
        ).to(self.dtype)

        # Force models equal
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=fsdp_mesh,
            ep_mesh=ep_mesh,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(8)
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

        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        moe_cfg = deepcopy(self.cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh).to(
            self.dtype
        )

        # Force models equal
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=hsdp_mesh,
            ep_mesh=ep_mesh,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd_fully_shard_moe_with_mp(self) -> None:
        # Everything in bfloat16
        dtype = torch.bfloat16
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(dtype)
        moe_cfg = deepcopy(self.cfg)
        model_ep = MambaLMHeadModel(moe_cfg, **self.factory_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        fully_shard_moe(
            model_ep,
            fsdp_mesh=ep_mesh,
            ep_mesh=ep_mesh,
            mp_policy=mp_policy,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        outputs_ep = model_ep(inputs).logits
        # NOTE: @goon - currently failing on ~0.8% of all inputs.
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_dcp_save_load(self) -> None:
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
            ep_mesh=ep_mesh,
        )
        # Large LR to create big changes:
        optim = torch.optim.AdamW(model_ep.parameters(), lr=1e-1)
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
        dcp.save(state_dict, checkpoint_id="/tmp/dcp")

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
        dcp.load(corrupted_state_dict, checkpoint_id="/tmp/dcp")

        # Check outputs agree pre- and post-taking another step
        reloaded_outputs = model_ep(inputs).logits
        torch.testing.assert_close(post_step_outputs, reloaded_outputs)

        mean_loss_fn(reloaded_outputs).backward()
        optim.step()
        optim.zero_grad()
        post_reload_step_outputs = model_ep(inputs).logits
        torch.testing.assert_close(post_reload_step_outputs, post_second_step_outputs)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_dcp_save_load_reconfigured(self) -> None:
        """
        Test reloading in a different sharded cfg.
        """
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
            ep_mesh=ep_mesh,
        )
        # Large LR to create big changes:
        lr = 1e-1
        optim = torch.optim.AdamW(model_ep.parameters(), lr=lr)
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
        dcp.save(state_dict, checkpoint_id="/tmp/dcp")

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
                ep_mesh=ep_mesh,
            )

            optim = torch.optim.AdamW(model_ep.parameters(), lr=lr)
            # Reload and overwrite the corrupted state
            reconfig_state_dict = get_dcp_state_dict(model_ep, optim)
            dcp.load(
                reconfig_state_dict,
                checkpoint_id="/tmp/dcp",
                process_group=ep_mesh.get_group(),  # NOTE: @goon - hangs, if not specified
            )

            # Check outputs agree pre- and post-taking another step
            reloaded_outputs = model_ep(inputs).logits
            torch.testing.assert_close(
                post_step_outputs, reloaded_outputs, atol=self.tol, rtol=self.tol
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
                atol=self.tol,
                rtol=self.tol,
            )

        dist.barrier()

    def test_attach_tok_count_hooks(self) -> None:
        # Test fuctionality
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        hook_dict = attach_tok_count_hooks(model)
        assert len(hook_dict) == sum(isinstance(m, MoE) for m in model.modules())
        inputs = self.get_input_toks()
        model(inputs)
        hook_dict.all_reduce()
        hook_dict.reduce()

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

    def test_attach_magnitude_hooks(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        init_moe(model)
        hook_dict = attach_magnitude_hooks(model, Block)
        assert len(hook_dict) == len(model.backbone.layers)
        inputs = self.get_input_toks()
        model(inputs)
        hook_dict.all_reduce()
        hook_dict.reduce()
        hook_dict.reset()

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
        apply_loss_free_moe_balancing(1.0, model, hook_dict)
        post_biases = [
            m.gate.bias.detach().clone() for m in model.modules() if isinstance(m, MoE)
        ]
        for b0, b1 in zip(pre_biases, post_biases):
            assert not torch.allclose(b0, b1)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_get_meshes(self) -> None:
        for hsdp in (False, self.world_size // 2):
            for ep in (False, self.world_size, self.world_size // 2):
                meshes = get_meshes(
                    world_size=self.world_size, hsdp=hsdp, ep=ep, pp=False
                )

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_clip_grad_norm_(self) -> None:
        torch.manual_seed(42)
        meshes = get_meshes(self.world_size, ep=self.world_size)
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs).to(self.dtype)
        model_ep = MambaLMHeadModel(self.cfg, **self.factory_kwargs, ep_mesh=meshes.ep)
        dtype = torch.bfloat16
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        init_moe(model_ep)
        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep, fsdp_mesh=meshes.dp, ep_mesh=meshes.ep, mp_policy=mp_policy
        )

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Populate grads
        mean_loss_fn(outputs.logits).backward()
        mean_loss_fn(outputs_ep.logits).backward()

        # Force the shared expert and non-shared expert grads to have grads of the same size to
        # ensure a non-trivial test. Force the L2 norm of the routed-exp and non-routeed-exp grads
        # to be O(1).
        total_params = sum(p.numel() for p in model.parameters())
        n_moe_params = sum(
            p.numel()
            for m in model.modules()
            for p in m.parameters()
            if isinstance(m, MoE)
        )
        non_moe_params = total_params - n_moe_params
        for mod in (model, model_ep):
            for m in mod.modules():
                if isinstance(m, MoE):
                    for p in m.parameters(recurse=False):
                        if p.grad is not None:
                            nn.init.constant_(p.grad, 1 / n_moe_params**0.5)
                else:
                    for p in m.parameters(recurse=False):
                        if p.grad is not None:
                            nn.init.constant_(p.grad, 1 / non_moe_params**0.5)

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_ep = clip_grad_norm_(model_ep.parameters(), 1.0)
        torch.testing.assert_close(norm, norm_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_e2e_ep(self) -> None:
        torch.manual_seed(42)
        meshes = get_meshes(world_size=self.world_size, ep=self.world_size)
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        model_ep = MambaLMHeadModel(self.cfg, **self.factory_kwargs, ep_mesh=meshes.ep)
        dtype = torch.bfloat16
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=meshes.dp,
            ep_mesh=meshes.ep,
            mp_policy=mp_policy,
        )
        optim = torch.optim.AdamW(model_ep.parameters(), lr=1e-1)

        inputs = self.get_input_toks()
        outputs = model(inputs).logits

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep).logits
        torch.testing.assert_close(
            outputs.to(outputs_ep).tensor_split(self.world_size, dim=0)[self.rank],
            outputs_ep,
            atol=self.tol,
            rtol=self.tol,
        )

        # Grads should match with an avg-over-batches type loss.
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_ep).backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_ep = clip_grad_norm_(model_ep.parameters(), 1.0)
        assert norm.item() > 0.0, f"{norm=}"
        assert norm_ep.item() > 0.0, f"{norm_ep=}"

        torch.testing.assert_close(norm, norm_ep, atol=self.tol, rtol=self.tol)

        _test_grads(model, model_ep, tol=self.tol)

        # Save state
        optim.step()
        optim.zero_grad()
        state_dict = get_dcp_state_dict(model_ep, optim)
        dcp.save(state_dict, checkpoint_id="/tmp/dcp")

        # Reload (just checking for no errors for now)
        state_dict_again = get_dcp_state_dict(model_ep, optim)
        dcp.load(state_dict_again, checkpoint_id="/tmp/dcp")

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_e2e_ep_replicated(self) -> None:
        torch.manual_seed(42)
        # Set ep != self.world_size to induce replicated experts
        meshes = get_meshes(world_size=self.world_size, ep=self.world_size // 2)
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        model_ep = MambaLMHeadModel(self.cfg, **self.factory_kwargs, ep_mesh=meshes.ep)
        dtype = torch.bfloat16
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        _copy_params(model, model_ep)
        fully_shard_moe(
            model_ep,
            fsdp_mesh=meshes.dp,
            ep_mesh=meshes.ep,
            mp_policy=mp_policy,
        )
        optim = torch.optim.AdamW(model_ep.parameters(), lr=1e-1)

        inputs = self.get_input_toks()
        outputs = model(inputs).logits

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep).logits

        torch.testing.assert_close(
            outputs.to(outputs_ep).tensor_split(self.world_size, dim=0)[self.rank],
            outputs_ep,
            atol=self.tol,
            rtol=self.tol,
        )

        # Grads should match with an avg-over-batches type loss.
        mean_loss_fn(outputs).backward()
        mean_loss_fn(outputs_ep).backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_ep = clip_grad_norm_(model_ep.parameters(), 1.0).full_tensor()
        assert norm.item() > 0.0, f"{norm=}"
        assert norm_ep.item() > 0.0, f"{norm_ep=}"

        torch.testing.assert_close(norm, norm_ep, atol=self.tol, rtol=self.tol)

        _test_grads(model, model_ep, tol=self.tol)

        # Save state
        optim.step()
        optim.zero_grad()
        state_dict = get_dcp_state_dict(model_ep, optim)
        dcp.save(state_dict, checkpoint_id="/tmp/dcp")

        # Reload (just checking for no errors for now)
        corrupted_state_dict = get_dcp_state_dict(model_ep, optim)
        dcp.load(corrupted_state_dict, checkpoint_id="/tmp/dcp")


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
    @pytest.mark.world_size(4)
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
