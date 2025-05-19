from copy import deepcopy
from typing import Any

import pytest
import torch
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import (
    EP_EXPERT_CLASSES,
    MoE,
    RoutedExpertsFC1Weights,
    RoutedExpertsFC2Weights,
    RoutedExpertsNoEPForLoop,
    RoutedExpertsTorchEPGroupedMM,
    _get_counts,
    _RoutedExperts,
)
from mamba_ssm.moe_utils import init_meta_moe
from mamba_ssm.moe_utils._utils import fully_shard_moe


def _copy_params_routed_experts(
    experts: _RoutedExperts, experts_ep: _RoutedExperts
) -> None:
    with torch.no_grad():
        experts_ep.fc1.weight.data.copy_(
            experts.fc1.weight.data[
                experts_ep.experts_start_idx : experts_ep.experts_end_idx
            ]
        )
        experts_ep.fc2.weight.data.copy_(
            experts.fc2.weight.data[
                experts_ep.experts_start_idx : experts_ep.experts_end_idx
            ]
        )


def _copy_params(model: nn.Module, model_fsdp: nn.Module) -> None:
    for n, m_fsdp in model_fsdp.named_modules():
        m = model.get_submodule(n)
        if isinstance(m, _RoutedExperts):
            _copy_params_routed_experts(m, m_fsdp)
        elif isinstance(m, (RoutedExpertsFC1Weights, RoutedExpertsFC2Weights)):
            # Already accounted for by the _RoutedExperts path.
            continue
        else:
            with torch.no_grad():
                for p_dest, p_src in zip(
                    m_fsdp.parameters(recurse=False), m.parameters(recurse=False)
                ):
                    p_dest.data.copy_(p_src.data)


def _test_grads_routed_experts(
    experts: _RoutedExperts, experts_ep: _RoutedExperts, tol: float
) -> None:
    for p, p_ep in zip(
        (experts.fc1.weight, experts_ep.fc1.weight),
        (experts.fc1.weight, experts_ep.fc2.weight),
    ):
        if p.grad is None:
            assert p_ep.grad is None
            return
        grad = p.grad
        grad_ep = p_ep.grad
        if isinstance(grad_ep, DTensor):
            grad_ep = grad_ep.full_tensor()
            torch.testing.assert_close(
                grad[experts_ep.experts_start_idx : experts_ep.experts_end_idx],
                grad_ep,
                atol=tol,
                rtol=tol,
            )


def _test_grads(model: nn.Module, model_fsdp: nn.Module, tol: float) -> None:
    with torch.no_grad():
        for n, m_fsdp in model_fsdp.named_modules():
            m = model.get_submodule(n)
            if isinstance(m, _RoutedExperts):
                _test_grads_routed_experts(m, m_fsdp, tol)
            elif isinstance(m, (RoutedExpertsFC1Weights, RoutedExpertsFC2Weights)):
                # Already accounted for by the _RoutedExperts path.
                continue
            else:
                for (n, p), (_, p_fsdp) in zip(
                    m.named_parameters(recurse=False),
                    m_fsdp.named_parameters(recurse=False),
                ):
                    if p.grad is None:
                        assert p_fsdp.grad is None
                        return
                    grad = p.grad
                    grad_fsdp = p_fsdp.grad
                    if isinstance(grad_fsdp, DTensor):
                        grad_fsdp = grad_fsdp.full_tensor()
                    try:
                        torch.testing.assert_close(grad, grad_fsdp, atol=tol, rtol=tol)
                    except Exception as e:
                        raise RuntimeError(f"Failed on {n=}") from e


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

    tol = 1e-2

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

    def get_inputs_weights_indices(
        self, seed: int = 42
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Returns the flattened inputs, weights, and indices used by routed experts.
        """
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
        return inputs, weights, indices


class TestRoutedExperts(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_fwd(self, cls) -> None:
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
        _copy_params_routed_experts(model, model_ep)

        inputs, weights, indices = self.get_inputs_weights_indices(seed=42 + self.rank)
        outputs = model(inputs, weights, indices)
        outputs_ep = model_ep(inputs, weights, indices)

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("cls", list(EP_EXPERT_CLASSES.values()))
    def test_bwd(self, cls) -> None:
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

        inputs, weights, indices = self.get_inputs_weights_indices()
        outputs = model(inputs, weights, indices)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        weights_ep = weights.tensor_split(self.world_size, dim=0)[self.rank]
        indices_ep = indices.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep, weights_ep, indices_ep)

        # Grads should match with an aver-over-batches type loss
        outputs.pow(2).mean().backward()
        outputs_ep.pow(2).mean().backward()

        _test_grads(model, model_ep, tol=self.tol)


class TestMoEEP(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd(self, moe_impl: str) -> None:
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

        # Grads should match with an average-over-batches type loss
        outputs.pow(2).mean().backward()
        outputs_ep.pow(2).mean().backward()

        _test_grads(model, model_ep, tol=self.tol)
        # Verify the routed experts are not sharded and everything else is
        try:
            for n, p in model_ep.named_parameters():
                if n.startswith("experts"):
                    assert not isinstance(p, DTensor)
                else:
                    assert isinstance(p, DTensor)
        except Exception as e:
            raise RuntimeError(f"Failed on {n=}, {p=}") from e


class TestModelEP(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd(self, moe_impl: str) -> None:
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
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an aver-over-batches type loss
        outputs.logits.pow(2).mean().backward()
        outputs_ep.logits.pow(2).mean().backward()

        _test_grads(model, model_ep, tol=self.tol)

        # Verify the routed experts are not sharded and everything else is
        try:
            for n, p in model_ep.named_parameters():
                if ".experts." in n:
                    assert not isinstance(p, DTensor)
                else:
                    assert isinstance(p, DTensor)
        except Exception as e:
            raise RuntimeError(f"Failed on {n=}, {p=}") from e

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd_compile(self, moe_impl: str) -> None:
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
    def test_meta_init(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        with torch.device("meta"):
            meta_model_ep = MambaLMHeadModel(self.cfg, ep_mesh=ep_mesh)

        init_meta_moe(meta_model_ep, verbose=False)
        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs_ep = meta_model_ep(inputs)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    @pytest.mark.parametrize("moe_impl", list(EP_EXPERT_CLASSES))
    def test_fwd_fully_shard_moe(self, moe_impl: str) -> None:
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
        fully_shard_moe(
            model_ep,
            ep_degree=ep_mesh.size(),
            world_size=ep_mesh.size(),
            fsdp_mesh=ep_mesh,
        )

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)


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

    counts = _get_counts(indices, n_routed_experts)
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
            inputs, _, indices = self.get_inputs_weights_indices(seed=seed)
            fn_compiled(inputs, indices, ep_mesh, self.n_routed_experts)
