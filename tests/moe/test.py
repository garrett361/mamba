from copy import deepcopy
from typing import Literal

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import (
    _GROUPED_MM_ALIGNMENT,
    NON_EP_EXPERT_CLASSES,
    NON_EP_EXPERT_CLASSES_AND_SIMPLE,
    Gate,
    MoE,
    RoutedExpertsNoEPForLoop,
    TokenCounter,
    _get_exp_outputs_grouped_mm,
    _get_exp_outputs_titan_cg_gemm,
    _get_local_expert_idxs,
    _get_single_exp_output,
    _RoutedExpertsNoEP,
    _SimpleRoutedExperts,
)
from mamba_ssm.moe_utils import (
    TokenCounterHook,
    attach_magnitude_hooks,
    attach_tok_count_hooks,
    init_moe,
)
from mamba_ssm.ops.triton.moe import pad_sorted_idxs, pad_sorted_idxs_torch
from tests.moe.test_utils import (
    mean_loss_fn,
    skip_if_no_h100s,
    skip_moe_impl_if_no_h100s,
)


def _copy_params_routed_experts(
    exp: _RoutedExpertsNoEP | _SimpleRoutedExperts,
    exp_other: _RoutedExpertsNoEP | _SimpleRoutedExperts,
) -> None:
    if isinstance(exp_other, _SimpleRoutedExperts):
        _copy_params_routed_experts(exp_other, exp)
    elif isinstance(exp, _SimpleRoutedExperts):
        with torch.no_grad():
            fc1_weight = torch.stack(
                [e.fc1.weight.data for e in exp.experts.values()], dim=0
            )
            fc2_weight = torch.stack(
                [e.fc2.weight.data for e in exp.experts.values()], dim=0
            )
            exp_other.fc1.weight.data.copy_(fc1_weight)
            exp_other.fc2.weight.data.copy_(fc2_weight)
    else:
        with torch.no_grad():
            exp_other.fc1.weight.data.copy_(exp.fc1.weight.data)
            exp_other.fc2.weight.data.copy_(exp.fc2.weight.data)


def _test_grads_routed_experts(
    simple_exp: _SimpleRoutedExperts, exp: _RoutedExpertsNoEP
) -> None:
    assert isinstance(simple_exp, _SimpleRoutedExperts)
    assert isinstance(exp, _RoutedExpertsNoEP)
    with torch.no_grad():
        # Need to handle the case where some of the simple_exp weights may have None grads
        simple_fc1_grads = []
        simple_fc2_grads = []
        for e in simple_exp.experts.values():
            simple_fc1_grads.append(
                e.fc1.weight.grad
                if e.fc1.weight.grad is not None
                else torch.zeros_like(e.fc1.weight)
            )
            simple_fc2_grads.append(
                e.fc2.weight.grad
                if e.fc2.weight.grad is not None
                else torch.zeros_like(e.fc2.weight)
            )

        simple_fc1_grads = torch.stack(simple_fc1_grads, dim=0)
        simple_fc2_grads = torch.stack(simple_fc2_grads, dim=0)
        torch.testing.assert_close(simple_fc1_grads, exp.fc1.weight.grad)
        torch.testing.assert_close(simple_fc2_grads, exp.fc2.weight.grad)


class _TestBase:
    in_features = 256
    d_intermediate = 2 * in_features
    n_routed_experts = 16
    n_shared_experts = 1
    n_activated_experts = 2
    n_layer = 2
    vocab_size = 512
    tie_embeddings = False

    batch_size = 2
    # ~16 toks/expert
    seqlen = int(16 * n_routed_experts / n_activated_experts / batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = {"layer": "Mamba2"}
    head_dim = 64
    attn_layer_idx = [n_layer - 1]
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
    moe_cfg = {
        "n_routed_experts": 16,
        "n_activated_experts": 1,
        "n_shared_experts": 1,
        "d_intermediate": 64,
    }

    cfg = MambaConfig(
        d_model=in_features,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        moe_layer_idx=moe_layer_idx,
        moe_cfg=moe_cfg,
        ssm_cfg=ssm_cfg,
    )
    tol = 1e-2

    def get_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size, self.seqlen, self.in_features, **self.factory_kwargs
        )

    def get_input_toks(self) -> torch.Tensor:
        return torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
        )

    def get_inputs_weights_indices_counts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
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


class TestGate(_TestBase):
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd(self, score_func: Literal["sigmoid", "softmax"]) -> None:
        torch.manual_seed(42)
        model = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func=score_func,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs().view(-1, self.in_features)
        weights, indices, counts = model(inputs)
        assert weights.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert indices.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert counts.shape == torch.Size([self.n_routed_experts])

    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd_with_exp_groups(
        self, score_func: Literal["sigmoid", "softmax"]
    ) -> None:
        torch.manual_seed(42)
        model = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_expert_groups=4,
            n_limited_groups=2,
            score_func=score_func,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs().view(-1, self.in_features)
        weights, indices, counts = model(inputs)
        assert weights.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert indices.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert counts.shape == torch.Size([self.n_routed_experts])

    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_bwd_with_exp_groups(
        self, score_func: Literal["sigmoid", "softmax"]
    ) -> None:
        """
        Regression test: using grouped expert options were previously giving in-place modification
        errors for bwd.
        """
        torch.manual_seed(42)
        model = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_expert_groups=4,
            n_limited_groups=2,
            score_func=score_func,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs().view(-1, self.in_features)
        weights, indices, counts = model(inputs)
        weights.sum().backward()


class TestRoutedExperts(_TestBase):
    def test_no_ep_naive(self) -> None:
        torch.manual_seed(42)
        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts()
        experts = RoutedExpertsNoEPForLoop(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )
        outputs = experts(inputs, weights, indices, counts)
        assert outputs.shape == inputs.shape

    @pytest.mark.parametrize("cls", list(NON_EP_EXPERT_CLASSES.values()))
    def test_equivalence(self, cls) -> None:
        skip_moe_impl_if_no_h100s(cls)
        torch.manual_seed(42)
        inputs, weights, indices, counts = self.get_inputs_weights_indices_counts()
        kwargs = dict(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            **self.factory_kwargs,
        )

        simple_exp = _SimpleRoutedExperts(**kwargs)
        exp = cls(**kwargs)
        _copy_params_routed_experts(simple_exp, exp)
        out = simple_exp(inputs, weights, indices, counts)
        out_other = exp(inputs, weights, indices, counts)

        # Test logits agreement.
        torch.testing.assert_close(out_other, out, atol=self.tol, rtol=self.tol)

        # Test grad agreement.
        mean_loss_fn(out).backward()
        mean_loss_fn(out_other).backward()
        _test_grads_routed_experts(simple_exp, exp)

    @pytest.mark.parametrize("cls", list(NON_EP_EXPERT_CLASSES.values()))
    def test_no_toks(self, cls) -> None:
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

        simple_exp = _SimpleRoutedExperts(**kwargs)
        exp = cls(**kwargs)
        _copy_params_routed_experts(simple_exp, exp)
        out = simple_exp(inputs, weights, indices, counts)
        out_other = exp(inputs, weights, indices, counts)

        # Test logits agreement.
        torch.testing.assert_close(out_other, out, atol=self.tol, rtol=self.tol)

        # Test grad agreement.
        mean_loss_fn(out).backward()
        mean_loss_fn(out_other).backward()
        _test_grads_routed_experts(simple_exp, exp)


class TestMoE(_TestBase):
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd(self, score_func: Literal["sigmoid", "softmax"]) -> None:
        torch.manual_seed(42)
        model = MoE(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_shared_experts=self.n_shared_experts,
            score_func=score_func,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs()
        outputs = model(inputs)
        assert outputs.shape == inputs.shape


class TestMoEModel(_TestBase):
    @pytest.mark.parametrize("moe_impl", list(NON_EP_EXPERT_CLASSES_AND_SIMPLE))
    def test_fwd(self, moe_impl) -> None:
        skip_moe_impl_if_no_h100s(NON_EP_EXPERT_CLASSES_AND_SIMPLE[moe_impl])
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        cfg.moe_cfg["moe_impl"] = moe_impl
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        for layer_idx in sorted(model.backbone.layers):
            mlp = model.backbone.layers[layer_idx].mlp
            if int(layer_idx) in self.moe_layer_idx:
                assert isinstance(mlp, MoE)
            else:
                assert isinstance(mlp, GatedMLP)
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        assert outputs.shape == inputs.shape + torch.Size([self.vocab_size])

    @pytest.mark.parametrize("moe_impl", list(NON_EP_EXPERT_CLASSES_AND_SIMPLE))
    def test_fwd_compile(self, moe_impl) -> None:
        skip_moe_impl_if_no_h100s(NON_EP_EXPERT_CLASSES_AND_SIMPLE[moe_impl])
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        cfg.moe_cfg["moe_impl"] = moe_impl
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        model = torch.compile(model)
        for layer_idx in sorted(model.backbone.layers):
            mlp = model.backbone.layers[layer_idx].mlp
            if int(layer_idx) in self.moe_layer_idx:
                assert isinstance(mlp, MoE)
            else:
                assert isinstance(mlp, GatedMLP)
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        assert outputs.shape == inputs.shape + torch.Size([self.vocab_size])

    @pytest.mark.gpu
    def test_meta_init(self) -> None:
        torch.manual_seed(42)
        with torch.device("meta"):
            meta_model = MambaLMHeadModel(self.cfg)
        assert all(p.device == torch.device("meta") for p in meta_model.parameters())
        init_moe(meta_model, verbose=False)
        assert all(p.device.type == "cuda" for p in meta_model.parameters())
        inputs = self.get_input_toks()
        meta_model(inputs)


def test_bincount_impl_equiv():
    """
    The DeepSeek-v3 repo uses `torch.bincount` which incurs a CUDA sync. Test equivalence with the
    code from torchtitan
    """
    torch.manual_seed(42)
    seqlen = 4096
    n_routed_experts = 64
    n_activated_experts = 8
    scores = torch.randn(seqlen, n_routed_experts)
    # Crucial: indices must be unique, which topk ensures
    indices = scores.topk(n_activated_experts, dim=-1)[1]

    counts_bincount = torch.bincount(indices.flatten(), minlength=n_routed_experts)

    # [seq_len, n_routed_experts]
    counts_scatter = indices.new_zeros((indices.shape[0], n_routed_experts))
    # Fill 1 to the selected experts
    counts_scatter.scatter_(1, indices, 1)
    counts_scatter = counts_scatter.sum(dim=0)
    torch.testing.assert_close(counts_bincount, counts_scatter)


class TestTitan(_TestBase):
    @pytest.mark.parametrize("num_ranks", [8, 64])
    @pytest.mark.parametrize("experts_per_rank", [4, 8])
    def test_generate_permute_indices(
        self, num_ranks: int, experts_per_rank: int
    ) -> None:
        """
        Working through understanding generate_permute_indices. The purpose of this function is that
        it takes the toks received from the all-to-all and makes them contiguous by local expert
        idx, whereas they're naturally instead in blocks ordered by sending rank, and ordered
        within each block by local expert idx.


        - tokens_per_expert_group ~ (num_ranks * exp_per_rank, ): tokens per expert

        - start_index_values = ( torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group)
          : index values where the tokens for each expert start in tokens_per_expert_group

        - chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0) ~ (exp_per_rank,) :
          num tokens per expert group. It's like column major because of how the all-to-all recv tok
          order is.

        - m_sizes: chunk_size_per_expert rounded up to the alignment criteria

        - m_offsets = torch.cumsum(m_sizes, 0) cumulative number of toks per expert group with
          alignment

        - write_offsets = torch.cumsum(m_sizes, 0) - m_sizes : starting points for each expert
          group's tokens

        - permuted_indices ~ (max_len, ) : max_len is like max expected number of tokens
        """
        from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

        torch.manual_seed(42)

        alignment = 16
        tokens_per_expert_group = torch.randint(
            4 * alignment,
            size=(num_ranks * experts_per_rank,),
            dtype=torch.int32,
            device=self.device,
        )

        # The tokens will be scattered into a tensor of size max_len which can accommodate them.
        # Set max_len to twice the number of tokens. Should be big enough
        max_len = 2 * tokens_per_expert_group.sum().item()

        permuted_indices_gpu, m_sizes, m_offsets = generate_permute_indices(
            tokens_per_expert_group,
            experts_per_rank,
            num_ranks,
            2 * max_len,
            alignment,
        )

        # permuted_indices_gpu should be equivalent to the below.

        # 1) Get the tok-to-local-exp-idx map
        local_expert_idxs = _get_local_expert_idxs(
            tokens_per_expert_group, experts_per_rank
        )
        # 2) Sort
        local_expert_idxs_argsort = local_expert_idxs.argsort().to(torch.int32)
        # The non-filler indices of permuted_indices_gpu should match the sorted idxs already
        non_filler_idxs = torch.where(permuted_indices_gpu != -1)[0]
        torch.testing.assert_close(
            local_expert_idxs_argsort, permuted_indices_gpu[non_filler_idxs]
        )

        # 3) Build the full tensor by padding out to the alignment criteria.
        idxs_align, sizes, offsets = pad_sorted_idxs_torch(
            tokens_per_expert_group.view(-1, experts_per_rank).sum(
                dim=0, dtype=torch.int32
            ),
            tokens_per_expert_group.sum(),
            alignment,
        )
        permuted_indices_gpu_alt = torch.full_like(permuted_indices_gpu, -1)
        permuted_indices_gpu_alt[idxs_align] = local_expert_idxs_argsort

        # The non-filler values should match
        torch.testing.assert_close(offsets, m_offsets)
        torch.testing.assert_close(sizes, m_sizes)
        torch.testing.assert_close(permuted_indices_gpu_alt, permuted_indices_gpu)

    def test_grouped_mm_equal(self) -> None:
        """
        torch's native GEMM

        Contiguous inputs logically separated into N groups which each pass through N different
        matmuls.

        """
        skip_if_no_h100s()
        torch.manual_seed(42)
        # Matmul dims all need to be divisible by 16 (everything but n_groups)
        n_experts = 4
        d_model = 16
        d_out = 4 * d_model
        bsz = 2
        seqlen = 8 * n_experts
        n_toks = bsz * seqlen

        toks = torch.randn(
            n_toks, d_model, device=self.device, dtype=self.dtype, requires_grad=True
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        # Send equal numbers of toks to each expert.
        offs = torch.arange(
            n_toks // n_experts,
            n_toks + 1,
            n_toks // n_experts,
            device=self.device,
            dtype=torch.int32,
        )

        out = torch._grouped_mm(
            toks, weight.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16
        )
        assert out.shape == torch.Size([n_toks, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(toks_copy.chunk(n_experts), weight_copy):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(
            out,
            out_alt,
            atol=self.tol,
            rtol=self.tol,
        )

        # Backwards
        # # For some reason, out.sum().backward() errors out?  Most other ops are fine, though.
        # out.sum().backward()
        # out_alt.sum().backward()
        grad = torch.randn_like(out)
        out.backward(grad)
        out_alt.backward(grad)

        torch.testing.assert_close(
            weight.grad, weight_copy.grad, atol=self.tol, rtol=self.tol
        )
        torch.testing.assert_close(
            toks.grad, toks_copy.grad, atol=self.tol, rtol=self.tol
        )

    def test_grouped_mm_diff_alignments(self) -> None:
        """
        Two experts, give the second expert twice the number of toks
        """
        skip_if_no_h100s()
        torch.manual_seed(42)
        # Matmul dims all need to be divisible by 16 (everything but n_groups)
        n_experts = 2
        d_model = 16
        d_out = 4 * d_model
        n_toks_expert_0 = 16
        n_toks_expert_1 = 2 * n_toks_expert_0

        toks = torch.randn(
            n_toks_expert_0 + n_toks_expert_1,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        offs = torch.tensor(
            [n_toks_expert_0, n_toks_expert_1], device=self.device
        ).cumsum(dim=0, dtype=torch.int32)

        out = torch._grouped_mm(
            toks, weight.transpose(-2, -1), offs=offs, out_dtype=toks.dtype
        )
        assert out.shape == torch.Size([n_toks_expert_0 + n_toks_expert_1, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(
            toks_copy.chunk(3), (weight_copy[0], weight_copy[1], weight_copy[1])
        ):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(
            out,
            out_alt,
            atol=self.tol,
            rtol=self.tol,
        )

        # Backwards
        # # For some reason, out.sum().backward() errors out? Other ops like mean() or pow(2).sum()
        # # are fine.
        # out.sum().backward()
        # out_alt.sum().backward()
        grad = torch.randn_like(out)
        out.backward(grad)
        out_alt.backward(grad)

        torch.testing.assert_close(
            toks.grad,
            toks_copy.grad,
            atol=self.tol,
            rtol=self.tol,
        )
        try:
            for exp_idx in range(n_experts):
                exp_grad = weight.grad[exp_idx]
                exp_grad_copy = weight_copy.grad[exp_idx]
                # Fails on the uneven weights. Would pass with 0.1 tolerances.
                torch.testing.assert_close(
                    exp_grad, exp_grad_copy, atol=self.tol, rtol=self.tol
                )
                print(f"Grad check passed for {exp_idx=} weights")
        except AssertionError as e:
            raise RuntimeError(f"Grad check failed for {exp_idx=} weights") from e

    def test_grouped_mm_zero_toks(self) -> None:
        """
        Test the case where an expert gets zero tokens
        """
        skip_if_no_h100s()
        torch.manual_seed(42)
        # Matmul dims all need to be divisible by 16 (everything but n_groups)
        n_experts = 4
        d_model = 16
        d_out = 4 * d_model
        bsz = 2
        seqlen = 8 * n_experts
        n_toks = bsz * seqlen

        toks = torch.randn(
            n_toks, d_model, device=self.device, dtype=self.dtype, requires_grad=True
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        # exp_idx=1 gets zero tokens, while exp_idx=2 gets double its share.
        offs = torch.tensor(
            [
                n_toks // n_experts,
                n_toks // n_experts,
                3 * n_toks // n_experts,
                4 * n_toks // n_experts,
            ],
            device=self.device,
            dtype=torch.int32,
        )

        out = torch._grouped_mm(
            toks, weight.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16
        )
        assert out.shape == torch.Size([n_toks, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(
            toks_copy.chunk(n_experts),
            (weight_copy[0], weight_copy[2], weight_copy[2], weight_copy[3]),
        ):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(
            out,
            out_alt,
            atol=self.tol,
            rtol=self.tol,
        )

        # Backwards. Currently hanging on bwd.
        grad = torch.randn_like(out)
        out.backward(grad)
        out_alt.backward(grad)
        #
        # torch.testing.assert_close(
        #     weight.grad, weight_copy.grad, atol=self.tol, rtol=self.tol
        # )
        # torch.testing.assert_close(
        #     toks.grad, toks_copy.grad, atol=self.tol, rtol=self.tol
        # )

    def test_titan_gemm_equal(self) -> None:
        """
        torchtitan GEMM
        """
        skip_if_no_h100s()
        from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
            ALIGN_SIZE_M,
            grouped_gemm_forward,
        )

        torch.manual_seed(42)

        # Tok alignments need to be rounded to ALIGN_SIZE_M, and it seems like we need d_model >=
        # 256? Also maybe needs to be a power of 2.
        n_experts = 4
        d_model = 512
        d_out = 4 * d_model
        bsz = 2
        seqlen = ALIGN_SIZE_M * n_experts
        n_toks = bsz * seqlen

        toks = torch.randn(
            n_toks, d_model, device=self.device, dtype=self.dtype, requires_grad=True
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Send equal numbers of toks to each expert. Note: grouped_gemm_forward uses the sizes, not
        # offsets.
        sizes = torch.full(
            (n_experts,), n_toks // n_experts, device=self.device, dtype=torch.int32
        )

        out = grouped_gemm_forward(toks, weight.view(-1, d_model), sizes)
        assert out.shape == torch.Size([n_toks, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(toks_copy.chunk(n_experts), weight_copy):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(out, out_alt)

        # Backwards. Not yet working (April 18):
        # - titan unit tests failing on shape assertions
        # - misplaced autograd.Function args in GroupedGEMM_mg

        # # For some reason, out.sum().backward() errors out?
        # grad = torch.randn_like(out)
        # out.backward(grad)
        # out_alt.backward(grad)
        #
        # torch.testing.assert_close(
        #     weight.grad, weight_copy.grad, atol=self.tol, rtol=self.tol
        # )
        # torch.testing.assert_close(
        #     toks.grad, toks_copy.grad, atol=self.tol, rtol=self.tol
        # )

    def test_titan_gemm_diff_alignments(self) -> None:
        skip_if_no_h100s()
        from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
            ALIGN_SIZE_M,
            grouped_gemm_forward,
        )

        torch.manual_seed(42)

        # Tok alignments need to be rounded to ALIGN_SIZE_M, and it seems like we need d_model >=
        # 256? Also maybe needs to be a power of 2.
        n_experts = 2
        d_model = 512
        d_out = 4 * d_model
        n_toks_expert_0 = ALIGN_SIZE_M
        n_toks_expert_1 = 2 * ALIGN_SIZE_M

        toks = torch.randn(
            n_toks_expert_0 + n_toks_expert_1,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Send equal numbers of toks to each expert. Note: grouped_gemm_forward uses the sizes, not
        # offsets.
        sizes = torch.tensor(
            [n_toks_expert_0, n_toks_expert_1], device=self.device, dtype=torch.int32
        )

        out = grouped_gemm_forward(toks, weight.view(-1, d_model), sizes)
        assert out.shape == torch.Size([n_toks_expert_0 + n_toks_expert_1, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(
            toks_copy.chunk(3), (weight_copy[0], weight_copy[1], weight_copy[1])
        ):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(out, out_alt)

        # Backwards. Not yet working (April 18):
        # - titan unit tests failing on shape assertions
        # - misplaced autograd.Function args in GroupedGEMM_mg

        # # For some reason, out.sum().backward() errors out?
        # grad = torch.randn_like(out)
        # out.backward(grad)
        # out_alt.backward(grad)
        #
        # torch.testing.assert_close(
        #     weight.grad, weight_copy.grad, atol=self.tol, rtol=self.tol
        # )
        # torch.testing.assert_close(
        #     toks.grad, toks_copy.grad, atol=self.tol, rtol=self.tol
        # )

    def test_titan_cg_gemm_equal(self) -> None:
        """
        torchtitan CG GEMM
        """
        from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_backward import (
            cg_grouped_gemm,
        )

        torch.manual_seed(42)

        n_experts = 4
        d_model = 512
        d_out = 4 * d_model
        bsz = 2
        group_size_m = 128
        seqlen = group_size_m * n_experts
        n_toks = bsz * seqlen

        toks = torch.randn(
            n_toks, d_model, device=self.device, dtype=self.dtype, requires_grad=True
        )
        weight = torch.randn(
            n_experts,
            d_out,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Send equal numbers of toks to each expert. Note: cg_grouped_gemm uses exp ids, not sizes
        # or offsets
        expert_indices = torch.cat(
            [
                torch.full(
                    (n_toks // n_experts,),
                    exp_idx,
                    device=self.device,
                    dtype=torch.int32,
                )
                for exp_idx in range(n_experts)
            ]
        )

        out = cg_grouped_gemm(toks, weight, expert_indices, group_size_m)
        assert out.shape == torch.Size([n_toks, d_out])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        weight_copy = deepcopy(weight)
        out_alt_list = []
        for tok_chunk, exp_weight in zip(toks_copy.chunk(n_experts), weight_copy):
            out_alt_list.append(tok_chunk @ exp_weight.t())
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(out, out_alt)

        # For some reason, out.sum().backward() errors out?
        grad = torch.randn_like(out)
        out.backward(grad)
        out_alt.backward(grad)

        torch.testing.assert_close(
            weight.grad, weight_copy.grad, atol=self.tol, rtol=self.tol
        )
        torch.testing.assert_close(
            toks.grad, toks_copy.grad, atol=self.tol, rtol=self.tol
        )


class TestMoeImpls(_TestBase):
    def test_bmm_equiv(self):
        """
        Testing the equivalence between different impls
        """
        torch.manual_seed(42)
        experts = nn.ModuleDict(
            {
                str(i): GatedMLP(
                    self.in_features,
                    self.d_intermediate,
                    multiple_of=1,
                    **self.factory_kwargs,
                )
                for i in range(self.n_routed_experts)
            }
        )

        # Common gating

        x, weights, indices, counts = self.get_inputs_weights_indices_counts()

        # DeepSeek-v3 impl:
        z_ds = torch.zeros_like(x)
        for i in range(self.n_routed_experts):
            # TODO: @goon - handle no-tokens edge case
            expert = experts[str(i)]
            idx, top = torch.where(indices == i)
            z_ds[idx] += expert(x[idx]) * weights[idx, top, None]

        # Alt impl
        z_alt = torch.empty(
            x.shape[0] * self.n_activated_experts,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

        for exp_idx in range(self.n_routed_experts):
            idxs = indices == exp_idx
            # TODO: @goon - handle no-tokens edge case
            z_alt[idxs.flatten()] = experts[str(exp_idx)](x[idxs.any(dim=-1)])

        z_alt = z_alt.reshape(*(weights.shape + z_alt.shape[-1:]))
        z_alt = torch.bmm(weights[:, None], z_alt).squeeze(1)

        torch.testing.assert_close(z_ds, z_alt, atol=self.tol, rtol=self.tol)

    def test_get_single_exp_output(self):
        torch.manual_seed(42)
        mlp = GatedMLP(
            in_features=self.in_features,
            hidden_features=self.d_intermediate,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs()
        outputs = mlp(inputs)
        outputs_alt = _get_single_exp_output(
            inputs, mlp.fc1.weight, mlp.fc2.weight, mlp.activation
        )
        torch.testing.assert_close(outputs, outputs_alt)

    def test_get_exp_outputs_grouped_mm(self):
        skip_if_no_h100s()
        torch.manual_seed(42)
        # Matmul dims all need to be divisible by 16 (everything but n_groups)
        n_experts = 2
        d_model = 16
        d_intermediate = 4 * d_model
        n_toks_expert_0 = 16
        n_toks_expert_1 = 2 * n_toks_expert_0

        toks = torch.randn(
            n_toks_expert_0 + n_toks_expert_1,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        fc1_weight = torch.randn(
            n_experts,
            2 * d_intermediate,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        fc2_weight = torch.randn(
            n_experts,
            d_model,
            d_intermediate,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        offs = torch.tensor(
            [n_toks_expert_0, n_toks_expert_1], device=self.device
        ).cumsum(dim=0, dtype=torch.int32)

        out = _get_exp_outputs_grouped_mm(
            toks, fc1_weight, fc2_weight, offs=offs, activation=F.silu
        )
        assert out.shape == torch.Size([n_toks_expert_0 + n_toks_expert_1, d_model])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        fc1_weight_copy = deepcopy(fc1_weight)
        fc2_weight_copy = deepcopy(fc2_weight)
        out_alt_list = []
        for tok_chunk, (fc1, fc2) in zip(
            toks_copy.chunk(3),
            zip(
                (fc1_weight_copy[0], fc1_weight_copy[1], fc1_weight_copy[1]),
                (fc2_weight_copy[0], fc2_weight_copy[1], fc2_weight_copy[1]),
            ),
        ):
            out_alt_list.append(_get_single_exp_output(tok_chunk, fc1, fc2, F.silu))
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(
            out,
            out_alt,
            atol=self.tol,
            rtol=self.tol,
        )

        # Backwards: just test that the _grouped_mm bwd doesn't error out for now.
        # TODO: @goon - test correctness.
        out.pow(2).sum().backward()

    def test_get_exp_outputs_titan_cg_gemm(self):
        skip_if_no_h100s()
        torch.manual_seed(42)
        # Matmul dims all need to be divisible by 16 (everything but n_groups)
        n_experts = 2
        d_model = 16
        d_intermediate = 4 * d_model
        n_toks_expert_0 = 16
        n_toks_expert_1 = 2 * n_toks_expert_0

        toks = torch.randn(
            n_toks_expert_0 + n_toks_expert_1,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        fc1_weight = torch.randn(
            n_experts,
            2 * d_intermediate,
            d_model,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        fc2_weight = torch.randn(
            n_experts,
            d_model,
            d_intermediate,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        expert_indices = torch.cat(
            [
                torch.full(
                    (n_toks_expert_0,), 0, device=self.device, dtype=torch.int32
                ),
                torch.full(
                    (n_toks_expert_1,), 1, device=self.device, dtype=torch.int32
                ),
            ]
        )
        out = _get_exp_outputs_titan_cg_gemm(
            toks,
            fc1_weight,
            fc2_weight,
            expert_indices=expert_indices,
            activation=F.silu,
            group_size_m=n_toks_expert_0,
        )
        assert out.shape == torch.Size([n_toks_expert_0 + n_toks_expert_1, d_model])

        # Compute the equivalent for-loop
        toks_copy = deepcopy(toks)
        fc1_weight_copy = deepcopy(fc1_weight)
        fc2_weight_copy = deepcopy(fc2_weight)
        out_alt_list = []
        for tok_chunk, (fc1, fc2) in zip(
            toks_copy.chunk(3),
            zip(
                (fc1_weight_copy[0], fc1_weight_copy[1], fc1_weight_copy[1]),
                (fc2_weight_copy[0], fc2_weight_copy[1], fc2_weight_copy[1]),
            ),
        ):
            out_alt_list.append(_get_single_exp_output(tok_chunk, fc1, fc2, F.silu))
        out_alt = torch.cat(out_alt_list, dim=0)
        torch.testing.assert_close(
            out,
            out_alt,
            atol=self.tol,
            rtol=self.tol,
        )

        # Backwards: just test that the _grouped_mm bwd doesn't error out for now.
        # TODO: @goon - test correctness.
        out.pow(2).sum().backward()


class TestTriton(_TestBase):
    @pytest.mark.parametrize("seqlen", [128, 1024, 8192, 32768])
    @pytest.mark.parametrize("n_routed_experts", [64, 256])
    @pytest.mark.parametrize("n_activated_experts", [8, 16])
    def test_pad_sorted_idxs(
        self, seqlen: int, n_routed_experts: int, n_activated_experts: int
    ) -> None:
        torch.manual_seed(42)
        indices = (
            torch.randn(
                seqlen,
                n_routed_experts,
                device=self.device,
            )
            .topk(n_activated_experts, dim=-1)
            .indices
        )
        counter = TokenCounter()
        counts = counter(indices, n_routed_experts)

        # Torch impl:
        idxs, sizes, offsets = pad_sorted_idxs_torch(
            counts, seqlen * n_activated_experts, _GROUPED_MM_ALIGNMENT
        )

        # Triton impl:
        idxs_alt, sizes_alt, offsets_alt = pad_sorted_idxs(
            counts, seqlen * n_activated_experts, _GROUPED_MM_ALIGNMENT
        )
        for t in (sizes, sizes_alt, offsets, offsets_alt, idxs, idxs_alt):
            assert t.dtype == torch.int32
        torch.testing.assert_close(sizes, sizes_alt)
        torch.testing.assert_close(offsets, offsets_alt)
        torch.testing.assert_close(idxs, idxs_alt)


class TestMoEUtils(_TestBase):
    def test_attach_tok_count_hooks(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        hook_dict = attach_tok_count_hooks(model)
        assert len(hook_dict) == sum(isinstance(m, MoE) for m in model.modules())
        inputs = self.get_input_toks()
        model(inputs)
        for h in hook_dict.values():
            assert h.value.numel() == cfg.moe_cfg["n_routed_experts"]

    def test_count_hooks_raise(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        # Can only attach to modules which own exactly only TokenCounter submodule
        with pytest.raises(RuntimeError):
            TokenCounterHook(model.lm_head)

    def test_attach_magnitude_hooks(self) -> None:
        torch.manual_seed(42)
        cfg = deepcopy(self.cfg)
        model = MambaLMHeadModel(cfg, **self.factory_kwargs)
        init_moe(model)
        hook_dict = attach_magnitude_hooks(model, [Block, model.lm_head])
        assert len(hook_dict) == len(model.backbone.layers) + 1
        inputs = self.get_input_toks()
        iters = 3
        for _ in range(iters):
            model(inputs)
        for h in hook_dict.values():
            assert isinstance(h.value, torch.Tensor)
            assert h._iters == iters
        stats = {n: h.value for n, h in hook_dict.items()}
        # Reset everything
        for h in hook_dict.values():
            h.reset()
            with pytest.raises(RuntimeError):
                h.value
            assert h._iters == 0
