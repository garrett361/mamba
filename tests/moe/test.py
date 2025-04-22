from copy import deepcopy
from typing import Literal

import pytest
import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.moe import (
    Gate,
    MoE,
    RoutedExpertsNoEPNaive,
    _get_single_exp_output,
)


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
    seqlen = 32
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
    tol = 1e-3

    def get_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size, self.seqlen, self.in_features, **self.factory_kwargs
        )

    def get_input_toks(self) -> torch.Tensor:
        return torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
        )


class TestGate(_TestBase):
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd(self, score_func: Literal["sigmoid", "softmax"]) -> None:
        model = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func=score_func,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs().view(-1, self.in_features)
        weights, indices = model(inputs)
        assert weights.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert indices.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )

    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd_with_exp_groups(
        self, score_func: Literal["sigmoid", "softmax"]
    ) -> None:
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
        weights, indices = model(inputs)
        assert weights.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )
        assert indices.shape == inputs.shape[:1] + torch.Size(
            [self.n_activated_experts]
        )


class TestRoutedExperts(_TestBase):
    def get_inputs_weights_indices(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        model = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_expert_groups=4,
            n_limited_groups=2,
            **self.factory_kwargs,
        )
        inputs = self.get_inputs().view(-1, self.in_features)
        weights, indices = model(inputs)
        return inputs, weights, indices

    def test_no_ep_naive(self) -> None:
        inputs, weights, indices = self.get_inputs_weights_indices()
        experts = RoutedExpertsNoEPNaive(
            in_features=self.in_features,
            d_intermediate=self.moe_cfg["d_intermediate"],
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            **self.factory_kwargs,
        )
        outputs = experts(inputs, weights, indices)
        assert outputs.shape == inputs.shape


class TestMoE(_TestBase):
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_fwd(self, score_func: Literal["sigmoid", "softmax"]) -> None:
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
    def test_fwd(self) -> None:
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        for layer_idx in sorted(model.backbone.layers):
            mlp = model.backbone.layers[layer_idx].mlp
            if int(layer_idx) in self.moe_layer_idx:
                assert isinstance(mlp, MoE)
            else:
                assert isinstance(mlp, GatedMLP)
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        assert outputs.shape == inputs.shape + torch.Size([self.vocab_size])


def test_bincount_impl_equiv():
    """
    The DeepSeek-v3 repo uses `torch.bincount` which incurs a CUDA sync. Test equivalence with the
    code from torchtitan
    """
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
    def test_generate_permute_indices(self) -> None:
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

        experts_per_rank = 2
        num_ranks = 8
        # tokens_per_expert_group = torch.arange(
        #     num_ranks * experts_per_rank, dtype=torch.int32, device="cuda"
        # )
        tokens_per_expert_group = torch.arange(
            1, num_ranks * experts_per_rank + 1, dtype=torch.int32, device="cuda"
        )
        # Here we know exactly how many elements are received, which sets max_len. This can be
        # unknown in other comms frameworks.
        max_len = tokens_per_expert_group.sum().item()
        alignment = 16
        permuted_indices_gpu, m_sizes, m_offsets = generate_permute_indices(
            tokens_per_expert_group,
            experts_per_rank,
            num_ranks,
            max_len,
            alignment,
        )

        # permuted_indices_gpu should be equivalent to the below.
        local_expert_idxs = (
            torch.arange(
                tokens_per_expert_group.numel(), device=tokens_per_expert_group.device
            )
            % experts_per_rank
        )
        # NOTE: @goon - repeat_interleave incurs a CUDA sync since it needs to wait on
        # the CUDA tensor tokens_per_expert_group to know the output shape
        local_expert_idxs = local_expert_idxs.repeat_interleave(tokens_per_expert_group)
        local_expert_idxs_argsort = local_expert_idxs.argsort()
        torch.testing.assert_close(
            local_expert_idxs_argsort.to(permuted_indices_gpu), permuted_indices_gpu
        )

    def test_grouped_mm_equal(self) -> None:
        """
        torch's native GEMM

        Contiguous inputs logically separated into N groups which each pass through N different
        matmuls.

        """
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
        torch.testing.assert_close(out, out_alt)

        # Backwards
        # # For some reason, out.sum().backward() errors out?
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
            toks, weight.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16
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
        torch.testing.assert_close(out, out_alt)

        # Backwards
        # # For some reason, out.sum().backward() errors out?
        # out.sum().backward()
        # out_alt.sum().backward()
        grad = torch.randn_like(out)
        out.backward(grad)
        out_alt.backward(grad)

        torch.testing.assert_close(weight.grad, weight_copy.grad)
        torch.testing.assert_close(toks.grad, toks_copy.grad)

    def test_titan_gemm_equal(self) -> None:
        """
        torchtitan GEMM
        """
        from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
            ALIGN_SIZE_M,
            grouped_gemm_forward,
        )

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
        from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
            ALIGN_SIZE_M,
            grouped_gemm_forward,
        )

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


class TestMoeImpls(_TestBase):
    def test_bmm_equiv(self):
        """
        Testing the equivalence between different impls
        """
        gate = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func="softmax",
            route_scale=1.0,
            **self.factory_kwargs,
        )
        self.experts = nn.ModuleDict(
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
        x = self.get_inputs()

        x = x.view(-1, self.in_features)

        weights, indices = gate(x)

        # DeepSeek-v3 impl:
        z_ds = torch.zeros_like(x)
        for i in range(self.n_routed_experts):
            # TODO: @goon - handle no-tokens edge case
            expert = self.experts[str(i)]
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
            z_alt[idxs.flatten()] = self.experts[str(exp_idx)](x[idxs.any(dim=-1)])

        # Store the unsorted results back in x_by_expert
        z_alt = z_alt.reshape(*(weights.shape + z_alt.shape[-1:]))
        z_alt = torch.bmm(weights[:, None], z_alt).squeeze(1)

        torch.testing.assert_close(z_ds, z_alt, atol=self.tol, rtol=self.tol)

    def test_get_single_exp_output(self):
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
