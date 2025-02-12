from copy import deepcopy

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import (
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _state_passing_fwd,
)
from mamba_ssm.ops.triton.ssd_combined_cp import mamba_chunk_scan_combined_split
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


def in_proj_split(inputs, model: Mamba2, seq_idx=None):
    batch, seqlen, dim = inputs.shape
    zxbcdt = model.in_proj(inputs)
    d_mlp = (
        zxbcdt.shape[-1]
        - 2 * model.d_ssm
        - 2 * model.ngroups * model.d_state
        - model.nheads
    ) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [
            d_mlp,
            d_mlp,
            model.d_ssm,
            model.d_ssm + 2 * model.ngroups * model.d_state,
            model.nheads,
        ],
        dim=-1,
    )
    return z0, x0, z, xBC, dt


def conv(xBC, model: Mamba2, conv_state=None, seq_idx=None) -> torch.Tensor:
    """
    Perform causal_conv1d_fn correcting for any previous conv_state, if any.
    """
    assert seq_idx is None, "seq_idx not currently supported"
    out = causal_conv1d_fn(
        xBC.transpose(1, 2),
        rearrange(model.conv1d.weight, "d 1 w -> d w"),
        bias=model.conv1d.bias,
        activation=model.activation,
        seq_idx=seq_idx,
    ).transpose(1, 2)
    if conv_state is not None:
        conv_state_seq_len = conv_state.shape[1]
        assert conv_state_seq_len == model.d_conv - 1
        conv_state_inputs = torch.cat([conv_state, xBC[:, :conv_state_seq_len]], dim=1)
        cont_state_out = conv(conv_state_inputs, model, None, seq_idx)[
            :, -conv_state_seq_len:
        ]
        out[:, :conv_state_seq_len] = cont_state_out
    return out


def scan(
    xBC: torch.Tensor,
    dt: torch.Tensor,
    z: torch.Tensor,
    model: Mamba2,
    seq_idx=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, B, C = torch.split(
        xBC,
        [model.d_ssm, model.ngroups * model.d_state, model.ngroups * model.d_state],
        dim=-1,
    )
    A = -torch.exp(model.A_log.float())  # (nheads) or (d_inner, d_state)
    y, final_state = mamba_chunk_scan_combined_split(
        rearrange(x, "b l (h p) -> b l h p", p=model.headdim),
        dt,
        A,
        rearrange(B, "b l (g n) -> b l g n", g=model.ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=model.ngroups),
        chunk_size=model.chunk_size,
        D=rearrange(model.D, "(h p) -> h p", p=model.headdim)
        if model.D_has_hdim
        else model.D,
        z=rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
        if not model.rmsnorm
        else None,
        dt_bias=model.dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        cu_seqlens=None,
        return_final_states=True,
        return_varlen_states=False,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    return y, final_state


def fwd(
    inputs: torch.Tensor, model: Mamba2, conv_state=None, seq_idx=None
) -> torch.Tensor:
    z0, x0, z, xBC, dt = in_proj_split(inputs, model)

    xBC = conv(xBC, model)
    y, final_state = scan(xBC, dt, z, model)

    if model.rmsnorm:
        y = model.norm(y, z)

    d_nonssm = (
        sum(t.shape[-1] for t in (z0, x0, z, xBC, dt))
        - 2 * model.d_model * model.expand
        - 2 * model.ngroups * model.d_state
        - model.nheads
    ) // 2
    assert d_nonssm >= 0
    if d_nonssm > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    out = model.out_proj(y)
    return out


class ChunkCumsumFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dt,
        A,
        chunk_size,
        dt_bias=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
    ):
        dA_cumsum, dt = _chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        ctx.save_for_backward(
            dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx
        )
        return dA_cumsum, dt

    @staticmethod
    def backward(ctx, dout, dfinal_states):
        _chunk_cumsum_bwd


class TestLocalCP:
    """
    Single-device mock ups of CP, and other related tests.
    """

    batch_size = 2
    cp_dim = 4
    chunk_size = 4
    seq_len = 4 * cp_dim * chunk_size
    n_chunks = seq_len // chunk_size
    d_model = 256
    d_state = 128
    factory_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    model = Mamba2(
        d_model=d_model, d_state=d_state, chunk_size=chunk_size, **factory_kwargs
    )

    def get_inputs(self, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_model,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

    def get_xBC(self, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.model.d_ssm + 2 * self.model.ngroups * self.model.d_state,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

    def get_states_dA_chunk_cumum(
        self, requires_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_heads = self.model.d_model * self.model.expand // self.model.headdim
        states = torch.randn(
            self.batch_size,
            self.n_chunks,
            n_heads,
            self.model.headdim,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )
        dA_chunk_cumsum = torch.randn(
            self.batch_size,
            n_heads,
            self.n_chunks,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )
        return states, dA_chunk_cumsum

    def test_fwd(self) -> None:
        model = Mamba2(self.d_model, **self.factory_kwargs)
        inputs = self.get_inputs()
        outputs = model(inputs)
        outputs_fwd = fwd(inputs, model)
        torch.testing.assert_close(outputs, outputs_fwd)

    def test_bwd(self) -> None:
        model_copy = deepcopy(self.model)
        torch.manual_seed(42)
        inputs = self.get_inputs(requires_grad=True)
        torch.manual_seed(42)
        inputs_copy = self.get_inputs(requires_grad=True)
        self.model(inputs).sum().backward()

        fwd(inputs_copy, model_copy).sum().backward()
        for p1, p2 in zip(self.model.parameters(), model_copy.parameters()):
            torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(inputs.grad, inputs_copy.grad)

    def test_conv(self) -> None:
        xBC = self.get_xBC()
        outputs = conv(xBC, self.model)
        outputs

    def test_conv_with_state_fwd(self) -> None:
        torch.manual_seed(42)
        xBC = self.get_xBC()

        # Shard and create the conv states
        xBC_cp = rearrange(xBC, "b (c l) d -> b l d c ", c=self.cp_dim)
        xBC_cp_conv_states = xBC_cp[:, -(self.model.d_conv - 1) :]
        xBC_cp_conv_states = xBC_cp_conv_states.roll(1, dims=-1)
        # First conv state is trivial (could also make it None)
        xBC_cp_conv_states[..., 0] = 0.0

        outputs = conv(xBC, self.model)
        outputs_cp_list: list[torch.Tensor] = []
        for cp_rank in range(self.cp_dim):
            cp_out = conv(
                xBC_cp[..., cp_rank],
                model=self.model,
                conv_state=xBC_cp_conv_states[..., cp_rank],
            )
            outputs_cp_list.append(cp_out)
        outputs_cp = torch.cat(outputs_cp_list, dim=1)
        torch.testing.assert_close(outputs, outputs_cp)

    def test_conv_with_state_bwd(self) -> None:
        torch.manual_seed(42)
        xBC = self.get_xBC(requires_grad=True)
        torch.manual_seed(42)
        xBC_clone = self.get_xBC(requires_grad=True)

        model_cp = deepcopy(self.model)

        # Shard and create the conv states
        xBC_cp = rearrange(xBC_clone, "b (r l) d -> b l d r ", r=self.cp_dim)
        xBC_cp_conv_states = xBC_cp[:, -(self.model.d_conv - 1) :]

        outputs = conv(xBC, self.model)
        outputs.sum().backward()

        for cp_rank in range(self.cp_dim):
            cp_out = conv(
                xBC_cp[..., cp_rank],
                model=model_cp,
                conv_state=None
                if not cp_rank
                else xBC_cp_conv_states[..., cp_rank - 1],
            )
            cp_out.sum().backward()
        for p1, p2 in zip(self.model.parameters(), model_cp.parameters()):
            torch.testing.assert_close(p1, p2)
        tol = 5e-3
        torch.testing.assert_close(xBC.grad, xBC_clone.grad, atol=tol, rtol=tol)

    def test_chunked_state_passing_sequential_fwd(self) -> None:
        """
        Simulated CP with serial state_passing step:
        - Rank 0 completes its state_passing with initial_sates = None, then passes final_state to
          Rank 1.
        - Rank 1 completes its state_passing using the received final_state as its initial_state,
          and passes the new final_state to Rank 2
        - ...
        - The last rank completes its state_passing using the received final_state as its
          initial_state.
        """
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()

        # Shard and create the conv states
        states_cp = rearrange(states, "b (r c) h p -> b c h p r", r=self.cp_dim)
        dA_chunk_cumsum_cp = rearrange(
            dA_chunk_cumsum, "b h (r c) -> b h c r ", r=self.cp_dim
        )

        out, final_state = _state_passing_fwd(states, dA_chunk_cumsum)

        out_cp_list = []
        initial_states = None
        for cp_rank in range(self.cp_dim):
            cp_out, initial_states = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
                initial_states,
            )
            out_cp_list.append(cp_out)
        out_cp = torch.cat(out_cp_list, dim=1)
        torch.testing.assert_close(out, out_cp)
        torch.testing.assert_close(final_state, initial_states)

    def test_chunked_final_state_all_gather(self) -> None:
        """
        Test that we are computing the final states correctly in the allgather strategy.
        """
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()
        # states: (b, c, h, p)
        # dA_chunk_cumsum: (b, h, c)

        states_cp = rearrange(states, "b (r c) h p -> b c h p r", r=self.cp_dim)
        dA_chunk_cumsum_cp = rearrange(
            dA_chunk_cumsum, "b h (r c) -> b h c r ", r=self.cp_dim
        )

        # The proper computation with final state passing
        actual_final_state_list = []
        final_state = None
        for cp_rank in range(self.cp_dim):
            _, final_state = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
                final_state,
            )
            actual_final_state_list.append(final_state)

        # Partial computation using only states local to each rank (trivial initial_states)
        out_partial_cp_list = []
        final_partial_cp_list = []
        for cp_rank in range(self.cp_dim):
            cp_out_partial, final_state_cp = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
            )
            out_partial_cp_list.append(cp_out_partial)
            final_partial_cp_list.append(final_state_cp)

        # Simulate all-gathering the final states summed dA_chunk_cumsum_cp per rank.
        final_states_cp = torch.stack(final_partial_cp_list, dim=1)
        # Important! Do the sum in float32, otherwise the tests won't pass due to numerics
        dA_chunk_cumsum_sum_cp = dA_chunk_cumsum_cp.to(torch.float32).sum(dim=2)

        # State passing on the final states.
        recomputed_final_states_cp, final_overall_state = _state_passing_fwd(
            final_states_cp,
            dA_chunk_cumsum_sum_cp,
        )  # (b, r, h, p)

        # Match up indices
        recomputed_final_state_list = [
            recomputed_final_states_cp[:, r] for r in range(1, self.cp_dim)
        ] + [final_overall_state]

        for rank in range(self.cp_dim):
            fs1 = actual_final_state_list[rank]
            fs2 = recomputed_final_state_list[rank]
            tol = 1e-4
            torch.testing.assert_close(fs1, fs2, rtol=tol, atol=tol)

    def test_chunked_out_state_allgather(self) -> None:
        """
        Test building the proper out-state with the allgather strategy.
        """
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()
        # states: (b, c, h, p)
        # dA_chunk_cumsum: (b, h, c)

        states_cp = rearrange(states, "b (r c) h p -> b c h p r", r=self.cp_dim)
        dA_chunk_cumsum_cp = rearrange(
            dA_chunk_cumsum, "b h (r c) -> b h c r ", r=self.cp_dim
        )

        # The proper computation with final state passing
        actual_initial_state_list = []
        actual_out_state_list = []
        actual_final_state_list = []
        initial_state = None
        for cp_rank in range(self.cp_dim):
            out_state, final_state = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
                initial_state,
            )
            if initial_state is not None:
                # Sanity check
                torch.testing.assert_close(initial_state, out_state[:, 0])
            actual_initial_state_list.append(initial_state)
            actual_out_state_list.append(out_state)
            actual_final_state_list.append(final_state)
            # Need to dtypes to match, otherwise the out_state[: 0] and initial_state only agree to
            # ~1e-2 level
            initial_state = final_state.to(out_state)

        # Partial computation using only states local to each rank (trivial initial_states)
        out_partial_cp_list = []
        for cp_rank in range(self.cp_dim):
            cp_out_partial, _ = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
            )
            out_partial_cp_list.append(cp_out_partial)

        # Then use local info, communicated final_states, and the partial results to build the
        # correct out_state
        for cp_rank in range(self.cp_dim):
            out_partial = out_partial_cp_list[cp_rank]
            if cp_rank == 0:
                torch.testing.assert_close(out_partial, actual_out_state_list[cp_rank])
            else:
                initial_states = actual_final_state_list[cp_rank - 1]  # b, h, p

                # Locally, we need to cumsum over the available dA_chunk_cumsum chunks
                # and offset the indexing by one.
                dA_chunk_cumsum_local = rearrange(
                    dA_chunk_cumsum_cp[..., cp_rank], "b h c -> b c h"
                ).to(torch.float32)
                dA_chunk_cumsum_local = dA_chunk_cumsum_local.roll(1, dims=1)
                dA_chunk_cumsum_local[:, 0] = 0.0
                dA_chunk_cumsum_local_processed = (
                    dA_chunk_cumsum_local.cumsum(dim=1).exp().to(out_partial)
                )
                corrected_out = (
                    out_partial
                    + initial_states[:, None].to(out_partial)
                    * dA_chunk_cumsum_local_processed[..., None]
                )  # b, c, h, p

                expected_out = actual_out_state_list[cp_rank]
                # Passes, but only with very lenient tolerances.
                tol = 1e-1
                torch.testing.assert_close(
                    corrected_out, expected_out, rtol=tol, atol=tol
                )

    def test_chunked_state_passing_allgather_fwd(self) -> None:
        """
        Simulated CP with somewhat more parallelized state_passing step:
        - Every rank completes its state_passing with initial_sates = None.
        - final_state and dA_cumsum.sum(dim=-1) all-gathered across ranks
        - state_passing used on gathered final_state with dA_cumsum.sum(dim=-1) as the args
        - Each ranks adds a slice of the output from the prev step to its state to correct for the
          missing parts
        """
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()
        # states: (b, c, h, p)
        # dA_chunk_cumsum: (b, h, c)

        # Shard and create the conv states
        states_cp = rearrange(states, "b (r c) h p -> b c h p r", r=self.cp_dim)
        dA_chunk_cumsum_cp = rearrange(
            dA_chunk_cumsum, "b h (r c) -> b h c r ", r=self.cp_dim
        )

        out, final_state = _state_passing_fwd(states, dA_chunk_cumsum)

        # Compute the states local to each rank (trivial initial_states)
        out_partial_cp_list = []
        final_state_cp_list = []
        for cp_rank in range(self.cp_dim):
            cp_out_partial, final_state_cp = _state_passing_fwd(
                states_cp[..., cp_rank],
                dA_chunk_cumsum_cp[..., cp_rank],
            )
            out_partial_cp_list.append(cp_out_partial)
            final_state_cp_list.append(final_state_cp)

        # Simulate all-gathering the final states summed dA_chunk_cumsum_cp per rank.
        out_partial_cp = torch.stack(out_partial_cp_list, dim=-1)
        final_states_cp = torch.stack(final_state_cp_list, dim=1)
        # Important! Do the sum in float32, otherwise the tests won't pass due to numerics
        dA_chunk_cumsum_sum_cp = dA_chunk_cumsum_cp.to(torch.float32).sum(dim=2)

        # Locally compute state passing on the final states, which defines the proper initial states
        # for each rank
        initial_states_cp, _ = _state_passing_fwd(
            final_states_cp,
            dA_chunk_cumsum_sum_cp,
        )
        initial_states_cp = rearrange(initial_states_cp, "b r h p -> b h p r")

        # Build the correction by multiplying by the exponentiated cumsum over chunks of A.
        # Use float32 again for better numerics
        dA_chunk_cumsum2_cp = dA_chunk_cumsum_cp.to(torch.float32)  # (b, h, c, r)
        # And requires index shifting and zeroing
        dA_chunk_cumsum2_cp = dA_chunk_cumsum2_cp.roll(1, dims=2)
        dA_chunk_cumsum2_cp[:, :, 0] = 0.0
        dA_chunk_cumsum2_cp = dA_chunk_cumsum2_cp.cumsum(dim=2).exp()

        state_corrections_cp = torch.einsum(
            "bhpr,bhcr->bchpr",
            initial_states_cp,
            dA_chunk_cumsum2_cp,
        )
        state_corrections = rearrange(state_corrections_cp, "b c h p r -> b (r c) h p")
        out_partial = rearrange(out_partial_cp, "b c h p r -> b (r c) h p")
        # high-precision computations
        out_partial_corrected = out_partial.to(state_corrections) + state_corrections
        out_partial_corrected = out_partial_corrected.to(out)
        # Again needs high tolerance to fully pass.
        tol = 1e-1
        torch.testing.assert_close(out, out_partial_corrected, atol=tol, rtol=tol)

    def test_state_passing_init_states(self) -> None:
        """
        Verifying that initial states become the first chunk of _state_passing_fwd's out states
        """
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()

        initial_states = torch.randn_like(states[:, 0])
        out, final_state = _state_passing_fwd(states, dA_chunk_cumsum, initial_states)
        torch.testing.assert_close(initial_states, out[:, 0])
