from copy import deepcopy

import torch
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import conv, _Mamba2Ref
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


# Breaking out various parts of the fwd for easier testing:


class _TestBase:
    batch_size = 2
    cp_dim = 4
    chunk_size = 4
    seq_len = 4 * cp_dim * chunk_size
    n_chunks = seq_len // chunk_size
    d_model = 256
    d_state = 128
    ngroups = 1
    expand = 2
    headdim = 64
    d_conv = 4
    d_inner = expand * d_model
    d_ssm = d_inner
    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"device": device, "dtype": dtype}
    model_kwargs = {
        "d_model": d_model,
        "d_state": d_state,
        "chunk_size": chunk_size,
        "ngroups": ngroups,
        "headdim": headdim,
        "d_conv": d_conv,
        "device": device,
        "dtype": dtype,
    }

    def get_mamba2(self, seed: int = 42) -> Mamba2:
        torch.manual_seed(seed)
        return Mamba2(**self.model_kwargs)

    def get_mamba2_ref(self, seed: int = 42) -> _Mamba2Ref:
        torch.manual_seed(seed)
        return _Mamba2Ref(**self.model_kwargs)

    def get_inputs(self, requires_grad: bool = False, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
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
            self.d_ssm + 2 * self.ngroups * self.d_state,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

    def get_states_dA_chunk_cumum(
        self, requires_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_heads = self.d_model * self.expand // self.headdim
        states = torch.randn(
            self.batch_size,
            self.n_chunks,
            n_heads,
            self.headdim,
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


class Test_Mamba2Ref(_TestBase):
    def test_fwd(self) -> None:
        mamba2 = self.get_mamba2()
        mamba2_ref = self.get_mamba2_ref()
        inputs = self.get_inputs()
        outputs = mamba2(inputs)
        outputs_copy = mamba2_ref(inputs)
        torch.testing.assert_close(outputs, outputs_copy)

    def test_bwd(self) -> None:
        mamba2 = self.get_mamba2()
        mamba2_copy = self.get_mamba2_ref()
        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        mamba2(inputs).sum().backward()
        mamba2_copy(inputs_copy).sum().backward()

        for p1, p2 in zip(mamba2.parameters(), mamba2_copy.parameters()):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad)
        torch.testing.assert_close(inputs.grad, inputs_copy.grad)


class TestLocalCP(_TestBase):
    """
    Single-device mock ups of CP implementations, and other related tests.
    """

    def test_conv_with_state_fwd(self) -> None:
        torch.manual_seed(42)
        xBC = self.get_xBC()

        # Shard and create the conv states
        xBC_cp = rearrange(xBC, "b (c l) d -> b l d c ", c=self.cp_dim)
        xBC_cp_conv_states = xBC_cp[:, -(self.d_conv - 1) :]
        xBC_cp_conv_states = xBC_cp_conv_states.roll(1, dims=-1)
        # First conv state is trivial (could also make it None)
        xBC_cp_conv_states[..., 0] = 0.0

        outputs = conv(xBC, self.get_mamba2())
        outputs_cp_list: list[torch.Tensor] = []
        for cp_rank in range(self.cp_dim):
            cp_out = conv(
                xBC_cp[..., cp_rank],
                mamba2=self.get_mamba2(),
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

        mamba2 = self.get_mamba2()
        mamba2_cp = deepcopy(mamba2)

        # Shard and create the conv states
        xBC_cp = rearrange(xBC_clone, "b (r l) d -> b l d r ", r=self.cp_dim)
        xBC_cp_conv_states = xBC_cp[:, -(self.d_conv - 1) :]

        outputs = conv(xBC, mamba2)
        outputs.sum().backward()

        for cp_rank in range(self.cp_dim):
            cp_out = conv(
                xBC_cp[..., cp_rank],
                mamba2=mamba2_cp,
                conv_state=None
                if not cp_rank
                else xBC_cp_conv_states[..., cp_rank - 1],
            )
            cp_out.sum().backward()
        for p1, p2 in zip(mamba2.parameters(), mamba2_cp.parameters()):
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
    def test_chunked_state_passing_allgather_fwd_alt(self) -> None:
        """
        Simulated CP with more parallelized state_passing step:
        - Every rank completes its state_passing with initial_sates = None.
        - final_state and dA_cumsum.sum(dim=-1) all-gathered across ranks
        - state_passing used on gathered final_state with dA_cumsum.sum(dim=-1) as the args
        - Each ranks adds a slice of the output from the prev step to its state to correct for the
          missing parts
        NOTE: omits the d_state dim
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

        # Ground truth
        out, final_state = _state_passing_fwd(states, dA_chunk_cumsum)

        # Compute the states local to each rank (trivial initial_states)
        final_state_cp_list = []
        for rank_cp in range(self.cp_dim):
            _, final_state_cp = _state_passing_fwd(
                states_cp[..., rank_cp],
                dA_chunk_cumsum_cp[..., rank_cp],
            )
            final_state_cp_list.append(final_state_cp)

        # Simulate all-gathering the final states summed dA_chunk_cumsum_cp per rank.
        final_states_allgather = torch.stack(final_state_cp_list, dim=1)
        # Important! Do the sum in float32, otherwise the tests won't pass due to numerics
        dA_chunk_sum_allgather = dA_chunk_cumsum_cp.to(torch.float32).sum(dim=2)

        # Locally compute state passing on the final states. These define what would logically be
        # the proper initial states that every rank should have started with.
        initial_states_cp, _ = _state_passing_fwd(
            final_states_allgather,
            dA_chunk_sum_allgather,
        )
        initial_states_cp = rearrange(initial_states_cp, "b r h p -> b h p r")

        # Then recompute the out states with the now-known initial states.
        out_cp_list: list[torch.Tensor] = []
        for rank_cp in range(self.cp_dim):
            out_cp, _ = _state_passing_fwd(
                states_cp[..., rank_cp],
                dA_chunk_cumsum_cp[..., rank_cp],
                initial_states_cp[..., rank_cp],
            )
            out_cp_list.append(out_cp)
        out_cp = torch.stack(out_cp_list, dim=-1)
        out_cp = rearrange(out_cp, "b c h p r -> b (r c) h p")

        # Again needs high tolerance to fully pass.
        tol = 1e-1
        torch.testing.assert_close(out, out_cp, atol=tol, rtol=tol)
