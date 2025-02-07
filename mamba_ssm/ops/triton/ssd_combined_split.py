import torch
import triton
import math
from einops import rearrange
from packaging import version

from mamba_ssm.ops.triton.ssd_combined import (
    _bmm_chunk_fwd,
    _bmm_chunk_bwd,
    _chunk_scan_bwd_dz,
    _chunk_scan_fwd,
    _chunk_scan_bwd_dstates,
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _state_passing_bwd,
    _chunk_scan_chunk_state_bwd_dx,
    _chunk_state_bwd_db,
    _chunk_scan_bwd_dC,
    _chunk_state_fwd,
    _chunk_scan_bwd_dcb,
    _state_passing_fwd,
    _chunk_scan_bwd_ddAcs_stable,
)

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def _mamba_chunk_scan_get_states_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None
        else None,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    ]
    return dt, dA_cumsum, states, final_states


def _mamba_chunk_scan_get_out_fwd(
    x,
    dt,
    B,
    C,
    states,
    dA_cumsum,
    chunk_size,
    D=None,
    z=None,
    seq_idx=None,
):
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(
        CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx
    )
    return out, out_x


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    dt, dA_cumsum, states, final_states = _mamba_chunk_scan_get_states_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        seq_idx,
        cu_seqlens,
        dt_softplus,
        dt_limit,
    )
    out, out_x = _mamba_chunk_scan_get_out_fwd(
        x,
        dt,
        B,
        C,
        states,
        dA_cumsum,
        chunk_size,
        D,
        z,
        seq_idx,
    )

    return out, out_x, dt, dA_cumsum, states, final_states


def _mamba_chunk_scan_combined_bwd(
    dout,
    x,
    dt,
    A,
    B,
    C,
    out,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dfinal_states=None,
    seq_idx=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    dx=None,
    ddt=None,
    dB=None,
    dC=None,
    dz=None,
    recompute_output=False,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    # TD: For some reason Triton (2.1.0 and 2.2.0) errors with
    # "[CUDA]: invalid device context" (e.g. during varlne test), and cloning makes it work. Idk why.
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt_in,
        A,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None
        else None,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(
            x,
            z,
            out,
            dout,
            chunk_size=chunk_size,
            has_ddAcs=False,
            D=D,
            dz=dz,
            recompute_output=recompute_output,
        )
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(
        C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype
    )
    # dstates has length nchunks, containing the gradient to initial states at index 0 and
    # gradient to the states of chunk (nchunks - 2) at index (nchunks - 1)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)")
        if dfinal_states is not None
        else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    # dstates has length nchunks, containing the gradient to states of chunk 0 at index 0 and
    # gradient to the final states at index (nchunks - 1)
    # states has length nchunks, containing the initial states at index 0 and the state for chunk (nchunks - 2) at index (nchunks - 1)
    # The final states is not stored.
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = (
        rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate)
        if dinitial_states is not None
        else None
    )
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
        x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx
    )
    # dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, ngroups=ngroups)
    dB, ddA_next = _chunk_state_bwd_db(
        x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups
    )
    # dC = _chunk_scan_bwd_dC(states[:, :-1].to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(
        states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups
    )
    # Computing ddA with the dcb kernel is much slower, so we're not using it for now
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    # dCB, ddA_tmp = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, CB=CB, ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    if z is None:
        dD = dD_from_x
    # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
    # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
    # However, this is numerically unstable: when we do the reverse cumsum on ddA_cumsum, there might
    # be a lot of underflow.

    # This is already done as part of bwd_dC kernel
    # ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states[:, :-1], C, dout, dA_cumsum, seq_idx=seq_idx)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    # This is already done as part of bwd_dB kernel
    # ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=seq_idx)
    # We don't need to pass in seq_idx because CB also zeros out entries where seq_idx[i] != seq_idx[j]
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(
        ddA,
        ddt,
        dt_in,
        A,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        ddt=ddt_given,
    )

    # These 2 lines are just to test ddt and dA being computed by old code
    # _, dA = selective_scan_bwd(dout, x, dt, A, B, C, D=D.float(), z=z)
    # ddt_given.copy_(ddt)

    return_vals = (
        dx,
        ddt_given,
        dA,
        dB_given,
        dC_given,
        dD,
        dz,
        ddt_bias,
        dinitial_states,
    )
    return return_vals if not recompute_output else (*return_vals, outz)


class MambaChunkScanCombinedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        return_varlen_states=False,
    ):
        ctx.dt_dtype = dt.dtype
        if not return_varlen_states:
            cu_seqlens = None
        else:
            assert (
                cu_seqlens is not None
            ), "cu_seqlens must be provided if return_varlen_states is True"
        out, out_x, dt_out, dA_cumsum, states, final_states, *rest = (
            _mamba_chunk_scan_combined_fwd(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                dt_softplus=dt_softplus,
                dt_limit=dt_limit,
            )
        )
        ctx.save_for_backward(
            out if z is None else out_x,
            x,
            dt,
            dA_cumsum,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            initial_states,
            seq_idx,
        )
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.return_varlen_states = return_varlen_states
        if not return_varlen_states:
            return out if not return_final_states else (out, final_states)
        else:
            varlen_states = rest[0]
            return (
                (out, varlen_states)
                if not return_final_states
                else (out, final_states, varlen_states)
            )

    @staticmethod
    def backward(ctx, dout, *args):
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = (
            ctx.saved_tensors
        )
        assert (
            not ctx.return_varlen_states
        ), "return_varlen_states is not supported in backward"
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (
            _mamba_chunk_scan_combined_bwd(
                dout,
                x,
                dt,
                A,
                B,
                C,
                out,
                ctx.chunk_size,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                dfinal_states=dfinal_states,
                seq_idx=seq_idx,
                dt_softplus=ctx.dt_softplus,
                dt_limit=ctx.dt_limit,
            )
        )
        return (
            dx,
            ddt,
            dA,
            dB,
            dC,
            None,
            dD,
            dz,
            ddt_bias,
            dinitial_states,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mamba_chunk_scan_combined_split(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False,
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return MambaChunkScanCombinedFn.apply(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        seq_idx,
        cu_seqlens,
        dt_softplus,
        dt_limit,
        return_final_states,
        return_varlen_states,
    )
