"""
Utilities for creating context-parallel autograd functions from the triton kernel building blocks.

Every kernel is embarassingly parallel in the sequence dimension, except for
`_state_passing_{fwd,bwd}`. The `_mamba_chunk_scan_combined_{fwd,bwd}_template` functions below
mirror their `_mamba_chunk_scan_combined_{fwd,bwd}` counterparts, but take a `_StatePassingImpl`
implementation which defines the replacements for the original `_state_passing_{fwd,bwd}` kernels.

After defining the `_StatePassingImpl.{fwd,bwd}` methods of an implementation, the
`_StatePassingImpl.get_chunk_scan_autograd_fn` class method generates the corresponding
`autograd.Function` using the templated functions above.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import torch
import torch.distributed as dist
import triton
from einops import rearrange
from packaging import version
from torch.profiler import record_function

from mamba_ssm.ops.triton.ssd_combined import (
    _bmm_chunk_bwd,
    _bmm_chunk_fwd,
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_dz,
    _chunk_scan_chunk_state_bwd_dx,
    _chunk_scan_fwd,
    _chunk_state_bwd_db,
    _chunk_state_fwd,
    _state_passing_bwd,
    _state_passing_fwd,
    chunk_state_varlen,
)

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


class _StatePassingImpl(ABC):
    @staticmethod
    @abstractmethod
    def fwd(
        chunk_size,
        states,  # (batch, nchunks, nheads, headdim, d_state)
        dA_chunk_cumsum,  # (batch, nheads, nchunks)
        initial_states=None,  # Optional[(batch, nheads, headdim, d_state)]
        seq_idx=None,
        out_dtype=None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Returns a tuple of (out_states, final_states, bwd_args), where bwd_args is anything that
        _StatePassingImpl.bwd might require for its backwards pass.

        Shapes:
        - out_states: (batch, nchunks, nheads, headdim, d_state)
        - final_states: (batch, nheads, headdim, d_state)
        """
        ...

    @staticmethod
    @abstractmethod
    def bwd(
        chunk_size: int,
        states: torch.Tensor,  # (batch, nchunks, nheads, headdim, d_state)
        dA_chunk_cumsum: torch.Tensor,  # (batch, nheads, nchunks)
        dstates: torch.Tensor,  # (batch, nchunks, nheads, headdim, d_state)
        dstates_dtype: torch.dtype,
        states_dtype: torch.dtype,
        initial_states: Optional[
            torch.Tensor
        ] = None,  # Optional[(batch, nheads, headdim, d_state)]
        dfinal_states: Optional[
            torch.Tensor
        ] = None,  # Optional[(batch, nheads, headdim, d_state)]
        seq_idx: Optional[torch.Tensor] = None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
        bwd_args: Optional[Any] = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Returns a tuple of (dstates, ddA_chunk_cumsum, dinitial_states, states). bwd_args is
        whatever may have been passed along by _StatePassingImpl.fwd.
        Shapes:
        - dstates_out: (batch, nchunks, nheads, headdim, d_state)
        - ddA_chunk_cumsum: (batch, nheads, nchunks, n_blocks),  n_blocks set by triton config
        - dinitial_states: Optional[(batch, nchunks, nheads, headdim, d_state)]
        - states:  Optional[(batch, nchunks, nheads, headdim, d_state)]
        """
        ...

    @classmethod
    def get_chunk_scan_autograd_fn(cls):
        class _Fn(torch.autograd.Function):
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
                cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
            ):
                ctx.dt_dtype = dt.dtype
                if not return_varlen_states:
                    cu_seqlens = None
                else:
                    assert cu_seqlens is not None, (
                        "cu_seqlens must be provided if return_varlen_states is True"
                    )
                out, out_x, _, _, _, final_states, *rest = (
                    _mamba_chunk_scan_combined_fwd_template(
                        cls,
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
                        cp_mesh=cp_mesh,
                    )
                )
                ctx.save_for_backward(
                    out if z is None else out_x,
                    x,
                    dt,
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
                ctx.cp_mesh = cp_mesh
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
                (
                    out,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z,
                    dt_bias,
                    initial_states,
                    seq_idx,
                ) = ctx.saved_tensors
                assert not ctx.return_varlen_states, (
                    "return_varlen_states is not supported in backward"
                )
                dfinal_states = args[0] if ctx.return_final_states else None
                dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (
                    _mamba_chunk_scan_combined_bwd_template(
                        cls,
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
                        cp_mesh=ctx.cp_mesh,
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
                    None,
                )

        _Fn.__name__ = f"{cls.__name__}Fn"

        def fn(
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
            cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
        ):
            """
            Argument:
                x: (batch, seqlen, nheads, headdim)
                dt: (batch, seqlen, nheads)
                A: (nheads)
                B: (batch, seqlen, ngroups, d_state)
                C: (batch, seqlen, ngroups, d_state)
                chunk_size: int
                D: (nheads, headdim) or (nheads,)
                z: (batch, seqlen, nheads, headdim)
                dt_bias: (nheads,)
                initial_states: (batch, nheads, headdim, d_state)
                seq_idx: (batch, seqlen)
                cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
                dt_softplus: Whether to apply softplus to dt
                cp_mesh: Optional[DeviceMesh]
            Return:
                out: (batch, seqlen, nheads, headdim)
            """
            return _Fn.apply(
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
                cp_mesh,
            )

        return fn


# Templates for fwd and bwd passes constructed like _mamba_chunk_scan_combined_{fwd,bwd}, but
# with customizable state passing impls.


def _mamba_chunk_scan_combined_fwd_template(
    state_passing_impl: Type[_StatePassingImpl],
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
    cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
):
    with record_function("pre_state_passing_impl_fwd"):
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
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        states = _chunk_state_fwd(
            B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True
        )

    with record_function("state_passing_impl_fwd"):
        states, final_states, _ = state_passing_impl.fwd(
            chunk_size=chunk_size,
            states=states,
            dA_chunk_cumsum=dA_cumsum[:, :, :, -1],
            initial_states=initial_states,
            seq_idx=seq_idx,
            out_dtype=C.dtype,
            cp_mesh=cp_mesh,
        )

    with record_function("post_state_passing_impl_fwd"):
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32
        )
        out, out_x = _chunk_scan_fwd(
            CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx
        )
        if cu_seqlens is None:
            return out, out_x, dt, dA_cumsum, states, final_states
        else:
            assert batch == 1, (
                "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
            )
            varlen_states = chunk_state_varlen(
                B.squeeze(0),
                x.squeeze(0),
                dt.squeeze(0),
                dA_cumsum.squeeze(0),
                cu_seqlens,
                states.squeeze(0),
            )
            return out, out_x, dt, dA_cumsum, states, final_states, varlen_states


def _mamba_chunk_scan_combined_bwd_template(
    state_passing_impl: Type[_StatePassingImpl],
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
    cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
):
    with record_function("pre_state_passing_impl_fwd"):
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        batch, seqlen, nheads, headdim = x.shape
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
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32
        )
        states = _chunk_state_fwd(
            B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True
        )

    with record_function("state_passing_impl_fwd"):
        states, _, bwd_args = state_passing_impl.fwd(
            chunk_size=chunk_size,
            states=states,
            dA_chunk_cumsum=dA_cumsum[:, :, :, -1],
            initial_states=initial_states,
            seq_idx=seq_idx,
            out_dtype=None,
            cp_mesh=cp_mesh,
        )

    with record_function("pre_state_passing_impl_bwd"):
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

    with record_function("state_passing_impl_bwd"):
        dstates, ddA_chunk_cumsum, dinitial_states, states = state_passing_impl.bwd(
            chunk_size=chunk_size,
            states=states,
            dA_chunk_cumsum=dA_cumsum[:, :, :, -1],
            dstates=dstates,
            dstates_dtype=x.dtype,
            states_dtype=x.dtype,
            initial_states=initial_states,
            dfinal_states=dfinal_states,
            seq_idx=seq_idx,
            cp_mesh=cp_mesh,
            bwd_args=bwd_args,
        )

    with record_function("post_state_passing_impl_bwd"):
        dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
            x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx
        )
        dB, ddA_next = _chunk_state_bwd_db(
            x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups
        )
        dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(
            states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups
        )
        dCB = _chunk_scan_bwd_dcb(
            x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups
        )
        dCB = dCB.to(CB.dtype)
        _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
        _bmm_chunk_bwd(
            B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given
        )
        # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
        # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
        if z is None:
            dD = dD_from_x
        ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
        ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
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


##### Non-CP impls. For testing and components of other impls.  #####


class StatePassingNonCP(_StatePassingImpl):
    @staticmethod
    def fwd(
        chunk_size,
        states,
        dA_chunk_cumsum,
        initial_states=None,
        seq_idx=None,
        out_dtype=None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,  # Intentionally unused
    ):
        d_state = states.shape[-1]
        out_states, final_states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"),
            dA_chunk_cumsum,
            initial_states=rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None,
            seq_idx=seq_idx,
            chunk_size=chunk_size,
            out_dtype=out_dtype,
        )
        out_states, final_states = [
            rearrange(t, "... (p n) -> ... p n", n=d_state)
            for t in [out_states, final_states]
        ]
        return out_states, final_states, None

    @staticmethod
    def bwd(
        chunk_size: int,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        dstates: torch.Tensor,
        dstates_dtype: torch.dtype,
        states_dtype: torch.dtype,
        initial_states: Optional[torch.Tensor] = None,
        dfinal_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
        bwd_args: Optional[Any] = None,  # Intentionally unused
    ):
        d_state = states.shape[-1]
        dstates_out, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
            rearrange(states, "... p n -> ... (p n)"),
            dA_chunk_cumsum,
            rearrange(dstates, "... p n -> ... (p n)"),
            dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)")
            if dfinal_states is not None
            else None,
            seq_idx=seq_idx,
            has_initial_states=initial_states is not None,
            dstates_dtype=dstates_dtype,
            states_dtype=states_dtype,
            chunk_size=chunk_size,
        )
        states = rearrange(states, "... (p n) -> ... p n", n=d_state)
        dstates_out = rearrange(dstates_out, "... (p n) -> ... p n", n=d_state)
        if dinitial_states is not None:
            dinitial_states = rearrange(
                dinitial_states, "... (p n) -> ... p n", n=d_state
            )
        return dstates_out, ddA_chunk_cumsum, dinitial_states, states


mamba_chunk_scan_combined_non_cp = StatePassingNonCP.get_chunk_scan_autograd_fn()


#### Start CP Impls ####


# Async send/recv wrappers.
# NOTE: @goon - For some reason, plain dist.isend(...).wait() calls are erroring out for me. Don't
# think is should be necessary to use the `batch_isend_irecv` wrapper.
def send(tensor, dst=None, group=None) -> None:
    for op in dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, dst, group)]):
        op.wait()


def recv(tensor, src=None, group=None) -> None:
    for op in dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, src, group)]):
        op.wait()


class StatePassingSerialCP(_StatePassingImpl):
    """
    TODO: @goon - seq_idx probably not being treated correctly
    """

    @staticmethod
    def fwd(
        chunk_size,
        states,
        dA_chunk_cumsum,
        initial_states=None,
        seq_idx=None,
        out_dtype=None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,  # Unused
    ):
        """
        Serially compute the final state on each rank and pass as initial_states to the next.
        """
        assert cp_mesh is not None
        local_rank = cp_mesh.get_local_rank()
        group = cp_mesh.get_group()
        is_lead_rank = local_rank == 0
        if not is_lead_rank and initial_states is not None:
            raise ValueError(
                "initial_states can only be non-trival on the lead CP rank."
            )
        recv_init_states = None
        mesh_size = cp_mesh.size()
        for send_rank, recv_rank in zip(range(mesh_size - 1), range(1, mesh_size)):
            if local_rank == send_rank:
                out_states, final_states, _ = StatePassingNonCP.fwd(
                    chunk_size=chunk_size,
                    states=states,
                    dA_chunk_cumsum=dA_chunk_cumsum,
                    initial_states=recv_init_states,
                    seq_idx=seq_idx,
                    out_dtype=out_dtype,
                )

                send(
                    final_states.contiguous(),
                    dst=dist.get_global_rank(group, recv_rank),
                    group=group,
                )
            elif local_rank == recv_rank:
                recv_init_states = torch.empty_like(states[:, 0])
                recv(
                    recv_init_states,
                    src=dist.get_global_rank(group, send_rank),
                    group=group,
                )

        # Final rank only:
        if local_rank == recv_rank:
            out_states, final_states, _ = StatePassingNonCP.fwd(
                chunk_size=chunk_size,
                states=states,
                dA_chunk_cumsum=dA_chunk_cumsum,
                initial_states=recv_init_states,
                seq_idx=seq_idx,
                out_dtype=out_dtype,
            )
        bwd_args = recv_init_states
        return out_states, final_states, bwd_args

    @staticmethod
    def bwd(
        chunk_size: int,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        dstates: torch.Tensor,
        dstates_dtype: torch.dtype,
        states_dtype: torch.dtype,
        initial_states: Optional[torch.Tensor] = None,
        dfinal_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
        bwd_args: Optional[Any] = None,
    ):
        """
        Starting from the final rank to compute in _state_passing_serial_cp_fwd, compute
        dinitial_states and pass these back as the dfinal_states of the preceding rank.
        """
        assert cp_mesh is not None
        local_rank = cp_mesh.get_local_rank()
        mesh_size = cp_mesh.size()
        group = cp_mesh.get_group()
        is_lead_rank = local_rank == 0
        if not is_lead_rank and initial_states is not None:
            raise ValueError(
                "initial_states can only be non-trival on the lead CP rank."
            )
        is_last_rank = local_rank == mesh_size - 1
        if not is_last_rank and dfinal_states is not None:
            raise ValueError(
                "dfinal_states can only be non-trival on the last CP rank."
            )
        recv_init_states = bwd_args
        recv_dfinal_states = None
        for send_rank, recv_rank in zip(
            range(mesh_size - 1, 0, -1), range(mesh_size - 2, -1, -1)
        ):
            if local_rank == send_rank:
                dstates_out, ddA_chunk_cumsum, send_dinitial_states, states = (
                    StatePassingNonCP.bwd(
                        chunk_size=chunk_size,
                        states=states,
                        dA_chunk_cumsum=dA_chunk_cumsum,
                        dstates=dstates,
                        initial_states=recv_init_states,
                        dfinal_states=recv_dfinal_states,
                        seq_idx=seq_idx,
                        dstates_dtype=dstates_dtype,
                        states_dtype=states_dtype,
                        cp_mesh=cp_mesh,
                    )
                )
                send(
                    send_dinitial_states.contiguous(),
                    dst=dist.get_global_rank(group, recv_rank),
                    group=group,
                )
            elif local_rank == recv_rank:
                recv_dfinal_states = torch.empty(
                    *states[:, 0].shape, dtype=dstates_dtype, device=states.device
                )
                recv(
                    recv_dfinal_states,
                    src=dist.get_global_rank(group, send_rank),
                    group=group,
                )

        if local_rank == recv_rank:
            # First rank only:
            dstates_out, ddA_chunk_cumsum, dinitial_states, states = (
                StatePassingNonCP.bwd(
                    chunk_size=chunk_size,
                    states=states,
                    dA_chunk_cumsum=dA_chunk_cumsum,
                    dstates=dstates,
                    initial_states=initial_states,  # Gets the original initial_states, if any
                    dfinal_states=recv_dfinal_states,
                    seq_idx=seq_idx,
                    dstates_dtype=dstates_dtype,
                    states_dtype=states_dtype,
                    cp_mesh=cp_mesh,
                )
            )
        else:
            # Only the first rank potentially had non-trivial initial_states as proper inputs, so
            # all other ranks get dinitial_state = None.
            dinitial_states = None
        return dstates_out, ddA_chunk_cumsum, dinitial_states, states


mamba_chunk_scan_combined_serial_cp = StatePassingSerialCP.get_chunk_scan_autograd_fn()


class StatePassingAllGatherCP(_StatePassingImpl):
    """
    TODO: @goon - seq_idx probably not being treated correctly
    """

    @staticmethod
    def fwd(
        chunk_size,
        states,
        dA_chunk_cumsum,
        initial_states=None,
        seq_idx=None,
        out_dtype=None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,  # Unused
    ):
        """
        1. Every rank completes its _state_passing_fwd with initial_sates = None.
        2. final_state and dA_cumsum.sum(dim=-1) all-gathered across ranks
        3. The above data can be passed through _state_passing_fwd again to get the corrected
           initial_states that each rank should have started with.
        4. Every rank computes its _state_passing_fwd again, now with its proper initial_states
        """
        assert cp_mesh is not None
        local_rank = cp_mesh.get_local_rank()
        is_lead_rank = local_rank == 0
        if not is_lead_rank and initial_states is not None:
            raise ValueError(
                "initial_states can only be non-trival on the lead CP rank."
            )

        # Start communicating info for later
        dA_chunk_sum = dA_chunk_cumsum.sum(dim=-1)
        dA_chunk_sum_allgather = torch.empty(
            cp_mesh.size(),
            *dA_chunk_sum.shape,
            device=dA_chunk_sum.device,
            dtype=dA_chunk_sum.dtype,
        )
        dA_comms_handle = dist.all_gather_into_tensor(
            dA_chunk_sum_allgather,
            dA_chunk_sum.contiguous(),
            group=cp_mesh.get_group(),
            async_op=True,
        )

        # Compute the partial states with the locally available information. I.e. use trivial
        # initial_states on all but, maybe, the lead rank.
        states_partial, final_states_partial, _ = StatePassingNonCP.fwd(
            chunk_size=chunk_size,
            states=states,
            dA_chunk_cumsum=dA_chunk_cumsum,
            initial_states=initial_states,
            seq_idx=seq_idx,
            out_dtype=out_dtype,
        )

        final_states_partial_allgather = torch.empty(
            cp_mesh.size(),
            *final_states_partial.shape,
            device=final_states_partial.device,
            dtype=final_states_partial.dtype,
        )
        dist.all_gather_into_tensor(
            final_states_partial_allgather,
            final_states_partial.contiguous(),
            group=cp_mesh.get_group(),
            async_op=False,
        )

        final_states_partial_allgather = rearrange(
            final_states_partial_allgather, "r b ... -> b r ... "
        )
        dA_comms_handle.wait()
        dA_chunk_sum_allgather = rearrange(
            dA_chunk_sum_allgather, "r ... -> ... r"
        ).contiguous()

        # TODO: @goon - write a more focused kernel for this step.
        if is_lead_rank:
            initial_states_corrected = initial_states
            final_states = final_states_partial
            out_states = states_partial
        else:
            # Build the initial_states that each rank should have started with.
            states_slice = final_states_partial_allgather[
                :, : cp_mesh.get_local_rank()
            ].contiguous()
            dA_chunk_cumsum_slice = dA_chunk_sum_allgather[
                ..., : cp_mesh.get_local_rank()
            ].contiguous()
            _, initial_states_corrected, _ = StatePassingNonCP.fwd(
                chunk_size=cp_mesh.size(),
                states=states_slice,
                dA_chunk_cumsum=dA_chunk_cumsum_slice,
                initial_states=None,
                seq_idx=seq_idx,
                out_dtype=out_dtype,
            )

            # And repeat the forward with the now-corrected initial_states
            out_states, final_states, _ = StatePassingNonCP.fwd(
                chunk_size=chunk_size,
                states=states,
                dA_chunk_cumsum=dA_chunk_cumsum,
                initial_states=initial_states_corrected,
                seq_idx=seq_idx,
                out_dtype=out_dtype,
            )

        bwd_args = (initial_states_corrected, dA_chunk_sum_allgather)

        return out_states, final_states, bwd_args

    @staticmethod
    def bwd(
        chunk_size: int,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        dstates: torch.Tensor,
        dstates_dtype: torch.dtype,
        states_dtype: torch.dtype,
        initial_states: Optional[torch.Tensor] = None,
        dfinal_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
        bwd_args: Optional[Any] = None,
    ):
        """
        1. Every rank completes its _state_passing_bwd with dfinal_states = None (except maybe on
            the final cp rank) and with the corrected initial_states from the forward.
        2. Allgather dfinal_states and use these to compute what the proper dfinal_states should
        have been on each rank
        3. Every rank recomputes its _state_passing_bwd with the proper {dfinal,initial}_states
        values.
        """
        assert cp_mesh is not None
        local_rank = cp_mesh.get_local_rank()
        is_lead_rank = local_rank == 0
        if not is_lead_rank and initial_states is not None:
            raise ValueError(
                "initial_states can only be non-trival on the lead CP rank."
            )
        is_last_rank = local_rank == cp_mesh.size() - 1
        if not is_last_rank and dfinal_states is not None:
            raise ValueError(
                "dfinal_states can only be non-trival on the last CP rank."
            )

        assert bwd_args is not None
        initial_states_corrected, dA_chunk_sum_allgather = bwd_args

        # Compute dinitial_states with the locally available information. I.e. use trivial
        # dfinal_states on all but, maybe, the last rank. These can be used to compute the
        # corrected dfinal_states each rank should have started with.
        dstates_partial, ddA_chunk_cumsum_partial, dinitial_states_partial, _ = (
            StatePassingNonCP.bwd(
                chunk_size=chunk_size,
                states=states,
                dA_chunk_cumsum=dA_chunk_cumsum,
                dstates=dstates,
                # HACK: initial_states=True ensures dinitial_states_partial is never None, maybe
                # just zeros
                initial_states=True,
                dfinal_states=dfinal_states,
                seq_idx=seq_idx,
                dstates_dtype=dstates_dtype,
                states_dtype=states_dtype,
                cp_mesh=cp_mesh,
            )
        )

        dinitial_states_partial_allgather = torch.empty(
            cp_mesh.size(),
            *dinitial_states_partial.shape,
            device=dinitial_states_partial.device,
            dtype=dinitial_states_partial.dtype,
        )
        dist.all_gather_into_tensor(
            dinitial_states_partial_allgather,
            dinitial_states_partial.contiguous(),
            group=cp_mesh.get_group(),
            async_op=False,
        )
        dinitial_states_partial_allgather = rearrange(
            dinitial_states_partial_allgather, "r b ... -> b r ... "
        )

        # TODO: @goon - write a more focused kernel for this step.
        if is_last_rank:
            dinitial_states = None
            dstates_out = dstates_partial
            ddA_chunk_cumsum = ddA_chunk_cumsum_partial
        else:
            # Build the dfinal_states that each rank should have started with.
            dstates_slice = dinitial_states_partial_allgather[
                :, cp_mesh.get_local_rank() + 1 :
            ]
            dA_chunk_cumsum_slice = dA_chunk_sum_allgather[
                ..., cp_mesh.get_local_rank() + 1 :
            ]
            _, _, dfinal_states_corrected, _ = StatePassingNonCP.bwd(
                chunk_size=chunk_size,
                # `states` is not used, but can't be None and it needs to pass shape checks
                states=dstates_slice,
                dA_chunk_cumsum=dA_chunk_cumsum_slice,
                dstates=dstates_slice,
                # HACK: initial_states=True ensures dfinal_states_corrected is never None, maybe
                # just zeros
                initial_states=True,
                dfinal_states=dfinal_states,
                seq_idx=seq_idx,
                dstates_dtype=dstates_dtype,
                states_dtype=states_dtype,
                cp_mesh=cp_mesh,
            )

            # And repeat the backward with the now-corrected dfinal_states
            dstates_out, ddA_chunk_cumsum, dinitial_states, states = (
                StatePassingNonCP.bwd(
                    chunk_size=chunk_size,
                    states=states,
                    dA_chunk_cumsum=dA_chunk_cumsum,
                    dstates=dstates,
                    initial_states=initial_states_corrected,
                    dfinal_states=dfinal_states_corrected,
                    seq_idx=seq_idx,
                    dstates_dtype=dstates_dtype,
                    states_dtype=states_dtype,
                    cp_mesh=cp_mesh,
                )
            )

            # Only the first rank potentially had non-trivial initial_states as proper inputs, so
            # all other ranks get dinitial_states = None.
            if not is_lead_rank:
                dinitial_states = None
        return dstates_out, ddA_chunk_cumsum, dinitial_states, states


mamba_chunk_scan_combined_allgather_cp = (
    StatePassingAllGatherCP.get_chunk_scan_autograd_fn()
)
