import torch
import torch.nn.functional as F
from typing import Callable, Optional
from einops import rearrange
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from mamba_ssm.modules.mamba2 import Mamba2

from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.ssd_combined_cp import (
    mamba_chunk_scan_combined_non_cp,
    mamba_chunk_scan_combined_serial_cp,
)


try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None, None


class CausalPassingFn(torch.autograd.Function):
    """
    Causally pass tensors from one rank to the next, with the ordering defined by the mesh.
    The first rank receives zeros.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, mesh: dist.device_mesh.DeviceMesh
    ) -> torch.Tensor:
        if mesh.ndim != 1:
            raise ValueError("Only supports 1D DeviceMesh instances.")
        mesh_size = mesh.size()
        mesh_rank = mesh.get_local_rank()
        send_to: Optional[int] = mesh_rank + 1
        if send_to == mesh_size:
            send_to = None
        recv_from: Optional[int] = mesh_rank - 1
        if recv_from == -1:
            recv_from = None

        ctx.device = tensor.device
        ctx.dtype = tensor.dtype
        ctx.mesh = mesh
        ctx.recv_from = recv_from
        ctx.send_to = send_to
        ctx.shape = tensor.shape
        ops = []
        tensor = tensor.contiguous()  # Crucial for correctness
        if send_to is not None:
            ops.append(
                dist.P2POp(dist.isend, tensor, None, mesh.get_group(), 0, send_to)
            )
        if recv_from is not None:
            recv_buffer = torch.empty_like(tensor)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_buffer, None, mesh.get_group(), 0, recv_from
                )
            )
        else:
            recv_buffer = torch.zeros_like(tensor)
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return recv_buffer

    @staticmethod
    def backward(ctx, dtensor: torch.Tensor) -> tuple[Optional[torch.Tensor], None]:
        dtensor = dtensor.contiguous()  # Crucial for correctness
        ops = []
        if ctx.send_to is not None:
            recv_buffer = torch.empty(*ctx.shape, dtype=ctx.dtype, device=ctx.device)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_buffer, None, ctx.mesh.get_group(), 0, ctx.send_to
                )
            )
        else:
            recv_buffer = None
        if ctx.recv_from is not None:
            ops.append(
                dist.P2POp(
                    dist.isend, dtensor, None, ctx.mesh.get_group(), 0, ctx.recv_from
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return recv_buffer, None


causal_passing_comms = CausalPassingFn.apply


class IdentityFwdAllReduceBwdFn(torch.autograd.Function):
    """
    Wrapper for all-gathering grads onto unsharded tensors which are used in rank-sharded
    operations.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, mesh: dist.device_mesh.DeviceMesh
    ) -> torch.Tensor:
        if mesh.ndim != 1:
            raise ValueError("Only supports 1D DeviceMesh instances.")
        ctx.mesh = mesh
        return tensor

    @staticmethod
    def backward(ctx, dtensor: torch.Tensor) -> tuple[torch.Tensor, None]:
        dtensor = funcol.all_reduce(dtensor, reduceOp="sum", group=ctx.mesh.get_group())
        return dtensor, None


identity_fwd_all_reduce_bwd = IdentityFwdAllReduceBwdFn.apply


# Break down the Mamba2 forward into components


def conv(xBC, mamba2: Mamba2, conv_state=None, seq_idx=None) -> torch.Tensor:
    """
    Perform causal_conv1d_fn correcting for any previous conv_state, if any.
    """
    if seq_idx is not None:
        raise NotImplementedError
    out = causal_conv1d_fn(
        xBC.transpose(1, 2),
        rearrange(mamba2.conv1d.weight, "d 1 w -> d w"),
        bias=mamba2.conv1d.bias,
        activation=mamba2.activation,
        seq_idx=seq_idx,
    ).transpose(1, 2)
    if conv_state is not None:
        conv_state_seq_len = conv_state.shape[1]
        assert conv_state_seq_len == mamba2.d_conv - 1
        conv_state_inputs = torch.cat([conv_state, xBC[:, :conv_state_seq_len]], dim=1)
        cont_state_out = conv(conv_state_inputs, mamba2, None, seq_idx)[
            :, -conv_state_seq_len:
        ]
        out[:, :conv_state_seq_len] = cont_state_out
    return out


def conv_cp(
    xBC,
    mamba2: Mamba2,
    mesh: dist.device_mesh.DeviceMesh,
    seq_idx=None,
) -> torch.Tensor:
    conv_state_send = xBC[:, -(mamba2.d_conv - 1) :]
    conv_state_recv = causal_passing_comms(conv_state_send, mesh)
    return conv(xBC, mamba2, conv_state_recv, seq_idx)


def in_proj_split(inputs, mamba2: Mamba2):
    batch, seqlen, _ = inputs.shape
    zxbcdt = mamba2.in_proj(inputs)
    d_mlp = (
        zxbcdt.shape[-1]
        - 2 * mamba2.d_ssm
        - 2 * mamba2.ngroups * mamba2.d_state
        - mamba2.nheads
    ) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [
            d_mlp,
            d_mlp,
            mamba2.d_ssm,
            mamba2.d_ssm + 2 * mamba2.ngroups * mamba2.d_state,
            mamba2.nheads,
        ],
        dim=-1,
    )
    return z0, x0, z, xBC, dt


def scan(
    chunk_scan_combined_impl: Callable,
    xBC: torch.Tensor,
    dt: torch.Tensor,
    z: torch.Tensor,
    mamba2: Mamba2,
    seq_idx=None,
    mesh: Optional[dist.device_mesh.DeviceMesh] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, B, C = torch.split(
        xBC,
        [
            mamba2.d_ssm,
            mamba2.ngroups * mamba2.d_state,
            mamba2.ngroups * mamba2.d_state,
        ],
        dim=-1,
    )
    A = -torch.exp(mamba2.A_log.float())  # (nheads) or (d_inner, d_state)
    y, final_state = chunk_scan_combined_impl(
        rearrange(x, "b l (h p) -> b l h p", p=mamba2.headdim),
        dt,
        A,
        rearrange(B, "b l (g n) -> b l g n", g=mamba2.ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=mamba2.ngroups),
        chunk_size=mamba2.chunk_size,
        D=rearrange(mamba2.D, "(h p) -> h p", p=mamba2.headdim)
        if mamba2.D_has_hdim
        else mamba2.D,
        z=rearrange(z, "b l (h p) -> b l h p", p=mamba2.headdim)
        if not mamba2.rmsnorm
        else None,
        dt_bias=mamba2.dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        cu_seqlens=None,
        return_final_states=True,
        return_varlen_states=False,
        mesh=mesh,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    return y, final_state


class _Mamba2Ref(Mamba2):
    """
    Class for testing correctness of the forward rewrite.
    """

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None):
        if seqlen is not None:
            raise NotImplementedError
        if seq_idx is not None:
            raise NotImplementedError
        if cu_seqlens is not None:
            raise NotImplementedError
        z0, x0, z, xBC, dt = in_proj_split(u, self)

        xBC = conv(xBC, self, seq_idx)
        y, _ = scan(
            mamba_chunk_scan_combined_non_cp, xBC, dt, z, self, seq_idx, mesh=None
        )

        if self.rmsnorm:
            y = self.norm(y, z)

        d_nonssm = (
            sum(t.shape[-1] for t in (z0, x0, z, xBC, dt))
            - 2 * self.d_model * self.expand
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        assert d_nonssm >= 0
        if d_nonssm > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out


class Mamba2CP(Mamba2):
    def __init__(self, mesh: dist.device_mesh.DeviceMesh, *args, **kwargs) -> None:
        self.mesh = mesh
        super().__init__(*args, **kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None):
        if seqlen is not None:
            raise NotImplementedError
        if seq_idx is not None:
            raise NotImplementedError
        if cu_seqlens is not None:
            raise NotImplementedError
        z0, x0, z, xBC, dt = in_proj_split(u, self)

        xBC = conv_cp(xBC, self, self.mesh, seq_idx)
        y, _ = scan(
            mamba_chunk_scan_combined_serial_cp,
            xBC,
            dt,
            z,
            self,
            seq_idx,
            mesh=self.mesh,
        )

        if self.rmsnorm:
            y = self.norm(y, z)

        d_nonssm = (
            sum(t.shape[-1] for t in (z0, x0, z, xBC, dt))
            - 2 * self.d_model * self.expand
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        assert d_nonssm >= 0
        if d_nonssm > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out


class MHACP(MHA):
    def __init__(self, mesh: dist.device_mesh.DeviceMesh, *args, **kwargs) -> None:
        self.mesh = mesh
        super().__init__(*args, **kwargs)
        if self.d_conv:
            raise NotImplementedError

    def forward(self, x, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError

        seqlen_offset = 0
        rotary_max_seqlen = None
        qkv = self.in_proj(x)

        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)

        q, kv = qkv.split(
            [self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim],
            dim=-1,
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        if self.rotary_emb_dim > 0:
            q, kv = self.rotary_emb(
                q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )

        k, v = kv.unbind(dim=-3)
        k = torch.repeat_interleave(
            k, dim=2, repeats=self.num_heads // self.num_heads_kv
        )
        v = torch.repeat_interleave(
            v, dim=2, repeats=self.num_heads // self.num_heads_kv
        )

        context = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
            scale=self.softmax_scale,
        ).transpose(1, 2)

        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out
