from typing import Callable, Literal, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.ssd_combined_cp import (
    mamba_chunk_scan_combined_allgather_cp,
    mamba_chunk_scan_combined_allgather_cp_recompute,
    mamba_chunk_scan_combined_non_cp,
    mamba_chunk_scan_combined_non_cp_recompute,
    mamba_chunk_scan_combined_serial_cp,
    mamba_chunk_scan_combined_serial_cp_recompute,
)

CP_MAMBA_IMPLS = {
    "serial": mamba_chunk_scan_combined_serial_cp,
    "allgather": mamba_chunk_scan_combined_allgather_cp,
}
CP_MAMBA_RECOMPUTE_IMPLS = {
    "serial": mamba_chunk_scan_combined_serial_cp_recompute,
    "allgather": mamba_chunk_scan_combined_allgather_cp_recompute,
}


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
        local_rank = mesh.get_local_rank()
        group = mesh.get_group()
        send_to: Optional[int] = local_rank + 1
        if send_to == mesh_size:
            send_to = None
        recv_from: Optional[int] = local_rank - 1
        if recv_from == -1:
            recv_from = None

        ctx.group = group
        ctx.recv_from = recv_from
        ctx.send_to = send_to
        ops = []
        tensor = tensor.contiguous()  # Crucial for correctness
        if send_to is not None:
            ops.append(
                dist.P2POp(
                    dist.isend, tensor, dist.get_global_rank(group, send_to), group
                )
            )
        if recv_from is not None:
            recv_buffer = torch.empty_like(tensor)
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buffer,
                    dist.get_global_rank(group, recv_from),
                    group,
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
            recv_buffer = torch.empty_like(dtensor)
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buffer,
                    dist.get_global_rank(ctx.group, ctx.send_to),
                    ctx.group,
                )
            )
        else:
            recv_buffer = None
        if ctx.recv_from is not None:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    dtensor,
                    dist.get_global_rank(ctx.group, ctx.recv_from),
                    ctx.group,
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return recv_buffer, None


causal_passing_comms = CausalPassingFn.apply


def _get_seq_to_zigzag_minishard_maps(
    group_size: int,
) -> tuple[dict[int, int], dict[int, int]]:
    minishard_list = list(range(2 * group_size))
    minishard_list[::2], minishard_list[1::2] = (
        minishard_list[:group_size],
        minishard_list[: group_size - 1 : -1],
    )
    seq_to_zigzag_map = {seq: zigzag for seq, zigzag in enumerate(minishard_list)}
    zigzag_to_seq_map = {zigzag: seq for seq, zigzag in seq_to_zigzag_map.items()}
    return seq_to_zigzag_map, zigzag_to_seq_map


class SeqToZigZagFn(torch.autograd.Function):
    """
    Convert from sequentially sharded tensors to zig-zag sharded tensors, as in ring-flash-attn.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, mesh: dist.device_mesh.DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        sharded_seq_len = tensor.shape[seq_dim]
        assert sharded_seq_len % 2 == 0
        if mesh.ndim != 1:
            raise ValueError("Only supports 1D DeviceMesh instances.")
        mesh_size = mesh.size()
        assert mesh_size % 2 == 0
        group = mesh.get_group()
        mesh_rank = mesh.get_local_rank()
        seq_to_zigzag_map, zigzag_to_seq_map = _get_seq_to_zigzag_minishard_maps(
            mesh_size
        )

        minishard_seq_idxs = (2 * mesh_rank, 2 * mesh_rank + 1)
        send_to_idxs = tuple(zigzag_to_seq_map[s] for s in minishard_seq_idxs)
        recv_from_idxs = tuple(seq_to_zigzag_map[s] for s in minishard_seq_idxs)

        ctx.group = group
        ctx.send_to_idxs = send_to_idxs
        ctx.recv_from_idxs = recv_from_idxs
        ctx.seq_dim = seq_dim

        mini_shard_0, mini_shard_1 = tensor.tensor_split(2, dim=seq_dim)
        # contiguous is crucial for correctness
        send_buffers = (mini_shard_0.contiguous(), mini_shard_1.contiguous())
        recv_buffers = (torch.empty_like(mini_shard_0), torch.empty_like(mini_shard_1))

        ops = []
        for send_buf, send_idx in zip(send_buffers, send_to_idxs):
            # Use send_idx to tag the comms. Convert to rank via rank = idx // 2
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_buf,
                    dist.get_global_rank(group, send_idx // 2),
                    group,
                    send_idx,
                )
            )
        for recv_buf, recv_idx, send_idx in zip(
            recv_buffers, recv_from_idxs, send_to_idxs
        ):
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buf,
                    dist.get_global_rank(group, recv_idx // 2),
                    group,
                    send_idx,
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return torch.cat(recv_buffers, dim=1).contiguous()

    @staticmethod
    def backward(ctx, dtensor: torch.Tensor) -> tuple[Optional[torch.Tensor], None]:
        mini_shard_0, mini_shard_1 = dtensor.tensor_split(2, dim=ctx.seq_dim)
        # contiguous is crucial for correctness
        send_buffers = (mini_shard_0.contiguous(), mini_shard_1.contiguous())
        recv_buffers = (torch.empty_like(mini_shard_0), torch.empty_like(mini_shard_1))

        ops = []
        for send_buf, send_idx in zip(send_buffers, ctx.recv_from_idxs):
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_buf,
                    dist.get_global_rank(ctx.group, send_idx // 2),
                    ctx.group,
                    send_idx,
                )
            )
        for recv_buf, recv_idx, send_idx in zip(
            recv_buffers, ctx.send_to_idxs, ctx.recv_from_idxs
        ):
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buf,
                    dist.get_global_rank(ctx.group, recv_idx // 2),
                    ctx.group,
                    send_idx,
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()

        return torch.cat(recv_buffers, dim=1).contiguous(), None, None


seq_to_zigzag_comms = SeqToZigZagFn.apply


class ZigZagToSeqFn(torch.autograd.Function):
    """
    Convert from zig-zag sharded tensors to sequentually sharded tensors, as in ring-flash-attn.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, mesh: dist.device_mesh.DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        sharded_seq_len = tensor.shape[seq_dim]
        assert sharded_seq_len % 2 == 0
        if mesh.ndim != 1:
            raise ValueError("Only supports 1D DeviceMesh instances.")
        mesh_size = mesh.size()
        assert mesh_size % 2 == 0
        group = mesh.get_group()
        mesh_rank = mesh.get_local_rank()
        seq_to_zigzag_map, zigzag_to_seq_map = _get_seq_to_zigzag_minishard_maps(
            mesh_size
        )

        minishard_seq_idxs = (2 * mesh_rank, 2 * mesh_rank + 1)
        send_to_idxs = tuple(seq_to_zigzag_map[s] for s in minishard_seq_idxs)
        recv_from_idxs = tuple(zigzag_to_seq_map[s] for s in minishard_seq_idxs)

        ctx.group = group
        ctx.send_to_idxs = send_to_idxs
        ctx.recv_from_idxs = recv_from_idxs
        ctx.seq_dim = seq_dim

        mini_shard_0, mini_shard_1 = tensor.tensor_split(2, dim=seq_dim)
        # contiguous is crucial for correctness
        send_buffers = (mini_shard_0.contiguous(), mini_shard_1.contiguous())
        recv_buffers = (torch.empty_like(mini_shard_0), torch.empty_like(mini_shard_1))

        ops = []
        for send_buf, send_idx in zip(send_buffers, send_to_idxs):
            # Use send_idx to tag the comms. Convert to rank via rank = idx // 2
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_buf,
                    dist.get_global_rank(group, send_idx // 2),
                    group,
                    send_idx,
                )
            )
        for recv_buf, recv_idx, send_idx in zip(
            recv_buffers, recv_from_idxs, send_to_idxs
        ):
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buf,
                    dist.get_global_rank(group, recv_idx // 2),
                    group,
                    send_idx,
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return torch.cat(recv_buffers, dim=1).contiguous()

    @staticmethod
    def backward(ctx, dtensor: torch.Tensor) -> tuple[Optional[torch.Tensor], None]:
        mini_shard_0, mini_shard_1 = dtensor.tensor_split(2, dim=ctx.seq_dim)
        # contiguous is crucial for correctness
        send_buffers = (mini_shard_0.contiguous(), mini_shard_1.contiguous())
        recv_buffers = (torch.empty_like(mini_shard_0), torch.empty_like(mini_shard_1))

        ops = []
        for send_buf, send_idx in zip(send_buffers, ctx.recv_from_idxs):
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_buf,
                    dist.get_global_rank(ctx.group, send_idx // 2),
                    ctx.group,
                    send_idx,
                )
            )
        for recv_buf, recv_idx, send_idx in zip(
            recv_buffers, ctx.send_to_idxs, ctx.recv_from_idxs
        ):
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_buf,
                    dist.get_global_rank(ctx.group, recv_idx // 2),
                    ctx.group,
                    send_idx,
                )
            )
        for op in dist.batch_isend_irecv(ops):
            op.wait()

        return torch.cat(recv_buffers, dim=1).contiguous(), None, None


zigzag_to_seq_comms = ZigZagToSeqFn.apply


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


_identity_fwd_all_reduce_bwd = IdentityFwdAllReduceBwdFn.apply


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
        conv_state_out = conv(conv_state_inputs, mamba2, None, seq_idx)[
            :, -conv_state_seq_len:
        ]
        out[:, :conv_state_seq_len] = conv_state_out
    return out


def conv_cp(
    xBC,
    mamba2: Mamba2,
    cp_mesh: dist.device_mesh.DeviceMesh,
    seq_idx=None,
) -> torch.Tensor:
    with record_function("conv_cp"):
        conv_state_send = xBC[:, -(mamba2.d_conv - 1) :]
        # TODO: @goon - make the conv_state send/recv async. Can overlap with the first conv.
        conv_state_recv = causal_passing_comms(conv_state_send, cp_mesh)
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
    cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None,
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
    y = chunk_scan_combined_impl(
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
        return_final_states=False,
        return_varlen_states=False,
        cp_mesh=cp_mesh,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    return y


class _Mamba2Ref(Mamba2):
    cp_mamba_recompute: bool = False
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
        y = scan(
            mamba_chunk_scan_combined_non_cp_recompute
            if self.cp_mamba_recompute
            else mamba_chunk_scan_combined_non_cp,
            xBC,
            dt,
            z,
            self,
            seq_idx,
            cp_mesh=None,
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
    """
    NOTE: @goon - currently we expect an external mechanism to all-reduce the grads which get
    populated on Mamba2CP instances. This is handled automatically if the model is wrapped in
    FSDP, but not otherwise. Need to raise a warning or offer some flag for enabling the all-reduce
    in other cases.
    """

    def __init__(
        self,
        *args,
        cp_mesh: dist.device_mesh.DeviceMesh,
        cp_mamba_impl: Literal["serial", "allgather"] = "allgather",
        cp_mamba_recompute: bool = False,
        **kwargs,
    ) -> None:
        self.cp_mesh = cp_mesh
        self.cp_mamba_impl = cp_mamba_impl
        if cp_mamba_recompute:
            self.cp_impl_fn = CP_MAMBA_RECOMPUTE_IMPLS[self.cp_mamba_impl]
        else:
            self.cp_impl_fn = CP_MAMBA_IMPLS[self.cp_mamba_impl]
        super().__init__(*args, **kwargs)

    def forward(
        self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None
    ):
        if seqlen is not None:
            raise NotImplementedError
        if seq_idx is not None:
            raise NotImplementedError
        if cu_seqlens is not None:
            raise NotImplementedError
        if inference_params is not None:
            raise NotImplementedError
        z0, x0, z, xBC, dt = in_proj_split(u, self)

        xBC = conv_cp(xBC, self, self.cp_mesh, seq_idx)
        y = scan(
            self.cp_impl_fn,
            xBC,
            dt,
            z,
            self,
            seq_idx,
            cp_mesh=self.cp_mesh,
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
    """
    NOTE: @goon - currently we expect an external mechanism to all-reduce the grads which get
    populated on MHACP instances. This is handled automatically if the model is wrapped in
    FSDP, but not otherwise. Need to raise a warning or offer some flag for enabling the all-reduce
    in other cases.
    """

    def __init__(
        self,
        *args,
        cp_mesh: dist.device_mesh.DeviceMesh,
        cp_attn_impl: Literal["zigzag", "ring"] = "zigzag",
        seq_dim: int = 1,
        **kwargs,
    ) -> None:
        if cp_mesh.ndim != 1:
            raise ValueError("Only supports 1D DeviceMesh instances.")
        self.cp_mesh = cp_mesh
        self.cp_attn_impl = cp_attn_impl
        self.seq_dim = seq_dim
        if cp_attn_impl == "ring":
            from ring_flash_attn import ring_flash_attn_func

            self.ring_flash_attn_impl = ring_flash_attn_func
        elif cp_attn_impl == "zigzag":
            from ring_flash_attn import zigzag_ring_flash_attn_func

            self.ring_flash_attn_impl = zigzag_ring_flash_attn_func
        else:
            raise ValueError(f"Unexpected {cp_attn_impl=}")

        super().__init__(*args, **kwargs)
        if self.d_conv:
            raise NotImplementedError

    def forward(self, x, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError

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
            # Important: need to offset the seqlens per the cp rank and input size.
            # Assumption: all ranks are getting inputs of the same seqlen.
            local_cp_rank = self.cp_mesh.get_local_rank()
            num_tok_per_rank = x.shape[self.seq_dim]
            seqlen_offset = local_cp_rank * num_tok_per_rank
            q, kv = self.rotary_emb(
                q,
                kv,
                seqlen_offset=seqlen_offset,
                max_seqlen=num_tok_per_rank * self.cp_mesh.size(),
            )

        if self.cp_attn_impl == "zigzag":
            # Important: change to zigzag sharding *after* applying RoPE.
            q = seq_to_zigzag_comms(q, self.cp_mesh, self.seq_dim)
            kv = seq_to_zigzag_comms(kv, self.cp_mesh, self.seq_dim)

        k, v = kv.unbind(dim=-3)
        context = self.ring_flash_attn_impl(
            q,
            k,
            v,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            group=self.cp_mesh.get_group(),
        )

        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        if self.cp_attn_impl == "zigzag":
            out = zigzag_to_seq_comms(out, self.cp_mesh, self.seq_dim)
        return out
