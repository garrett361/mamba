import torch
from typing import Optional
from einops import rearrange
import torch.distributed as dist

from mamba_ssm.modules.mamba2 import Mamba2

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None, None


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


def causal_passing_comms(tensor: torch.Tensor, group: dist.ProcessGroup):
    return CausalPassingFn.apply(tensor, group)
