import torch
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


class RingCommsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        group_size = group.size()
        group_rank = group.rank()
        send_to = (group_rank + 1) % group_size
        recv_from = (group_rank - 1) % group_size
        ctx.group = group
        ctx.send_to = send_to
        ctx.recv_from = recv_from
        ctx.shape = tensor.shape
        ctx.dtype = tensor.dtype
        ctx.device = tensor.device
        ops = []
        recv_buffer = torch.empty_like(tensor)
        ops.append(dist.P2POp(dist.isend, tensor, None, group, 0, send_to))
        ops.append(dist.P2POp(dist.irecv, recv_buffer, None, group, 0, recv_from))
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return recv_buffer

    @staticmethod
    def backward(ctx, dtensor):
        send_to = ctx.send_to
        recv_from = ctx.recv_from
        ops = []
        recv_buffer = torch.empty(*ctx.shape, dtype=ctx.dtype, device=ctx.device)
        ops.append(dist.P2POp(dist.irecv, recv_buffer, None, ctx.group, 0, send_to))
        ops.append(dist.P2POp(dist.isend, dtensor, None, ctx.group, 0, recv_from))
        for op in dist.batch_isend_irecv(ops):
            op.wait()
        return recv_buffer, None


def causal_ring_comms(tensor: torch.Tensor, group: dist.ProcessGroup):
    return RingCommsFn.apply(tensor, group)
