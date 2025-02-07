import torch
from einops import rearrange

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


from mamba_ssm.modules.mamba2 import Mamba2


def in_proj_split(inputs, model: Mamba2, seq_idx=None):
    batch, seqlen, dim = inputs.shape
    zxbcdt = model.in_proj(inputs)
    A = -torch.exp(model.A_log.float())  # (nheads) or (d_inner, d_state)
    dt_limit_kwargs = (
        {} if model.dt_limit == (0.0, float("inf")) else dict(dt_limit=model.dt_limit)
    )
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


class TestLocalChunking:
    batch_size = 2
    cp_dim = 4
    seq_len = 32
    d_model = 256
    factory_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    model = Mamba2(d_model, **factory_kwargs)

    def get_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size, self.seq_len, self.d_model, **self.factory_kwargs
        )

    def get_xBC(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.model.d_ssm + 2 * self.model.ngroups * self.model.d_state,
            **self.factory_kwargs,
        )

    def test(self) -> None:
        model = Mamba2(self.d_model, **self.factory_kwargs)
        inputs = self.get_inputs()
        outputs = model(inputs)
        outputs

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
