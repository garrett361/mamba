from copy import deepcopy
import torch
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined_split import mamba_chunk_scan_combined_split

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


class TestLocalChunking:
    batch_size = 2
    cp_dim = 4
    seq_len = 32
    d_model = 256
    factory_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    model = Mamba2(d_model, **factory_kwargs)

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

    def test(self) -> None:
        model = Mamba2(self.d_model, **self.factory_kwargs)
        inputs = self.get_inputs()
        outputs = model(inputs)
        outputs

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
        xBC_cp = rearrange(xBC_clone, "b (c l) d -> b l d c ", c=self.cp_dim)
        xBC_cp_conv_states = xBC_cp[:, -(self.model.d_conv - 1) :]
        xBC_cp_conv_states = xBC_cp_conv_states.roll(1, dims=-1)
        # First conv state is trivial (could also make it None)
        xBC_cp_conv_states[..., 0] = 0.0

        outputs = conv(xBC, self.model)
        outputs.sum().backward()

        for cp_rank in range(self.cp_dim):
            cp_out = conv(
                xBC_cp[..., cp_rank],
                model=model_cp,
                conv_state=xBC_cp_conv_states[..., cp_rank],
            )
            # Not sure why, but I am being required to retain the graph.
            cp_out.sum().backward(retain_graph=True)
        for p1, p2 in zip(self.model.parameters(), model_cp.parameters()):
            torch.testing.assert_close(p1, p2)
        tol = 5e-3
        torch.testing.assert_close(xBC.grad, xBC_clone.grad, atol=tol, rtol=tol)

    def test_chunked_state_passing_sequential_fwd(self) -> None:
        torch.manual_seed(42)
        states, dA_chunk_cumsum = self.get_states_dA_chunk_cumum()

        # Shard and create the conv states
        states_cp = rearrange(states, "b (r c) h p -> b c h p r", r=self.cp_dim)
        dA_chunk_cumsum_cp = rearrange(
            dA_chunk_cumsum, "b h (r c) -> b h c r ", r=self.cp_dim
        )

        out, _ = _state_passing_fwd(states, dA_chunk_cumsum)

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
