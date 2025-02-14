from copy import deepcopy
import torch
import torch.distributed as dist
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from test_fake_cp import in_proj_split

from dtest import DTest
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import RingCommsFn
from mamba_ssm.ops.triton.ssd_combined_cp import (
    mamba_chunk_scan_combined_serial_cp,
)


class TestCausalRingCommsFn(DTest):
    def test_fwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.rank], group)
        self.print_rank(f"{t_recv=}")

        torch.testing.assert_close(t_recv, t_send[(self.rank - 1) % self.world_size])

    def test_bwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.rank], group)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad[self.rank]
        torch.testing.assert_close(grad, t_send[self.rank])

        other_idxs = torch.arange(self.world_size, device=self.device) != self.rank
        zero_grads = t_send.grad[other_idxs]
        torch.testing.assert_close(zero_grads, torch.zeros_like(zero_grads))


class TestSerialCP(DTest):
    batch_size = 2
    chunk_size = 4
    d_model = 256
    d_state = 128

    @property
    def seq_len(self) -> int:
        return 4 * self.world_size * self.chunk_size

    @property
    def n_chunks(self) -> int:
        return self.seq_len // self.chunk_size

    @property
    def factory_kwargs(self):
        return {"dtype": torch.bfloat16, "device": self.device}

    def get_model(self, seed: int = 42):
        torch.manual_seed(seed)
        return Mamba2(
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            **self.factory_kwargs,
        )

    def get_inputs(self, requires_grad: bool = False, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_model,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

    def _get_non_seq_dim_kwargs(
        self,
        model,
    ):
        """
        Common scan kwargs which don't have a seq_len dim to shard.
        """
        A = -torch.exp(model.A_log.float())  # (nheads) or (d_inner, d_state)

        non_seq_dim_kwargs = dict(
            A=A,
            chunk_size=model.chunk_size,
            D=rearrange(model.D, "(h p) -> h p", p=model.headdim)
            if model.D_has_hdim
            else model.D,
            dt_bias=model.dt_bias,
            dt_softplus=True,
        )
        return non_seq_dim_kwargs

    def get_scan_kwargs(self, inputs, model):
        """
        Get all kwargs for the non-cp scan.
        """
        _, _, z, xBC, dt = in_proj_split(inputs, model)
        x, B, C = torch.split(
            xBC,
            [model.d_ssm, model.ngroups * model.d_state, model.ngroups * model.d_state],
            dim=-1,
        )

        seq_dim_kwargs = dict(
            x=rearrange(x, "b l (h p) -> b l h p", p=model.headdim),
            dt=dt,
            B=rearrange(B, "b l (g n) -> b l g n", g=model.ngroups),
            C=rearrange(C, "b l (g n) -> b l g n", g=model.ngroups),
            z=rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
            if not model.rmsnorm
            else None,
        )
        non_seq_dim_kwargs = self._get_non_seq_dim_kwargs(model)
        return {**seq_dim_kwargs, **non_seq_dim_kwargs}

    def get_cp_scan_kwargs(self, inputs, model):
        """
        Get all kwargs for the cp scan.
        """
        _, _, z, xBC, dt = in_proj_split(inputs, model)
        x, B, C = torch.split(
            xBC,
            [model.d_ssm, model.ngroups * model.d_state, model.ngroups * model.d_state],
            dim=-1,
        )

        seq_dim_kwargs = dict(
            x=rearrange(x, "b l (h p) -> b l h p", p=model.headdim),
            dt=dt,
            B=rearrange(B, "b l (g n) -> b l g n", g=model.ngroups),
            C=rearrange(C, "b l (g n) -> b l g n", g=model.ngroups),
            z=rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
            if not model.rmsnorm
            else None,
        )
        cp_seq_dim_kwargs = {
            k: rearrange(t, "b (r l) ... -> b r l ...", r=self.world_size)[
                :, self.rank
            ].contiguous()
            if t is not None
            else None
            for k, t in seq_dim_kwargs.items()
        }

        non_seq_dim_kwargs = self._get_non_seq_dim_kwargs(model)
        return {**cp_seq_dim_kwargs, **non_seq_dim_kwargs}

    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        inputs = self.get_inputs()
        model = self.get_model()

        # The correct global outputs:
        kwargs = self.get_scan_kwargs(inputs, model)
        outputs = mamba_chunk_scan_combined(**kwargs)

        # And the CP output:
        cp_kwargs = self.get_cp_scan_kwargs(inputs, model)
        outputs_cp = mamba_chunk_scan_combined_serial_cp(mesh=mesh, **cp_kwargs)

        # All-gather and verify correctness
        outputs_cp_all_gathered = torch.empty(
            self.world_size,
            *outputs_cp.shape,
            dtype=outputs_cp.dtype,
            device=outputs_cp.device,
        )
        dist.all_gather_into_tensor(
            outputs_cp_all_gathered, outputs_cp, mesh.get_group()
        )
        outputs_cp_all_gathered = rearrange(
            outputs_cp_all_gathered, "r b l ... -> b (r l) ..."
        )
        torch.testing.assert_close(outputs, outputs_cp_all_gathered)

    def test_bwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        inputs = self.get_inputs(requires_grad=True)
        model = self.get_model()

        inputs_cp = self.get_inputs(requires_grad=True)
        model_cp = deepcopy(model)

        # Backward on the correct global result
        kwargs = self.get_scan_kwargs(inputs, model)
        mamba_chunk_scan_combined(**kwargs).sum().backward()

        # And on the CP output:
        cp_kwargs = self.get_cp_scan_kwargs(inputs_cp, model_cp)
        mamba_chunk_scan_combined_serial_cp(mesh=mesh, **cp_kwargs).sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = rearrange(
            inputs.grad, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]
        inputs_cp_grad_shard = rearrange(
            inputs_cp.grad, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]

        tol = 1e-2
        try:
            torch.testing.assert_close(
                inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
            )
        except Exception as e:
            self.print_rank(f"Caught Exception: {e}")
            raise e
