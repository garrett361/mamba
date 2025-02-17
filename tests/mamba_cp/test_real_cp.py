from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from test_fake_cp import in_proj_split

from dtest import DTest
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import (
    causal_passing_comms,
    conv,
    conv_cp,
    identity_fwd_all_reduce_bwd,
)
from mamba_ssm.ops.triton.ssd_combined_cp import (
    mamba_chunk_scan_combined_serial_cp,
)


class TestCausalPassingFn(DTest):
    def test_fwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = causal_passing_comms(t_send[self.rank], mesh)
        if self.rank == 0:
            torch.testing.assert_close(t_recv, torch.zeros_like(t_recv))
        else:
            torch.testing.assert_close(t_recv, t_send[(self.rank - 1)])

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
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = causal_passing_comms(t_send[self.rank], mesh)
        t_recv.pow(2).div(2).sum().backward()

        if self.rank == self.world_size - 1:
            assert t_send.grad is None
        else:
            grad = t_send.grad[self.rank]
            torch.testing.assert_close(grad, t_send[self.rank])
            other_idxs = torch.arange(self.world_size, device=self.device) != self.rank
            zero_grads = t_send.grad[other_idxs]
            torch.testing.assert_close(zero_grads, torch.zeros_like(zero_grads))


class TestIdentityFwdAllGatherBwdFn(DTest):
    def test_fwd(self):
        dim = 16
        torch.manual_seed(42)
        weight = torch.randn(
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        weight_copy = deepcopy(weight)

        weight = nn.Parameter(weight)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        weight_copy = nn.Parameter(weight_copy)

        inputs = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        inputs_shard = inputs.tensor_split(self.world_size, 0)[self.rank]

        output = inputs * weight
        output_shard = inputs_shard * identity_fwd_all_reduce_bwd(weight_copy, mesh)

        all_gathered_output_shards = torch.empty_like(inputs)
        dist.all_gather_into_tensor(
            all_gathered_output_shards, output_shard, group=mesh.get_group()
        )
        out = [torch.empty_like(output_shard) for _ in range(self.world_size)]
        dist.all_gather(out, output_shard, group=mesh.get_group())

        torch.testing.assert_close(all_gathered_output_shards, output)

    def test_bwd(self):
        dim = 16
        torch.manual_seed(42)
        weight = torch.randn(
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        weight_copy = deepcopy(weight)

        weight = nn.Parameter(weight)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        weight_copy = nn.Parameter(weight_copy)

        inputs = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        inputs_shard = inputs.tensor_split(self.world_size, 0)[self.rank]

        (inputs * weight).sum().backward()
        (inputs_shard * identity_fwd_all_reduce_bwd(weight_copy, mesh)).sum().backward()

        torch.testing.assert_close(weight.grad, weight_copy.grad)


class _DTestModelBase(DTest):
    batch_size = 2
    chunk_size = 4
    d_model = 256
    d_state = 128
    ngroups = 1
    expand = 2
    d_conv = 4
    d_inner = expand * d_model
    d_ssm = d_inner

    @property
    def seq_len(self) -> int:
        return 4 * self.world_size * self.chunk_size

    @property
    def n_chunks(self) -> int:
        return self.seq_len // self.chunk_size

    @property
    def factory_kwargs(self):
        return {"dtype": torch.float32, "device": self.device}

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


class TestConvCP(_DTestModelBase):
    def get_xBC(self, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_ssm + 2 * self.ngroups * self.d_state,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

    def test_fwd(self):
        torch.manual_seed(42)
        model = self.get_model()
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        xBC = self.get_xBC()

        xBC_cp = rearrange(xBC, "b (c l) d -> c b l d ", c=self.world_size)[self.rank]

        outputs = conv(xBC, model)
        outputs_cp = conv_cp(xBC_cp, model, mesh)
        torch.testing.assert_close(
            outputs_cp, outputs.tensor_split(self.world_size, dim=1)[self.rank]
        )


class TestSerialCP(_DTestModelBase):
    def get_scan_kwargs(self, inputs, model):
        """
        Get all kwargs for the scan.
        """
        A = -torch.exp(model.A_log.float())  # (nheads) or (d_inner, d_state)
        _, _, z, xBC, dt = in_proj_split(inputs, model)
        x, B, C = torch.split(
            xBC,
            [model.d_ssm, model.ngroups * model.d_state, model.ngroups * model.d_state],
            dim=-1,
        )

        scan_kwargs = dict(
            A=A,
            chunk_size=model.chunk_size,
            D=rearrange(model.D, "(h p) -> h p", p=model.headdim)
            if model.D_has_hdim
            else model.D,
            dt_bias=model.dt_bias,
            dt_softplus=True,
            x=rearrange(x, "b l (h p) -> b l h p", p=model.headdim),
            dt=dt,
            B=rearrange(B, "b l (g n) -> b l g n", g=model.ngroups),
            C=rearrange(C, "b l (g n) -> b l g n", g=model.ngroups),
            z=rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
            if not model.rmsnorm
            else None,
        )
        return scan_kwargs

    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        inputs = self.get_inputs()
        model = self.get_model()

        # The correct global outputs:
        kwargs = self.get_scan_kwargs(inputs, model)
        outputs = mamba_chunk_scan_combined(**kwargs)

        # And the CP output:
        inputs_cp_shard = rearrange(
            inputs, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]
        cp_kwargs = self.get_scan_kwargs(inputs_cp_shard, model)
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
        tol = 1e-3
        torch.testing.assert_close(outputs, outputs_cp_all_gathered, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        inputs = self.get_inputs(requires_grad=True)
        model = self.get_model()

        inputs_cp = self.get_inputs(requires_grad=True)
        inputs_cp_shard = rearrange(
            inputs_cp, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]
        model_cp = deepcopy(model)

        # Backward on the correct global result
        kwargs = self.get_scan_kwargs(inputs, model)
        outputs = mamba_chunk_scan_combined(**kwargs)
        outputs.sum().backward()

        # And on the CP output:
        kwargs_cp = self.get_scan_kwargs(inputs_cp_shard, model_cp)
        outputs_cp = mamba_chunk_scan_combined_serial_cp(mesh=mesh, **kwargs_cp)
        outputs_cp.sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = rearrange(
            inputs.grad, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]
        inputs_cp_grad_shard = rearrange(
            inputs_cp.grad, "b (r l) ... -> b r l ...", r=self.world_size
        )[:, self.rank]

        tol = 1e-3
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        grads_cp = {
            n: deepcopy(p.grad)
            for n, p in model_cp.named_parameters()
            if p.grad is not None
        }
        for g in grads_cp.values():
            dist.all_reduce(g)
        dist.barrier()  # Apparently needed for correctness if running the test under a debugger.
        assert set(grads) == set(grads_cp)
        for n, g_cp in grads_cp.items():
            g = grads[n]
            torch.testing.assert_close(
                g, g_cp, atol=tol, rtol=tol, msg=f"Failed on {n}"
            )
