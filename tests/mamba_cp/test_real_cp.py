from copy import deepcopy
from typing import Optional

from mamba_ssm.modules.mha import MHA
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from dtest import DTest
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import (
    MHACP,
    Mamba2CP,
    causal_passing_comms,
    conv,
    conv_cp,
    identity_fwd_all_reduce_bwd,
    in_proj_split,
)
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
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
    embed_dim = d_model
    head_dim = 64
    num_heads = d_model // head_dim
    dtype = torch.float32

    @property
    def seq_len(self) -> int:
        return 4 * self.world_size * self.chunk_size

    @property
    def n_chunks(self) -> int:
        return self.seq_len // self.chunk_size

    @property
    def factory_kwargs(self):
        return {"dtype": self.dtype, "device": self.device}

    def get_mamba2(self, seed: int = 42) -> Mamba2:
        torch.manual_seed(seed)
        return Mamba2(
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            **self.factory_kwargs,
        )

    def get_mamba2_cp(
        self, mesh: dist.device_mesh.DeviceMesh, seed: int = 42
    ) -> Mamba2:
        torch.manual_seed(seed)
        return Mamba2CP(
            mesh=mesh,
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            **self.factory_kwargs,
        )

    def get_mha(self, seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> MHA:
        torch.manual_seed(seed)
        return MHA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            device=self.device,
            dtype=dtype,
        )

    def get_mha_cp(
        self,
        mesh: dist.device_mesh.DeviceMesh,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MHA:
        torch.manual_seed(seed)
        return MHACP(
            mesh=mesh,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            device=self.device,
            dtype=dtype,
        )

    def get_inputs(
        self,
        requires_grad: bool = False,
        seed: int = 42,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_model,
            device=self.device,
            dtype=dtype or self.dtype,
            requires_grad=requires_grad,
        )

    def get_cp_shard(
        self, tensor: torch.Tensor, n_shards: Optional[int] = None
    ) -> torch.Tensor:
        n_shards = n_shards or self.world_size
        shard = rearrange(tensor, "b (r l) ... -> b r l ...", r=self.world_size)[
            :, self.rank
        ]
        return shard


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
        mamba2 = self.get_mamba2()
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        xBC = self.get_xBC()

        xBC_cp = self.get_cp_shard(xBC)

        outputs = conv(xBC, mamba2)
        outputs_cp = conv_cp(xBC_cp, mamba2, mesh)
        torch.testing.assert_close(
            outputs_cp, outputs.tensor_split(self.world_size, dim=1)[self.rank]
        )

    def test_bwd(self):
        torch.manual_seed(42)
        mamba2 = self.get_mamba2()
        mamba2_cp = deepcopy(mamba2)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        xBC = self.get_xBC(requires_grad=True)
        xBC_cp = deepcopy(xBC)

        xBC_cp_shard = self.get_cp_shard(xBC_cp)
        outputs = conv(xBC, mamba2)
        outputs.sum().backward()
        outputs_cp = conv_cp(xBC_cp_shard, mamba2_cp, mesh)
        outputs_cp.sum().backward()

        xBC_grad_shard = self.get_cp_shard(xBC.grad)
        xBC_cp_grad_shard = self.get_cp_shard(xBC_cp.grad)
        torch.testing.assert_close(xBC_grad_shard, xBC_cp_grad_shard)


class TestSerialCP(_DTestModelBase):
    def get_scan_kwargs(self, inputs, mamba2):
        """
        Get all kwargs for the scan.
        """
        A = -torch.exp(mamba2.A_log.float())  # (nheads) or (d_inner, d_state)
        _, _, z, xBC, dt = in_proj_split(inputs, mamba2)
        x, B, C = torch.split(
            xBC,
            [
                mamba2.d_ssm,
                mamba2.ngroups * mamba2.d_state,
                mamba2.ngroups * mamba2.d_state,
            ],
            dim=-1,
        )

        scan_kwargs = dict(
            A=A,
            chunk_size=mamba2.chunk_size,
            D=rearrange(mamba2.D, "(h p) -> h p", p=mamba2.headdim)
            if mamba2.D_has_hdim
            else mamba2.D,
            dt_bias=mamba2.dt_bias,
            dt_softplus=True,
            x=rearrange(x, "b l (h p) -> b l h p", p=mamba2.headdim),
            dt=dt,
            B=rearrange(B, "b l (g n) -> b l g n", g=mamba2.ngroups),
            C=rearrange(C, "b l (g n) -> b l g n", g=mamba2.ngroups),
            z=rearrange(z, "b l (h p) -> b l h p", p=mamba2.headdim)
            if not mamba2.rmsnorm
            else None,
        )
        return scan_kwargs

    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        inputs = self.get_inputs()
        mamba2 = self.get_mamba2()

        # The correct global outputs:
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)

        # And the CP output:
        inputs_cp_shard = self.get_cp_shard(inputs)
        cp_kwargs = self.get_scan_kwargs(inputs_cp_shard, mamba2)
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
        mamba2 = self.get_mamba2()

        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)
        mamba2_cp = deepcopy(mamba2)

        # Backward on the correct global result
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)
        outputs.sum().backward()

        # And on the CP output:
        kwargs_cp = self.get_scan_kwargs(inputs_cp_shard, mamba2_cp)
        outputs_cp = mamba_chunk_scan_combined_serial_cp(mesh=mesh, **kwargs_cp)
        outputs_cp.sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_cp.grad)

        tol = 1e-3
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        grads = {n: p.grad for n, p in mamba2.named_parameters() if p.grad is not None}
        grads_cp = {
            n: deepcopy(p.grad)
            for n, p in mamba2_cp.named_parameters()
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


class TestMamba2CPSerial(_DTestModelBase):
    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(mesh=mesh)

        inputs = self.get_inputs()
        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)

        outputs = mamba2(inputs)
        outputs_cp = mamba2_cp(inputs_cp_shard)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp, outputs_shard)

    def test_bwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(mesh=mesh)

        inputs = self.get_inputs(requires_grad=True)
        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)

        outputs = mamba2(inputs)
        outputs.sum().backward()
        outputs_cp = mamba2_cp(inputs_cp_shard)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_cp.grad)
        torch.testing.assert_close(inputs_grad_shard, inputs_cp_grad_shard)

        # Parameter grads should match after all-reducing.
        grads = {n: p.grad for n, p in mamba2.named_parameters() if p.grad is not None}
        grads_cp = {
            n: deepcopy(p.grad)
            for n, p in mamba2_cp.named_parameters()
            if p.grad is not None
        }
        for g in grads_cp.values():
            dist.all_reduce(g)
        dist.barrier()  # Apparently needed for correctness if running the test under a debugger.
        assert set(grads) == set(grads_cp)
        tol = 1e-3
        for n, g_cp in grads_cp.items():
            g = grads[n]
            torch.testing.assert_close(
                g, g_cp, atol=tol, rtol=tol, msg=f"Failed on {n}"
            )


class TestMHACP(_DTestModelBase):
    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(mesh=mesh)

        # Requires bfloat16
        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)

        outputs = mha(inputs)
        outputs_cp = mha_cp(inputs_cp_shard)

        outputs_shard = self.get_cp_shard(outputs)
        tol = 1e-2
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(mesh=mesh)

        # Requires bfloat16
        inputs = self.get_inputs(requires_grad=True, dtype=torch.bfloat16)
        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)

        outputs = mha(inputs)
        outputs.sum().backward()
        outputs_cp = mha_cp(inputs_cp_shard)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_cp.grad)
        tol = 1e-2
        dist.barrier()
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        grads = {n: p.grad for n, p in mha.named_parameters() if p.grad is not None}
        grads_cp = {
            n: deepcopy(p.grad)
            for n, p in mha_cp.named_parameters()
            if p.grad is not None
        }
        for g in grads_cp.values():
            dist.all_reduce(g)
        dist.barrier()
        assert set(grads) == set(grads_cp)
        tol = 1e-1
        for n, g_cp in grads_cp.items():
            g = grads[n]
            torch.testing.assert_close(
                g, g_cp, atol=tol, rtol=tol, msg=f"Failed on {n}"
            )


class TestFSDP1MambaCPSerial(_DTestModelBase):
    def test_fwd(self):
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(*[self.get_mamba2_cp(mesh=mesh) for _ in range(3)])
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs()
        inputs_cp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp)

        outputs = model(inputs)
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_shard)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp_fsdp, outputs_shard)

    def test_bwd(self):
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(*[self.get_mamba2_cp(mesh=mesh) for _ in range(3)])
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # Note that 2) is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        inputs = self.get_inputs(requires_grad=True)
        inputs_cp_fsdp = deepcopy(inputs)
        inputs_cp_shard = self.get_cp_shard(inputs_cp_fsdp)

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_shard)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            grads = {
                n: deepcopy(p.grad)
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            grads_cp_fsdp = {
                n: deepcopy(p.grad)
                for n, p in model_cp_fsdp.named_parameters()
                if p.grad is not None
            }
            tol = 1e-3
            for n, g_cp_fsdp in grads_cp_fsdp.items():
                g = grads[n]
                torch.testing.assert_close(
                    g,
                    g_cp_fsdp,
                    atol=tol,
                    rtol=tol,
                    msg=f"Failed on {n}: {(g - g_cp_fsdp).abs().mean()=}, {(g - g_cp_fsdp).abs().max()=}",
                )
