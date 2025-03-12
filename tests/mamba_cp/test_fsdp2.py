from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2_cp import (
    CP_MAMBA_IMPLS,
    MHACP,
    Mamba2CP,
)
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .test_real_cp import _DTestModelBase, _test_model_model_cp_grads_close

"""
TODO: Parametrize the tests. Currently left unparametrized for easier debugging/dev workflow.
"""


class TestFSDP2(_DTestModelBase):
    def test_fwd(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(
            *[nn.Linear(10, 10, device=self.device) for _ in range(3)]
        )
        model_copy = deepcopy(model)
        for lin in model:
            fully_shard(lin, mesh=mesh)
        fully_shard(model, mesh=mesh)
        inputs = torch.randn(1, 10, device=self.device)
        out = model(inputs)
        out_copy = model_copy(inputs)
        torch.testing.assert_close(out, out_copy)

    def test_mamba2(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_copy = deepcopy(model)

        for module in model:
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        inputs = self.get_inputs()
        out_copy = model_copy(inputs)
        out = model(inputs)
        torch.testing.assert_close(out, out_copy)
        out.sum().backward()
        out_copy.sum().backward()
        out

    def test_fwd_model(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        # FSDP2 expects uniform dtype
        model = self.get_model(dtype=torch.float32)
        model_copy = deepcopy(model)
        input_toks = self.get_input_toks()

        for module in model.modules():
            if isinstance(module, Block):
                fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        out_copy = model_copy(input_toks)
        out = model(input_toks)
        torch.testing.assert_close(out.logits, out_copy.logits)


class TestSerialScanCP(_DTestModelBase):
    cp_mamba_impl = "serial"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        inputs = self.get_inputs()
        mamba2 = self.get_mamba2()

        # The correct global outputs:
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)

        # And the CP output:
        inputs_cp_shard = self.get_cp_shard(inputs)
        kwargs_cp = self.get_scan_kwargs(inputs_cp_shard, mamba2)
        outputs_cp = CP_MAMBA_IMPLS[self.cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)

        # All-gather and verify correctness
        outputs_cp_all_gathered = torch.empty(
            self.world_size,
            *outputs_cp.shape,
            dtype=outputs_cp.dtype,
            device=outputs_cp.device,
        )
        dist.all_gather_into_tensor(
            outputs_cp_all_gathered, outputs_cp, cp_mesh.get_group()
        )
        outputs_cp_all_gathered = rearrange(
            outputs_cp_all_gathered, "r b l ... -> b (r l) ..."
        )
        tol = 1e-3
        torch.testing.assert_close(outputs, outputs_cp_all_gathered, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        inputs = self.get_inputs(requires_grad=True)
        mamba2 = self.get_mamba2()

        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)
        mamba2_cp = deepcopy(mamba2)

        # Backward on the correct global result
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)
        outputs.sum().backward()

        # And on the CP output:
        kwargs_cp = self.get_scan_kwargs(inputs_cp, mamba2_cp)
        outputs_cp = CP_MAMBA_IMPLS[self.cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)
        outputs_cp.sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)

        tol = 1e-2
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestAllGatherScanCP(_DTestModelBase):
    cp_mamba_impl = "allgather"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        inputs = self.get_inputs()
        mamba2 = self.get_mamba2()

        # The correct global outputs:
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)

        # And the CP output:
        inputs_cp_shard = self.get_cp_shard(inputs)
        kwargs_cp = self.get_scan_kwargs(inputs_cp_shard, mamba2)
        outputs_cp = CP_MAMBA_IMPLS[self.cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)

        # All-gather and verify correctness
        outputs_cp_all_gathered = torch.empty(
            self.world_size,
            *outputs_cp.shape,
            dtype=outputs_cp.dtype,
            device=outputs_cp.device,
        )
        dist.all_gather_into_tensor(
            outputs_cp_all_gathered, outputs_cp, cp_mesh.get_group()
        )
        outputs_cp_all_gathered = rearrange(
            outputs_cp_all_gathered, "r b l ... -> b (r l) ..."
        )
        tol = 1e-3
        torch.testing.assert_close(outputs, outputs_cp_all_gathered, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        inputs = self.get_inputs(requires_grad=True)
        mamba2 = self.get_mamba2()

        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)
        mamba2_cp = deepcopy(mamba2)

        # Backward on the correct global result
        kwargs = self.get_scan_kwargs(inputs, mamba2)
        outputs = mamba_chunk_scan_combined(**kwargs)
        outputs.sum().backward()

        # And on the CP output:
        kwargs_cp = self.get_scan_kwargs(inputs_cp, mamba2_cp)
        outputs_cp = CP_MAMBA_IMPLS[self.cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)
        outputs_cp.sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)

        tol = 1e-2
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestMamba2CPSerial(_DTestModelBase):
    cp_mamba_impl = "serial"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(
            cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl
        )

        inputs = self.get_inputs()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mamba2(inputs)
        outputs_cp = mamba2_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp, outputs_shard)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(
            cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl
        )

        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = mamba2(inputs)
        outputs.sum().backward()
        outputs_cp = mamba2_cp(inputs_cp)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)
        torch.testing.assert_close(inputs_grad_shard, inputs_cp_grad_shard)

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestMamba2CPAllGather(_DTestModelBase):
    cp_mamba_impl = "allgather"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(
            cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl
        )

        inputs = self.get_inputs()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mamba2(inputs)
        outputs_cp = mamba2_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp, outputs_shard)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(
            cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl
        )

        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = mamba2(inputs)
        outputs.sum().backward()
        outputs_cp = mamba2_cp(inputs_cp)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)
        torch.testing.assert_close(inputs_grad_shard, inputs_cp_grad_shard)

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestMHACPRing(_DTestModelBase):
    cp_attn_impl = "ring"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)

        # Requires bfloat16
        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mha(inputs)
        outputs_cp = mha_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        tol = 1e-2
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)

        # Requires bfloat16
        inputs = self.get_inputs(requires_grad=True, dtype=torch.bfloat16)
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = mha(inputs)
        outputs.sum().backward()
        outputs_cp = mha_cp(inputs_cp)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)
        tol = 1e-2
        dist.barrier()
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        # Needs a high tol to pass?
        _test_model_model_cp_grads_close(mha, mha_cp)


class TestMHACPZigZag(_DTestModelBase):
    cp_attn_impl = "zigzag"

    def test_fwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)

        # Requires bfloat16
        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mha(inputs)
        outputs_cp = mha_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        tol = 1e-2
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)

        # Requires bfloat16
        inputs = self.get_inputs(requires_grad=True, dtype=torch.bfloat16)
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = mha(inputs)
        outputs.sum().backward()
        outputs_cp = mha_cp(inputs_cp)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)
        tol = 1e-2
        dist.barrier()
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=tol, rtol=tol
        )

        # Parameter grads should match after all-reducing.
        # Needs a high tol to pass?
        _test_model_model_cp_grads_close(mha, mha_cp)


class TestFSDP1MambaCPSerial(_DTestModelBase):
    cp_mamba_impl = "serial"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs)
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp_fsdp, outputs_shard)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        inputs_cp_fsdp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestFSDP1MambaCPAllGather(_DTestModelBase):
    cp_mamba_impl = "allgather"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs)
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(outputs_cp_fsdp, outputs_shard)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        inputs_cp_fsdp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestFSDP1MHACPRing(_DTestModelBase):
    cp_attn_impl = "ring"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs)
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        tol = 1e-3
        torch.testing.assert_close(outputs_cp_fsdp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        inputs = self.get_inputs(requires_grad=True, dtype=torch.bfloat16)
        inputs_copy = deepcopy(inputs)
        inputs_cp_fsdp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestFSDP1MHACPZigZag(_DTestModelBase):
    cp_attn_impl = "zigzag"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs)
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        tol = 1e-3
        torch.testing.assert_close(outputs_cp_fsdp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_fsdp = FSDP(
            model_cp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        inputs = self.get_inputs(requires_grad=True, dtype=torch.bfloat16)
        inputs_copy = deepcopy(inputs)
        inputs_cp_fsdp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestHSDP1MambaCPAllGather(_DTestModelBase):
    """
    Test HSDP + CP for MambaCP where the HSDP and CP dims cover the same subgroups
    """

    cp_mamba_impl = "allgather"

    @pytest.mark.world_size(4)
    def test_fwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # Different CP groups get different inputs
        inputs = self.get_inputs(seed=42 + mesh["inter_node"].get_local_rank())
        inputs_cp = self.get_cp_shard(
            inputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )

        outputs = model(inputs)
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(
            outputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )
        torch.testing.assert_close(outputs_cp_hsdp, outputs_shard)

    @pytest.mark.world_size(4)
    def test_bwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=self.cp_mamba_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        # Different CP groups get different inputs

        n_cp_groups = mesh["inter_node"].size()
        inputs = self.get_inputs(requires_grad=True, batch_size=n_cp_groups)
        inputs_copy = deepcopy(inputs)
        inputs_cp_hsdp = self.get_cp_shard(
            inputs_copy.tensor_split(n_cp_groups, dim=0)[cp_mesh.get_local_rank()],
            n_shards=cp_mesh.size(),
            rank=cp_mesh.get_local_rank(),
        )

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_hsdp(inputs_cp_hsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_hsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_hsdp, all_reduce=False)


class TestHSDP1MHACPRing(_DTestModelBase):
    """
    Test HSDP + CP for MHACP where the HSDP and CP dims cover the same subgroups
    """

    cp_attn_impl = "ring"

    @pytest.mark.world_size(4)
    def test_fwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # Different CP groups get different inputs
        inputs = self.get_inputs(
            seed=42 + mesh["inter_node"].get_local_rank(), dtype=torch.bfloat16
        )
        inputs_cp = self.get_cp_shard(
            inputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )

        outputs = model(inputs)
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(
            outputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )
        tol = 1e-3
        torch.testing.assert_close(outputs_cp_hsdp, outputs_shard, atol=tol, rtol=tol)

    @pytest.mark.world_size(4)
    def test_bwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        # Different CP groups get different inputs

        n_cp_groups = mesh["inter_node"].size()
        inputs = self.get_inputs(
            requires_grad=True, batch_size=n_cp_groups, dtype=torch.bfloat16
        )
        inputs_copy = deepcopy(inputs)
        inputs_cp_hsdp = self.get_cp_shard(
            inputs_copy.tensor_split(n_cp_groups, dim=0)[cp_mesh.get_local_rank()],
            n_shards=cp_mesh.size(),
            rank=cp_mesh.get_local_rank(),
        )

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_hsdp(inputs_cp_hsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_hsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_hsdp, all_reduce=False)


class TestHSDP1MHACPZigZag(_DTestModelBase):
    """
    Test HSDP + CP for MHACP where the HSDP and CP dims cover the same subgroups
    """

    cp_attn_impl = "zigzag"

    @pytest.mark.world_size(4)
    def test_fwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # Different CP groups get different inputs
        inputs = self.get_inputs(
            seed=42 + mesh["inter_node"].get_local_rank(), dtype=torch.bfloat16
        )
        inputs_cp = self.get_cp_shard(
            inputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )

        outputs = model(inputs)
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp)

        outputs_shard = self.get_cp_shard(
            outputs, n_shards=cp_mesh.size(), rank=cp_mesh.get_local_rank()
        )
        tol = 1e-3
        torch.testing.assert_close(outputs_cp_hsdp, outputs_shard, atol=tol, rtol=tol)

    @pytest.mark.world_size(4)
    def test_bwd(self):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=self.cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )
        # NOTE: for grads to match the non-FSDP case, some scaling need to be performed. Options:
        # 1) Change FSDP._gradient_predivide_factor from the DP world size to 1.0
        # 2) Scale up the loss by a factor of the DP world size
        # The latter is also automatically handled if the loss function scales by the world size,
        # as in the case of a mean or default cross_entropy loss with equals tokens per rank, but it
        # also makes the outputs mismatch, while 1) keeps the FSDP and non-FSDP outputs the same.
        #
        # We just use the mean for the loss below.

        # Different CP groups get different inputs

        n_cp_groups = mesh["inter_node"].size()
        inputs = self.get_inputs(
            requires_grad=True, batch_size=n_cp_groups, dtype=torch.bfloat16
        )
        inputs_copy = deepcopy(inputs)
        inputs_cp_hsdp = self.get_cp_shard(
            inputs_copy.tensor_split(n_cp_groups, dim=0)[cp_mesh.get_local_rank()],
            n_shards=cp_mesh.size(),
            rank=cp_mesh.get_local_rank(),
        )

        outputs = model(inputs)
        outputs.mean().backward()
        outputs_cp_fsdp = model_cp_hsdp(inputs_cp_hsdp)
        outputs_cp_fsdp.mean().backward()

        with FSDP.summon_full_params(model_cp_hsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_hsdp, all_reduce=False)


class TestModelSerialRing(_DTestModelBase):
    cp_mamba_impl = "serial"
    cp_attn_impl = "ring"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs).logits
        outputs_cp = model_cp(inputs_cp).logits

        outputs_shard = self.get_cp_shard(outputs)
        # Requires a higher tolerance.
        tol = 1e-1
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs).logits
        outputs.sum().backward()
        outputs_cp = model_cp(inputs_cp).logits
        outputs_cp.sum().backward()

        # Parameter grads should match after all-reducing.
        # Requires high tol
        _test_model_model_cp_grads_close(model, model_cp)


class TestModelAllGatherZigZag(_DTestModelBase):
    cp_mamba_impl = "allgather"
    cp_attn_impl = "zigzag"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs).logits
        outputs_cp = model_cp(inputs_cp).logits

        outputs_shard = self.get_cp_shard(outputs)
        # Requires a higher tolerance.
        tol = 1e-1
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs).logits
        outputs.sum().backward()
        outputs_cp = model_cp(inputs_cp).logits
        outputs_cp.sum().backward()

        # Parameter grads should match after all-reducing.
        # Requires high tol
        _test_model_model_cp_grads_close(model, model_cp)


class TestModelSerialZigZag(_DTestModelBase):
    cp_mamba_impl = "serial"
    cp_attn_impl = "zigzag"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs).logits
        outputs_cp = model_cp(inputs_cp).logits

        outputs_shard = self.get_cp_shard(outputs)
        # Requires a higher tolerance.
        tol = 1e-1
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs).logits
        outputs.sum().backward()
        outputs_cp = model_cp(inputs_cp).logits
        outputs_cp.sum().backward()

        # Parameter grads should match after all-reducing.
        # Requires high tol
        _test_model_model_cp_grads_close(model, model_cp)


class TestModelAllGatherRing(_DTestModelBase):
    cp_mamba_impl = "allgather"
    cp_attn_impl = "ring"

    def test_fwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs).logits
        outputs_cp = model_cp(inputs_cp).logits

        outputs_shard = self.get_cp_shard(outputs)
        # Requires a higher tolerance.
        tol = 1e-1
        torch.testing.assert_close(outputs_cp, outputs_shard, atol=tol, rtol=tol)

    def test_bwd(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=self.cp_mamba_impl, cp_attn_impl=self.cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = model(inputs).logits
        outputs.sum().backward()
        outputs_cp = model_cp(inputs_cp).logits
        outputs_cp.sum().backward()

        # Parameter grads should match after all-reducing.
        # Requires high tol
        _test_model_model_cp_grads_close(model, model_cp)
