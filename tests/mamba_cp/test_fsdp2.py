from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from test_real_cp import _DTestModelBase
from torch.distributed.fsdp import fully_shard

from mamba_ssm.modules.block import Block

"""
TODO: Parametrize the tests. Currently left unparametrized for easier debugging/dev workflow.
"""


class TestFSDP2(_DTestModelBase):
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

    def test_bwd_model(self) -> None:
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
        out.mean().backward()
        out_copy.mean().backward()
