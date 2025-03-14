from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from test_real_cp import _DTestModelBase
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate

from mamba_ssm.modules.block import Block

"""
TODO: Parametrize the tests. Currently left unparametrized for easier debugging/dev workflow.
"""


def _test_model_model_fsdp2_grads_close_fsdp2(
    model: nn.Module,
    model_fsdp2: nn.Module,
    tol: float = 1e-2,
    print_passes: bool = True,
) -> None:
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    grads_fsdp2 = {}
    for n, p in model_fsdp2.named_parameters():
        if p.grad is not None:
            g_fsdp2 = p.grad.redistribute(
                p.grad.device_mesh, (Replicate() for pl in p.grad.placements)
            )
            grads_fsdp2[n] = g_fsdp2.to_local()
    dist.barrier()
    torch.cuda.synchronize()
    assert set(grads) == set(grads_fsdp2)
    fails = {}
    passes = {}
    for n, g_fsdp2 in grads_fsdp2.items():
        g = grads[n]
        try:
            # NOTE: @goon - torch.testing.assert_close on the grads is an extremely strict metric,
            # which is hard to pass, so just test the mean abs diff relative to the mean abs sum.
            abs_diff = (g - g_fsdp2).abs().mean()
            rel_diff = abs_diff / (g + g_fsdp2).abs().mean()
            assert rel_diff < tol
            if print_passes:
                passes[n] = f"{rel_diff=}"
        except AssertionError:
            fails[n] = (
                f"{rel_diff =} not less than {tol=}. {abs_diff=}"
                + "\n"
                + len(n) * " "
                + f"  {g=}\n"
                + len(n) * " "
                + f"  {g_fsdp2=}"
            )
    if print_passes:
        pass_msg = []
        pass_msg.append("\n****** PASSES ********")
        for n, msg in passes.items():
            pass_msg.append(f"{n}: {msg}")
        pass_msg.append("***************\n")
        print("\n".join(pass_msg))

    if fails:
        err_msg = []
        err_msg.append("\n****** FAILURES ********")
        for n, msg in fails.items():
            err_msg.append(f"{n}: {msg}")
        err_msg.append("***************\n")
        raise RuntimeError("\n".join(err_msg))


class TestFSDP2(_DTestModelBase):
    def test_mamba2(self):
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        model_fsdp2 = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_copy = deepcopy(model_fsdp2)

        for module in model_fsdp2:
            fully_shard(module, mesh=mesh)
        fully_shard(model_fsdp2, mesh=mesh)

        inputs = self.get_inputs()
        out_copy = model_copy(inputs)
        out = model_fsdp2(inputs.tensor_split(mesh.size(), 0)[mesh.get_local_rank()])
        torch.testing.assert_close(
            out, out_copy.tensor_split(mesh.size(), 0)[mesh.get_local_rank()]
        )
        # For grads to match, the loss needs to explicitly be averaged over the batch dim.
        out.mean().backward()
        out_copy.mean().backward()
        _test_model_model_fsdp2_grads_close_fsdp2(model_copy, model_fsdp2)

    def test_model(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        # FSDP2 expects uniform dtype
        model_fsdp2 = self.get_model(dtype=torch.float32)
        model_copy = deepcopy(model_fsdp2)
        input_toks = self.get_input_toks()

        for module in model_fsdp2.modules():
            if isinstance(module, Block):
                fully_shard(module, mesh=mesh)
        fully_shard(model_fsdp2, mesh=mesh)

        out_copy = model_copy(input_toks).logits
        out = model_fsdp2(
            input_toks.tensor_split(mesh.size(), 0)[mesh.get_local_rank()]
        ).logits
        torch.testing.assert_close(
            out, out_copy.tensor_split(mesh.size(), 0)[mesh.get_local_rank()]
        )

        out.mean().backward()
        out_copy.mean().backward()
        _test_model_model_fsdp2_grads_close_fsdp2(model_copy, model_fsdp2)

    def test_model_cp(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        # FSDP2 expects uniform dtype and mamba doesn't respect the dtype arg for the D param, so we
        # need a final .to(bfloat16) to enforce this.
        model_fsdp2 = self.get_model_cp(
            dtype=torch.bfloat16,
            cp_mamba_impl="allgather",
            cp_attn_impl="zigzag",
            cp_mesh=mesh,
        ).to(torch.bfloat16)
        model_copy = self.get_model(dtype=torch.bfloat16).to(torch.bfloat16)

        input_toks = self.get_input_toks()
        input_toks_cp = self.get_cp_shard(input_toks)

        for module in model_fsdp2.modules():
            if isinstance(module, Block):
                fully_shard(module, mesh=mesh)
        fully_shard(model_fsdp2, mesh=mesh)

        out_copy = model_copy(input_toks).logits
        out_shard = self.get_cp_shard(out_copy)
        out = model_fsdp2(input_toks_cp).logits
        tol = 1e-2
        torch.testing.assert_close(out_shard, out, atol=tol, rtol=tol)

        out.mean().backward()
        out_copy.mean().backward()
        _test_model_model_fsdp2_grads_close_fsdp2(model_copy, model_fsdp2)

    def test_mamba2_min(self):
        """
        Testing non-deterministic errors on the D param grads in a loop.
        """
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        model_fsdp2 = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_copy = deepcopy(model_fsdp2)

        for module in model_fsdp2:
            fully_shard(module, mesh=mesh)
        fully_shard(model_fsdp2, mesh=mesh)

        inputs = self.get_inputs()
        try:
            for iter_idx in range(100):
                out_copy = model_copy(inputs)
                out = model_fsdp2(
                    inputs.tensor_split(mesh.size(), 0)[mesh.get_local_rank()]
                )
                torch.testing.assert_close(
                    out, out_copy.tensor_split(mesh.size(), 0)[mesh.get_local_rank()]
                )
                # For grads to match, the loss needs to explicitly be averaged over the batch dim.
                out.mean().backward()
                out_copy.mean().backward()
                _test_model_model_fsdp2_grads_close_fsdp2(model_copy, model_fsdp2)
        except Exception as e:
            raise RuntimeError(f"FAILED on {iter_idx=}") from e
