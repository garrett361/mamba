from copy import deepcopy
from typing import Literal, Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import (
    CP_MAMBA_IMPLS,
    MHACP,
    Mamba2CP,
    causal_passing_comms,
    conv,
    conv_cp,
    identity_fwd_all_reduce_bwd,
    in_proj_split,
    seq_to_zigzag_comms,
    zigzag_to_seq_comms,
)
from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

"""
TODO: Parametrize the tests. Currently left unparametrized for easier debugging/dev workflow.
"""


def _test_model_model_cp_grads_close(
    model: nn.Module,
    model_cp: nn.Module,
    tol: float = 1e-2,
    all_reduce: bool = True,
) -> None:
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    grads_cp = {
        n: deepcopy(p.grad)
        for n, p in model_cp.named_parameters()
        if p.grad is not None
    }
    if all_reduce:
        for g_cp in grads_cp.values():
            dist.all_reduce(g_cp)
    dist.barrier()
    torch.cuda.synchronize()
    assert set(grads) == set(grads_cp)
    fails = {}
    for n, g_cp in grads_cp.items():
        g = grads[n]
        try:
            # NOTE: @goon - torch.testing.assert_close on the grads is an extremely strict metric,
            # which is hard to pass, so just test the mean abs diff relative to the mean abs sum.
            abs_diff = (g - g_cp).abs().mean()
            rel_diff = abs_diff / (g + g_cp).abs().mean()
            assert rel_diff < tol, f"{rel_diff =} not less than {tol=}. {abs_diff=}"
        except AssertionError as e:
            fails[n] = e
    if fails:
        err_msg = []
        err_msg.append("\n***************")
        for n, err in fails.items():
            err_msg.append(f"FAILED on {n}: {err}")
        err_msg.append("***************\n")
        raise RuntimeError("\n".join(err_msg))


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
            torch.testing.assert_close(
                t_recv,
                torch.zeros_like(t_recv),
            )
        else:
            torch.testing.assert_close(
                t_recv,
                t_send[(self.rank - 1)],
            )

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
            torch.testing.assert_close(
                grad,
                t_send[self.rank],
            )
            other_idxs = torch.arange(self.world_size, device=self.device) != self.rank
            zero_grads = t_send.grad[other_idxs]
            torch.testing.assert_close(
                zero_grads,
                torch.zeros_like(zero_grads),
            )


class TestSeqToZigZagFn(DTest):
    def test_fwd(self):
        seq_dim = 1
        dtype = torch.bfloat16
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[2 * self.rank, 2 * self.rank + 1]],
            device=self.device,
            dtype=dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = seq_to_zigzag_comms(t_send, mesh, seq_dim)
        t_expected = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=dtype,
        )
        torch.testing.assert_close(
            t_recv,
            t_expected,
        )

    def test_bwd(self):
        seq_dim = 1
        dtype = torch.bfloat16
        t_send = torch.tensor(
            [[2 * self.rank, 2 * self.rank + 1]],
            device=self.device,
            dtype=dtype,
            requires_grad=True,
        )

        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = seq_to_zigzag_comms(t_send, mesh, seq_dim)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad
        torch.testing.assert_close(
            grad,
            t_send,
        )


class TestZigZagToSeqFn(DTest):
    def test_fwd(self):
        seq_dim = 1
        dtype = torch.bfloat16
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = zigzag_to_seq_comms(t_send, mesh, seq_dim)
        t_expected = torch.tensor(
            [[2 * self.rank, 2 * self.rank + 1]],
            device=self.device,
            dtype=dtype,
        )
        torch.testing.assert_close(
            t_recv,
            t_expected,
        )

    def test_bwd(self):
        seq_dim = 1
        dtype = torch.bfloat16
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=dtype,
            requires_grad=True,
        )

        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = seq_to_zigzag_comms(t_send, mesh, seq_dim)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad
        torch.testing.assert_close(
            grad,
            t_send,
        )


class TestZigZagToSeqInverse(DTest):
    """
    zigzag_to_seq_comms and seq_to_zigzag_comms should be inverses of each other.
    """

    def test_seq_then_zig(self) -> None:
        seq_dim = 1
        dtype = torch.bfloat16
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        # Send and send again
        t_recv = seq_to_zigzag_comms(t_send, mesh, seq_dim)
        t_recv = zigzag_to_seq_comms(t_recv, mesh, seq_dim)
        torch.testing.assert_close(
            t_recv,
            t_send,
        )

    def test_zig_then_seq(self) -> None:
        seq_dim = 1
        dtype = torch.bfloat16
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        # Send and send again
        t_recv = zigzag_to_seq_comms(t_send, mesh, seq_dim)
        t_recv = seq_to_zigzag_comms(t_recv, mesh, seq_dim)
        torch.testing.assert_close(
            t_recv,
            t_send,
        )


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
    dtype = torch.bfloat16
    ssm_cfg = {"layer": "Mamba2"}
    vocab_size = 1024
    n_layer = 2
    attn_layer_idx = [n_layer - 1]
    attn_cfg = {
        "causal": True,
        "d_conv": 0,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,
    }
    cfg = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )

    @property
    def tol(self) -> float:
        return 1e-2 if self.dtype == torch.bfloat16 else 1e-3

    @property
    def batch_size(self) -> int:
        """
        batch_size == world_size is a reasonable default for testing FSDP-sharded models:
        easy to distribute world_size batch elements for a local model to batch-size-1 trained FSDP model.
        """
        return self.world_size

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
        self,
        cp_mesh: dist.device_mesh.DeviceMesh,
        seed: int = 42,
        cp_mamba_impl: str = "allgather",
    ) -> Mamba2:
        torch.manual_seed(seed)
        return Mamba2CP(
            cp_mesh=cp_mesh,
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            cp_mamba_impl=cp_mamba_impl,
            **self.factory_kwargs,
        )

    def get_mha(self, seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> MHA:
        torch.manual_seed(seed)
        # Use the bamba value rotary_emb_dim=headdim//2 so that rope is non-trivial
        return MHA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            causal=True,
            rotary_emb_dim=self.head_dim // 2,
            device=self.device,
            dtype=dtype,
        )

    def get_mha_cp(
        self,
        cp_mesh: dist.device_mesh.DeviceMesh,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
        cp_attn_impl: Literal["zigzag", "ring"] = "zigzag",
    ) -> MHA:
        torch.manual_seed(seed)
        # Use the bamba value rotary_emb_dim=headdim//2 so that rope is non-trivial
        return MHACP(
            cp_mesh=cp_mesh,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            causal=True,
            rotary_emb_dim=self.head_dim // 2,
            device=self.device,
            dtype=dtype,
            cp_attn_impl=cp_attn_impl,
        )

    def get_model(
        self, seed: int = 42, dtype: torch.dtype = torch.bfloat16
    ) -> MambaLMHeadModel:
        torch.manual_seed(seed)
        return MambaLMHeadModel(config=self.cfg, device=self.device, dtype=dtype)

    def get_model_cp(
        self,
        cp_mesh: dist.device_mesh.DeviceMesh,
        cp_mamba_impl: str,
        cp_attn_impl: str,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MambaLMHeadModel:
        torch.manual_seed(seed)
        return MambaLMHeadModel(
            config=self.cfg,
            cp_mesh=cp_mesh,
            cp_mamba_impl=cp_mamba_impl,
            cp_attn_impl=cp_attn_impl,
            device=self.device,
            dtype=dtype,
        )

    def get_input_toks(
        self,
        seed: int = 42,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randint(
            self.vocab_size,
            size=(
                self.batch_size,
                self.seq_len,
            ),
            device=self.device,
        )

    def get_inputs(
        self,
        requires_grad: bool = False,
        seed: int = 42,
        dtype: Optional[torch.dtype] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        if batch_size is None:
            batch_size = self.batch_size
        return torch.randn(
            batch_size,
            self.seq_len,
            self.d_model,
            device=self.device,
            dtype=dtype or self.dtype,
            requires_grad=requires_grad,
        )

    def get_cp_shard(
        self,
        tensor: torch.Tensor,
        n_shards: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        if n_shards is None:
            n_shards = self.world_size
        if rank is None:
            rank = self.rank

        shard = rearrange(tensor, "b (r l) ... -> b r l ...", r=n_shards)[:, rank]
        return shard

    def get_cp_hsdp_shard(
        self,
        tensor: torch.Tensor,
        mesh: dist.device_mesh.DeviceMesh,
        dp_mesh_dim: str = "inter_node",
        cp_mesh_dim: str = "intra_node",
    ) -> torch.Tensor:
        tensor_dp_shard = tensor.tensor_split(mesh[dp_mesh_dim].size(), dim=0)[
            mesh[dp_mesh_dim].get_local_rank()
        ]
        tensor_dp_cp_shard = self.get_cp_shard(
            tensor_dp_shard,
            mesh[cp_mesh_dim].size(),
            mesh[cp_mesh_dim].get_local_rank(),
        )

        return tensor_dp_cp_shard

    def get_xBC(self, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_ssm + 2 * self.ngroups * self.d_state,
            **self.factory_kwargs,
            requires_grad=requires_grad,
        )

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


class TestConvCP(_DTestModelBase):
    def test_fwd(self):
        torch.manual_seed(42)
        mamba2 = self.get_mamba2()
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        xBC = self.get_xBC()

        xBC_cp = self.get_cp_shard(xBC)

        outputs = conv(xBC, mamba2)
        outputs_cp = conv_cp(xBC_cp, mamba2, cp_mesh)
        torch.testing.assert_close(
            outputs_cp,
            outputs.tensor_split(self.world_size, dim=1)[self.rank],
            atol=self.tol,
            rtol=self.tol,
        )

    def test_bwd(self):
        torch.manual_seed(42)
        mamba2 = self.get_mamba2()
        mamba2_cp = deepcopy(mamba2)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))

        xBC = self.get_xBC(requires_grad=True)
        xBC_copy = deepcopy(xBC)

        xBC_cp = self.get_cp_shard(xBC_copy)
        outputs = conv(xBC, mamba2)
        outputs.sum().backward()
        outputs_cp = conv_cp(xBC_cp, mamba2_cp, cp_mesh)
        outputs_cp.sum().backward()

        xBC_grad_shard = self.get_cp_shard(xBC.grad)
        xBC_cp_grad_shard = self.get_cp_shard(xBC_copy.grad)
        torch.testing.assert_close(
            xBC_grad_shard, xBC_cp_grad_shard, atol=self.tol, rtol=self.tol
        )


class TestScanCP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_fwd(self, cp_mamba_impl):
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
        outputs_cp = CP_MAMBA_IMPLS[cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)

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
        torch.testing.assert_close(
            outputs, outputs_cp_all_gathered, atol=self.tol, rtol=self.tol
        )

    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_bwd(self, cp_mamba_impl):
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
        outputs_cp = CP_MAMBA_IMPLS[cp_mamba_impl](cp_mesh=cp_mesh, **kwargs_cp)
        outputs_cp.sum().backward()

        # Rearrange grads to compare proper slices
        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)

        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestMamba2CP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_fwd(self, cp_mamba_impl):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)

        inputs = self.get_inputs()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mamba2(inputs)
        outputs_cp = mamba2_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(
            outputs_cp, outputs_shard, atol=self.tol, rtol=self.tol
        )

    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_bwd(self, cp_mamba_impl):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)

        inputs = self.get_inputs(requires_grad=True)
        inputs_copy = deepcopy(inputs)
        inputs_cp = self.get_cp_shard(inputs_copy)

        outputs = mamba2(inputs)
        outputs.sum().backward()
        outputs_cp = mamba2_cp(inputs_cp)
        outputs_cp.sum().backward()

        inputs_grad_shard = self.get_cp_shard(inputs.grad)
        inputs_cp_grad_shard = self.get_cp_shard(inputs_copy.grad)
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp)


class TestMHACP(_DTestModelBase):
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_attn_impl):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)

        # Requires bfloat16
        inputs = self.get_inputs(dtype=torch.bfloat16)
        inputs_cp = self.get_cp_shard(inputs)

        outputs = mha(inputs)
        outputs_cp = mha_cp(inputs_cp)

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(
            outputs_cp, outputs_shard, atol=self.tol, rtol=self.tol
        )

    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_bwd(self, cp_attn_impl):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mha = self.get_mha()
        mha_cp = self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)

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
        dist.barrier()
        torch.testing.assert_close(
            inputs_grad_shard, inputs_cp_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        # Needs a high tol to pass?
        _test_model_model_cp_grads_close(mha, mha_cp)


class TestFSDP1MambaCP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_fwd(self, cp_mamba_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
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
        torch.testing.assert_close(
            outputs_cp_fsdp, outputs_shard, atol=self.tol, rtol=self.tol
        )

    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_bwd(self, cp_mamba_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
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
        loss = outputs.mean()
        loss.backward()

        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        loss_cp_fsdp = outputs_cp_fsdp.mean()
        loss_cp_fsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_fsdp)
            mean_loss_cp_fsdp = loss_cp_fsdp / self.world_size
            torch.testing.assert_close(
                loss, mean_loss_cp_fsdp, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm, grad_norm_cp_fsdp, atol=self.tol, rtol=self.tol
        )

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestFSDP1MHACP(_DTestModelBase):
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_attn_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
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
        torch.testing.assert_close(
            outputs_cp_fsdp,
            outputs_shard,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_bwd(self, cp_attn_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
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
        loss = outputs.mean()
        loss.backward()

        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp)
        loss_cp_fsdp = outputs_cp_fsdp.mean()
        loss_cp_fsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_fsdp)
            mean_loss_cp_fsdp = loss_cp_fsdp / self.world_size
            torch.testing.assert_close(
                loss, mean_loss_cp_fsdp, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm, grad_norm_cp_fsdp, atol=self.tol, rtol=self.tol
        )

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestHSDP1MambaCP(_DTestModelBase):
    """
    Test HSDP + CP for MambaCP where the HSDP and CP dims cover the same subgroups
    """

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_fwd(self, cp_mamba_impl):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs(batch_size=mesh["inter_node"].size())
        inputs_cp_hsdp = self.get_cp_hsdp_shard(inputs, mesh)

        outputs = model(inputs)
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp_hsdp)

        outputs_shard = self.get_cp_hsdp_shard(outputs, mesh)
        torch.testing.assert_close(
            outputs_cp_hsdp,
            outputs_shard,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    def test_bwd(self, cp_mamba_impl):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(cp_mesh=cp_mesh, cp_mamba_impl=cp_mamba_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
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

        inputs = self.get_inputs(
            batch_size=mesh["inter_node"].size(), requires_grad=True
        )
        inputs_copy = deepcopy(inputs)
        inputs_cp_hsdp = self.get_cp_hsdp_shard(inputs_copy, mesh)

        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp_hsdp)
        loss_cp_hsdp = outputs_cp_hsdp.mean()
        loss_cp_hsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_hsdp)
            mean_loss_cp_hsdp = loss_cp_hsdp / self.world_size
            torch.testing.assert_close(
                loss, mean_loss_cp_hsdp, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_hsdp = model_cp_hsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm, grad_norm_cp_hsdp, atol=self.tol, rtol=self.tol
        )

        inputs_grad_shard = self.get_cp_hsdp_shard(inputs.grad, mesh)
        inputs_copy_grad_shard = self.get_cp_hsdp_shard(inputs_copy.grad, mesh)
        torch.testing.assert_close(
            inputs_copy_grad_shard,
            inputs_grad_shard,
            atol=self.tol,
            rtol=self.tol,
        )

        with FSDP.summon_full_params(model_cp_hsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_hsdp, all_reduce=False)


class TestHSDP1MHACP(_DTestModelBase):
    """
    Test HSDP + CP for MHACP where the HSDP and CP dims cover the same subgroups
    """

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_attn_impl):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_inputs(
            batch_size=mesh["inter_node"].size(), dtype=torch.bfloat16
        )
        inputs_cp_hsdp = self.get_cp_hsdp_shard(inputs, mesh)

        outputs = model(inputs)
        outputs_cp_hsdp = model_cp_hsdp(inputs_cp_hsdp)

        outputs_shard = self.get_cp_hsdp_shard(outputs, mesh)
        torch.testing.assert_close(
            outputs_cp_hsdp,
            outputs_shard,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_bwd(self, cp_attn_impl):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mha() for _ in range(3)]).to(self.dtype)
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                for _ in range(3)
            ]
        ).to(self.dtype)
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
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

        inputs = self.get_inputs(
            batch_size=mesh["inter_node"].size(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inputs_copy = deepcopy(inputs)
        inputs_cp_hsdp = self.get_cp_hsdp_shard(inputs_copy, mesh)

        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()

        outputs_cp_hsdp = model_cp_hsdp(inputs_cp_hsdp)
        loss_cp_hsdp = outputs_cp_hsdp.mean()
        loss_cp_hsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_hsdp)
            mean_loss_cp_hsdp = loss_cp_hsdp / self.world_size
            torch.testing.assert_close(
                loss, mean_loss_cp_hsdp, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_hsdp = model_cp_hsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm, grad_norm_cp_hsdp, atol=self.tol, rtol=self.tol
        )

        inputs_grad_shard = self.get_cp_hsdp_shard(inputs.grad, mesh)
        inputs_copy_grad_shard = self.get_cp_hsdp_shard(inputs_copy.grad, mesh)
        torch.testing.assert_close(
            inputs_copy_grad_shard,
            inputs_grad_shard,
            atol=self.tol,
            rtol=self.tol,
        )

        with FSDP.summon_full_params(model_cp_hsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_hsdp, all_reduce=False)


class TestModel(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_mamba_impl, cp_attn_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=cp_mamba_impl, cp_attn_impl=cp_attn_impl
        )

        inputs = self.get_input_toks()
        inputs_cp = self.get_cp_shard(inputs)

        outputs = model(inputs).logits
        outputs_cp = model_cp(inputs_cp).logits

        outputs_shard = self.get_cp_shard(outputs)
        torch.testing.assert_close(
            outputs_cp,
            outputs_shard,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.parametrize("cp_mamba_impl", ("serial", "allgather"))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_bwd(self, cp_mamba_impl, cp_attn_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp = self.get_model_cp(
            cp_mesh, cp_mamba_impl=cp_mamba_impl, cp_attn_impl=cp_attn_impl
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
