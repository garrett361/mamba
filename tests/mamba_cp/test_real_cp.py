import warnings
from copy import deepcopy
from typing import Literal, Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.tensor import DTensor, Replicate

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import (
    CP_MAMBA_IMPLS,
    MHACP,
    Mamba2CP,
    causal_passing_comms,
    conv,
    conv_cp,
    in_proj_split,
    seq_to_zigzag_comms,
    zigzag_to_seq_comms,
)
from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

"""
TODO: Parametrize the tests. Currently left unparametrized for easier debugging/dev workflow.
"""

# HACK: @goon - work around D-grad issues
PARAM_SUBSTRS_TO_IGNORE = [".D"]


def _test_model_model_cp_grads_close(
    model: nn.Module,
    model_cp: nn.Module,
    tol: float = 1e-2,
    all_reduce: bool = False,
    print_passes: bool = True,
    param_substrs_to_ignore: Optional[list[str]] = PARAM_SUBSTRS_TO_IGNORE,
) -> None:
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    grads_cp = {}
    for n, p in model_cp.named_parameters():
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                # FSDP2
                g_cp = p.grad.redistribute(
                    p.grad.device_mesh, (Replicate() for pl in p.grad.placements)
                ).to_local()
            else:
                g_cp = deepcopy(p.grad)
            if all_reduce:
                dist.all_reduce(g_cp)
            grads_cp[n] = g_cp

    dist.barrier()
    assert set(grads) == set(grads_cp)
    fails = {}
    passes = {}
    for n, g_cp in grads_cp.items():
        if param_substrs_to_ignore is not None and any(
            s in n for s in param_substrs_to_ignore
        ):
            continue
        g = grads[n]
        try:
            # NOTE: @goon - torch.testing.assert_close on the grads is an extremely strict metric,
            # which is hard to pass, so just test the mean abs diff relative to the mean abs sum.
            abs_diff = (g - g_cp).abs().mean()
            rel_diff = abs_diff / (g + g_cp).abs().mean()
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
                + f"  {g_cp=}"
            )
    if param_substrs_to_ignore:
        ignored_params_msg = (
            ["\n****** IGNORED PARAMS ********"]
            + param_substrs_to_ignore
            + ["***************\n"]
        )
        warnings.warn("\n".join(ignored_params_msg), stacklevel=1)

    if print_passes:
        pass_msg = []
        pass_msg.append("\n****** PASSES ********")
        for n, msg in passes.items():
            pass_msg.append(f"{n}: {msg}")
        pass_msg.append("***************\n")
        print("\n".join(pass_msg), flush=True)

    if fails:
        err_msg = []
        err_msg.append("\n****** FAILURES ********")
        for n, msg in fails.items():
            err_msg.append(f"{n}: {msg}")
        err_msg.append("***************\n")
        raise RuntimeError("\n".join(err_msg))


class TestCausalPassingFn(DTest):
    def test_fwd(self):
        with torch.no_grad():
            torch.manual_seed(42)
            dim = 256
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
        dim = 256
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


class TestSeqAndZigZagFns(DTest):
    seq_dim = 1
    dtype = torch.bfloat16

    def test_seq_to_zigzag_fwd(self):
        with torch.no_grad():
            # Send the mini shard idx tensors around
            t_send = torch.tensor(
                [[2 * self.rank, 2 * self.rank + 1]],
                device=self.device,
                dtype=self.dtype,
            )
            mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            t_recv = seq_to_zigzag_comms(t_send, mesh, self.seq_dim)
            t_expected = torch.tensor(
                [[self.rank, 2 * self.world_size - self.rank - 1]],
                device=self.device,
                dtype=self.dtype,
            )
            torch.testing.assert_close(
                t_recv,
                t_expected,
            )

    def test_seq_to_zigzag_bwd(self):
        t_send = torch.tensor(
            [[2 * self.rank, 2 * self.rank + 1]],
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = seq_to_zigzag_comms(t_send, mesh, self.seq_dim)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad
        torch.testing.assert_close(
            grad,
            t_send,
        )

    def test_zigzag_to_seq_fwd(self):
        with torch.no_grad():
            # Send the mini shard idx tensors around
            t_send = torch.tensor(
                [[self.rank, 2 * self.world_size - self.rank - 1]],
                device=self.device,
                dtype=self.dtype,
            )
            mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            t_recv = zigzag_to_seq_comms(t_send, mesh, self.seq_dim)
            t_expected = torch.tensor(
                [[2 * self.rank, 2 * self.rank + 1]],
                device=self.device,
                dtype=self.dtype,
            )
            torch.testing.assert_close(
                t_recv,
                t_expected,
            )

    def test_zigzag_to_seq_bwd(self):
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        t_recv = seq_to_zigzag_comms(t_send, mesh, self.seq_dim)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad
        torch.testing.assert_close(
            grad,
            t_send,
        )

    def test_seq_then_zigzag_identity_function(self) -> None:
        """
        zigzag_to_seq_comms and seq_to_zigzag_comms should be inverses of each other.
        """
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=self.dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        # Send and send again
        t_recv = seq_to_zigzag_comms(t_send, mesh, self.seq_dim)
        t_recv = zigzag_to_seq_comms(t_recv, mesh, self.seq_dim)
        torch.testing.assert_close(
            t_recv,
            t_send,
        )

    def test_zigzag_then_seq_identity_function(self) -> None:
        """
        zigzag_to_seq_comms and seq_to_zigzag_comms should be inverses of each other.
        """
        # Send the mini shard idx tensors around
        t_send = torch.tensor(
            [[self.rank, 2 * self.world_size - self.rank - 1]],
            device=self.device,
            dtype=self.dtype,
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        # Send and send again
        t_recv = zigzag_to_seq_comms(t_send, mesh, self.seq_dim)
        t_recv = seq_to_zigzag_comms(t_recv, mesh, self.seq_dim)
        torch.testing.assert_close(
            t_recv,
            t_send,
        )


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
        # The D param doesn't respect the dtype constructor arg, so need an extra `to` call for
        # uniform dtype, as required by FSDP.
        return Mamba2(
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            **self.factory_kwargs,
        ).to(self.dtype)

    def get_mamba2_cp(
        self,
        cp_mesh: dist.device_mesh.DeviceMesh,
        seed: int = 42,
        cp_mamba_impl: str = "allgather",
        cp_mamba_recompute: bool = False,
    ) -> Mamba2:
        torch.manual_seed(seed)
        # The D param doesn't respect the dtype constructor arg, so need an extra `to` call for
        # uniform dtype, as required by FSDP.
        return Mamba2CP(
            cp_mesh=cp_mesh,
            d_model=self.d_model,
            d_state=self.d_state,
            chunk_size=self.chunk_size,
            cp_mamba_impl=cp_mamba_impl,
            cp_mamba_recompute=cp_mamba_recompute,
            **self.factory_kwargs,
        ).to(self.dtype)

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

    def get_model(self, seed: int = 42) -> MambaLMHeadModel:
        torch.manual_seed(seed)
        # The D param doesn't respect the dtype constructor arg, so need an extra `to` call for
        # uniform dtype, as required by FSDP.
        return MambaLMHeadModel(
            config=self.cfg, device=self.device, dtype=self.dtype
        ).to(self.dtype)

    def get_model_cp(
        self,
        cp_mesh: dist.device_mesh.DeviceMesh,
        cp_mamba_impl: str,
        cp_attn_impl: str,
        cp_mamba_recompute: bool = False,
        seed: int = 42,
    ) -> MambaLMHeadModel:
        torch.manual_seed(seed)
        # The D param doesn't respect the dtype constructor arg, so need an extra `to` call for
        # uniform dtype, as required by FSDP.
        return MambaLMHeadModel(
            config=self.cfg,
            cp_mesh=cp_mesh,
            cp_mamba_impl=cp_mamba_impl,
            cp_attn_impl=cp_attn_impl,
            cp_mamba_recompute=cp_mamba_recompute,
            device=self.device,
            dtype=self.dtype,
        ).to(self.dtype)

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
        with torch.no_grad():
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
        outputs_cp = conv_cp(xBC_cp, mamba2_cp, cp_mesh)
        torch.testing.assert_close(
            outputs_cp,
            outputs.tensor_split(self.world_size, dim=1)[self.rank],
        )
        outputs.sum().backward()
        outputs_cp.sum().backward()

        xBC_grad_shard = self.get_cp_shard(xBC.grad)
        xBC_cp_grad_shard = self.get_cp_shard(xBC_copy.grad)
        torch.testing.assert_close(
            xBC_cp_grad_shard, xBC_grad_shard, atol=self.tol, rtol=self.tol
        )


class TestScanCP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_fwd(self, cp_mamba_impl):
        with torch.no_grad():
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
                outputs_cp_all_gathered, outputs, atol=self.tol, rtol=self.tol
            )

    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
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
            inputs_cp_grad_shard, inputs_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp, all_reduce=True)


class TestMamba2CP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_fwd(self, cp_mamba_impl: str):
        with torch.no_grad():
            torch.manual_seed(42)
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            mamba2 = self.get_mamba2()
            mamba2_cp = self.get_mamba2_cp(
                cp_mesh=cp_mesh,
                cp_mamba_impl=cp_mamba_impl,
            )

            inputs = self.get_inputs()
            inputs_cp = self.get_cp_shard(inputs)

            outputs = mamba2(inputs)
            outputs_cp = mamba2_cp(inputs_cp)

            outputs_shard = self.get_cp_shard(outputs)
            torch.testing.assert_close(
                outputs_cp, outputs_shard, atol=self.tol, rtol=self.tol
            )

    @pytest.mark.parametrize("cp_mamba_recompute", (True, False))
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_bwd(self, cp_mamba_impl: str, cp_mamba_recompute: bool):
        torch.manual_seed(42)
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        mamba2 = self.get_mamba2()
        mamba2_cp = self.get_mamba2_cp(
            cp_mesh=cp_mesh,
            cp_mamba_impl=cp_mamba_impl,
            cp_mamba_recompute=cp_mamba_recompute,
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
        torch.testing.assert_close(
            inputs_cp_grad_shard, inputs_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mamba2, mamba2_cp, all_reduce=True)


class TestMHACP(_DTestModelBase):
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_attn_impl):
        with torch.no_grad():
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
            inputs_cp_grad_shard, inputs_grad_shard, atol=self.tol, rtol=self.tol
        )

        # Parameter grads should match after all-reducing.
        _test_model_model_cp_grads_close(mha, mha_cp, all_reduce=True)


class TestFSDP1MambaCP(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_fwd(self, cp_mamba_impl: str):
        with torch.no_grad():
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
            model_cp = nn.Sequential(
                *[
                    self.get_mamba2_cp(
                        cp_mesh=cp_mesh,
                        cp_mamba_impl=cp_mamba_impl,
                    )
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
            torch.testing.assert_close(
                outputs_cp_fsdp, outputs_shard, atol=self.tol, rtol=self.tol
            )

    @pytest.mark.parametrize("cp_mamba_recompute", (True, False))
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_bwd(self, cp_mamba_impl: str, cp_mamba_recompute: bool):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(
                    cp_mesh=cp_mesh,
                    cp_mamba_impl=cp_mamba_impl,
                    cp_mamba_recompute=cp_mamba_recompute,
                )
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

        # NOTE: for grads to match the non-FSDP case, it's required that the loss be an average
        # over the entire batch.

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
                mean_loss_cp_fsdp, loss, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm_cp_fsdp, grad_norm, atol=self.tol, rtol=self.tol
        )

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestFSDP1MHACP(_DTestModelBase):
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_attn_impl):
        with torch.no_grad():
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            model = nn.Sequential(*[self.get_mha() for _ in range(3)])
            model_cp = nn.Sequential(
                *[
                    self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
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
            torch.testing.assert_close(
                outputs_cp_fsdp,
                outputs_shard,
                atol=self.tol,
                rtol=self.tol,
            )

    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_bwd(self, cp_attn_impl):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
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

        # NOTE: for grads to match the non-FSDP case, it's required that the loss be an average
        # over the entire batch.

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
                mean_loss_cp_fsdp, loss, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm_cp_fsdp, grad_norm, atol=self.tol, rtol=self.tol
        )

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestHSDP1MambaCP(_DTestModelBase):
    """
    Test HSDP + CP for MambaCP where the HSDP and CP dims cover the same subgroups
    """

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_fwd(self, cp_mamba_impl: str):
        with torch.no_grad():
            mesh = dist.device_mesh.init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
            )
            cp_mesh = mesh["intra_node"]
            model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
            model_cp = nn.Sequential(
                *[
                    self.get_mamba2_cp(
                        cp_mesh=cp_mesh,
                        cp_mamba_impl=cp_mamba_impl,
                    )
                    for _ in range(3)
                ]
            )
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
    @pytest.mark.parametrize("cp_mamba_recompute", (True, False))
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    def test_bwd(self, cp_mamba_impl: str, cp_mamba_recompute: bool):
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
        )
        cp_mesh = mesh["intra_node"]
        model = nn.Sequential(*[self.get_mamba2() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mamba2_cp(
                    cp_mesh=cp_mesh,
                    cp_mamba_impl=cp_mamba_impl,
                    cp_mamba_recompute=cp_mamba_recompute,
                )
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([Mamba2CP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        # NOTE: for grads to match the non-FSDP case, it's required that the loss be an average
        # over the entire batch.

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
                mean_loss_cp_hsdp, loss, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_hsdp = model_cp_hsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm_cp_hsdp, grad_norm, atol=self.tol, rtol=self.tol
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
        with torch.no_grad():
            mesh = dist.device_mesh.init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("inter_node", "intra_node")
            )
            cp_mesh = mesh["intra_node"]
            model = nn.Sequential(*[self.get_mha() for _ in range(3)])
            model_cp = nn.Sequential(
                *[
                    self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                    for _ in range(3)
                ]
            )
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
        model = nn.Sequential(*[self.get_mha() for _ in range(3)])
        model_cp = nn.Sequential(
            *[
                self.get_mha_cp(cp_mesh=cp_mesh, cp_attn_impl=cp_attn_impl)
                for _ in range(3)
            ]
        )
        model_cp_hsdp = FSDP(
            model_cp,
            device_mesh=mesh,
            auto_wrap_policy=ModuleWrapPolicy([MHACP]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        # NOTE: for grads to match the non-FSDP case, it's required that the loss be an average
        # over the entire batch.

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
                mean_loss_cp_hsdp, loss, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_hsdp = model_cp_hsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm_cp_hsdp, grad_norm, atol=self.tol, rtol=self.tol
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


class TestModelCPFSDP1(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_mamba_impl, cp_attn_impl):
        with torch.no_grad():
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            model = self.get_model()
            model_cp_fsdp = self.get_model_cp(
                cp_mesh,
                cp_mamba_impl=cp_mamba_impl,
                cp_attn_impl=cp_attn_impl,
            )
            model_cp_fsdp = FSDP(
                model_cp_fsdp,
                process_group=cp_mesh.get_group(),
                auto_wrap_policy=ModuleWrapPolicy([Block]),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
                device_id=self.device,
            )

            inputs = self.get_input_toks()
            inputs_cp = self.get_cp_shard(inputs)

            outputs = model(inputs).logits
            outputs_cp = model_cp_fsdp(inputs_cp).logits

            outputs_shard = self.get_cp_shard(outputs)
            torch.testing.assert_close(
                outputs_cp,
                outputs_shard,
                atol=self.tol,
                rtol=self.tol,
            )

    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    @pytest.mark.parametrize("cp_mamba_recompute", (True, False))
    def test_bwd(self, cp_mamba_impl, cp_attn_impl, cp_mamba_recompute):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp_fsdp = self.get_model_cp(
            cp_mesh,
            cp_mamba_impl=cp_mamba_impl,
            cp_attn_impl=cp_attn_impl,
            cp_mamba_recompute=cp_mamba_recompute,
        )
        model_cp_fsdp = FSDP(
            model_cp_fsdp,
            process_group=cp_mesh.get_group(),
            auto_wrap_policy=ModuleWrapPolicy([Block]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=self.device,
        )

        inputs = self.get_input_toks()
        inputs_cp_fsdp = self.get_cp_shard(deepcopy(inputs))

        outputs = model(inputs).logits
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1).long()
        )
        loss.backward()

        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp).logits
        loss_cp_fsdp = F.cross_entropy(
            outputs_cp_fsdp.reshape(-1, outputs_cp_fsdp.size(-1)),
            inputs_cp_fsdp.reshape(-1).long(),
        )
        loss_cp_fsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_fsdp)
            mean_loss_cp_fsdp = loss_cp_fsdp / self.world_size
            torch.testing.assert_close(
                mean_loss_cp_fsdp, loss, atol=self.tol, rtol=self.tol
            )

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        torch.testing.assert_close(
            grad_norm_cp_fsdp, grad_norm, atol=self.tol, rtol=self.tol
        )

        with FSDP.summon_full_params(model_cp_fsdp, with_grads=True):
            dist.barrier()
            _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)


class TestModelCPFSDP2(_DTestModelBase):
    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    def test_fwd(self, cp_mamba_impl: str, cp_attn_impl: str):
        with torch.no_grad():
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            model = self.get_model()
            model_cp_fsdp = self.get_model_cp(
                cp_mesh,
                cp_mamba_impl=cp_mamba_impl,
                cp_attn_impl=cp_attn_impl,
            )

            for module in model_cp_fsdp.modules():
                if isinstance(module, Block):
                    fully_shard(module, mesh=cp_mesh)
            fully_shard(model_cp_fsdp, mesh=cp_mesh)

            inputs = self.get_input_toks()
            inputs_cp = self.get_cp_shard(inputs)

            outputs = model(inputs).logits
            outputs_cp = model_cp_fsdp(inputs_cp).logits

            outputs_shard = self.get_cp_shard(outputs)
            torch.testing.assert_close(
                outputs_cp,
                outputs_shard,
                atol=self.tol,
                rtol=self.tol,
            )

    @pytest.mark.parametrize("cp_mamba_impl", list(CP_MAMBA_IMPLS))
    @pytest.mark.parametrize("cp_attn_impl", ("ring", "zigzag"))
    @pytest.mark.parametrize("cp_mamba_recompute", (True, False))
    def test_bwd(self, cp_mamba_impl, cp_attn_impl, cp_mamba_recompute):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = self.get_model()
        model_cp_fsdp = self.get_model_cp(
            cp_mesh,
            cp_mamba_impl=cp_mamba_impl,
            cp_attn_impl=cp_attn_impl,
            cp_mamba_recompute=cp_mamba_recompute,
        )

        for module in model_cp_fsdp.modules():
            if isinstance(module, Block):
                fully_shard(module, mesh=cp_mesh)
        fully_shard(model_cp_fsdp, mesh=cp_mesh)

        inputs = self.get_input_toks()
        inputs_cp_fsdp = self.get_cp_shard(deepcopy(inputs))

        outputs = model(inputs).logits
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1).long()
        )
        loss.backward()

        outputs_cp_fsdp = model_cp_fsdp(inputs_cp_fsdp).logits
        loss_cp_fsdp = F.cross_entropy(
            outputs_cp_fsdp.reshape(-1, outputs_cp_fsdp.size(-1)),
            inputs_cp_fsdp.reshape(-1).long(),
        )
        loss_cp_fsdp.backward()

        # Check the losses and grad norms are the same.
        with torch.no_grad():
            dist.all_reduce(loss_cp_fsdp)
            mean_loss_cp_fsdp = loss_cp_fsdp / self.world_size
            torch.testing.assert_close(
                mean_loss_cp_fsdp, loss, atol=self.tol, rtol=self.tol
            )

        # TODO: @goon - FSDP2 grad clipping is more involved; see torch-titan.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # grad_norm_cp_fsdp = model_cp_fsdp.clip_grad_norm_(1.0)
        # torch.testing.assert_close(
        #     grad_norm, grad_norm_cp_fsdp, atol=self.tol, rtol=self.tol
        # )

        dist.barrier()
        _test_model_model_cp_grads_close(model, model_cp_fsdp, all_reduce=False)
