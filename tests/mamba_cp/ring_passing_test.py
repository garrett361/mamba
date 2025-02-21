import datetime
import os
from functools import partial
from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from mamba_ssm.modules.block import Block

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def get_mem_dict():
    mem_dict = {}
    for k, v in torch.cuda.memory_stats().items():
        if all(s in k for s in (".all.", "bytes")) and any(
            s in k for s in ("current", "peak")
        ):
            mem_dict[k.replace("bytes", "gib")] = v / 2**30

    return mem_dict


def apply_fsdp_checkpointing(model, block: Type[nn.Module] = Block, p=1):
    """
    Apply selective activation checkpointing.

    Selectivity is defined as a percentage p, which means we apply ac
    on p of the total blocks. p is a floating number in the range of
    [0, 1].

    Some examples:
    p = 0: no ac for all blocks. same as `fsdp_activation_checkpointing=False`
    p = 1: apply ac on every block. i.e. "full ac".
    p = 1/2: [ac, no-ac, ac, no-ac, ...]
    p = 1/3: [no-ac, ac, no-ac,   no-ac, ac, no-ac,   ...]
    p = 2/3: [ac, no-ac, ac,    ac, no-ac, ac,    ...]
    Since blocks are homogeneous, we make ac blocks evenly spaced among
    all blocks.

    Implementation:
    For a given ac ratio p, we should essentially apply ac on every "1/p"
    blocks. The first ac block can be as early as the 0th block, or as
    late as the "1/p"th block, and we pick the middle one: (0.5p)th block.
    Therefore, we are essentially to apply ac on:
    (0.5/p)th block, (1.5/p)th block, (2.5/p)th block, etc., and of course,
    with these values rounding to integers.
    Since ac is applied recursively, we can simply use the following math
    in the code to apply ac on corresponding blocks.
    """
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, block):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


bamba_9dot8b_defaults = {
    "d_model": 4096,
    "d_intermediate": 14336,
    "n_layer": 32,
    "vocab_size": 128256,
    "ssm_cfg": {"layer": "Mamba2"},
    "attn_layer_idx": [9, 18, 27],
    "attn_cfg": {
        "causal": True,
        "d_conv": 0,
        "head_dim": 128,
        "num_heads": 32,
        "num_heads_kv": 8,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": 64,
    },
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": False,
}


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=30),
    )
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
    final_states = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
    recv_init_states = torch.empty_like(final_states)
    if not rank:
        print(f"{mesh=}")
    for send_rank, recv_rank in zip(mesh.mesh[:-1], mesh.mesh[1:]):
        if not rank:
            print(f"{send_rank=}, {recv_rank=}")
        print(f"{mesh=}")
        if rank == send_rank:
            dist.send(
                final_states.contiguous(),
                dst=recv_rank,
                group=mesh.get_group(),
            )
        elif rank == recv_rank:
            dist.recv(
                recv_init_states,
                src=send_rank,
                group=mesh.get_group(),
            )
        dist.barrier()

    print(f"[{rank=}]: {final_states=}")

    dist.destroy_process_group()
