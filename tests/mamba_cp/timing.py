import argparse
import datetime
import os
from functools import partial
from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import Mamba2CP

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--mamba_only", action="store_true")
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--no_ac", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=bamba_9dot8b_defaults["n_layer"])
    args = parser.parse_args()

    cli_args = {"n_layer": args.n_layer}
    if args.mamba_only:
        cli_args["attn_layer_idx"] = []
    config = MambaConfig(**{**bamba_9dot8b_defaults, **cli_args})

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not rank:
        print(f"{args=}")
        print(f"{config=}")

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=30), device_id=device
    )
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

    model = MambaLMHeadModel(config=config, cp_mesh=mesh if args.cp else None)
    # I don't see why HYBRID_SHARD wouldn't be fine itself, but I'm OOM-ing with HYBRID_SHARD on
    # 8 GPUs.
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy([Mamba2, Mamba2CP]),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD
        if args.hsdp
        else ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        ),
    )
    if not args.no_ac:
        apply_fsdp_checkpointing(model)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-7, foreach=False)

    inputs = torch.randint(
        config.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
    )

    def train_one_step():
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.shape[-1]), inputs.view(-1)
        )
        loss.backward()
        optim.step()
        optim.zero_grad()

    for _ in range(args.warmups):
        train_one_step()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        train_one_step()
    dist.barrier()
    stop.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(stop) / 1e3

    total_toks = args.batch_size * args.seq_len * world_size * args.iters
    toks_per_sec = total_toks / secs
    toks_per_sec_per_gpu = toks_per_sec / world_size
    if not rank:
        print(f"Total tokens: {total_toks}")
        print(f"Total Secs: {secs}")
        print(f"Tok/sec {toks_per_sec}")
        print(f"Tok/sec/gpu {toks_per_sec_per_gpu}")

        print(f"{get_mem_dict()}")

    dist.destroy_process_group()
