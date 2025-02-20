import argparse
import datetime
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import Mamba2CP

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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=bamba_9dot8b_defaults["n_layer"])
    args = parser.parse_args()

    config = MambaConfig(**{**bamba_9dot8b_defaults, **{"n_layer": args.n_layer}})

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not rank:
        print(f"{args=}")
        print(f"{config=}")

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=30),
    )
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
    model = MambaLMHeadModel(config=config, cp_mesh=mesh if args.cp else None)
    model_fsdp = FSDP(
        model,
        process_group=mesh.get_group(),
        auto_wrap_policy=ModuleWrapPolicy([Mamba2, Mamba2CP]),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        ),
    )
    if not rank:
        print(f"{model_fsdp=}")

    inputs = torch.randint(
        config.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
    )

    for _ in range(args.warmups):
        outputs = model_fsdp(inputs)
        outputs.logits.mean().backward()
        model_fsdp.zero_grad()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        outputs = model_fsdp(inputs)
        F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.shape[-1]), inputs.view(-1)
        )
        model_fsdp.zero_grad()
    dist.barrier()
    stop.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(stop) / 1e3

    total_toks = args.batch_size * args.seq_len * world_size * args.iters
    toks_per_sec = total_toks / secs
    if not rank:
        print(f"Total tokens: {total_toks}")
        print(f"Total Secs: {secs}")
        print(f"Tok/sec {toks_per_sec}")
