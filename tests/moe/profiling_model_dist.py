import json
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from timing_utils import get_ep_mesh
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.profiler import ProfilerActivity, profile, record_function

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import (
    MambaLMHeadModel,
    act_ckpt_moe,
    fully_shard_moe,
    init_meta_moe,
)
from mamba_ssm.modules.moe import EP_EXPERT_CLASSES


def trace_handler(prof):
    global impl
    global args
    global model
    global rank
    global world_size
    global ep_degree

    if not args.all_ranks and rank:
        return

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
            header=f"{args.sharding_strategy}, {ep_degree=}, {impl=}",
        )
    )
    trace_dir = Path(args.trace_dir)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    subdir = trace_dir.joinpath(
        f"world_{world_size}_ep_{ep_degree}_{args.sharding_strategy}/{timestamp}/"
    )
    subdir.mkdir(parents=True, exist_ok=True)
    filename = f"trace_{impl}_step_{prof.step_num}_rank_{rank}_bsz_{args.bsz}_seq_{args.seqlen}"
    if args.act_ckpt:
        filename += "_act_ckpt"
    filename += ".json"

    prof.export_chrome_trace(str(subdir.joinpath(f"{filename}")))
    if not rank:
        with open(subdir.joinpath(f"args_{impl}.json"), "w") as fp:
            json.dump(vars(args), fp)
        short_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with open(subdir.joinpath(short_hash), "w") as _:
            pass


if __name__ == "__main__":
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # some setups
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        torch.cuda.empty_cache()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        parser = ArgumentParser()
        parser.add_argument(
            "--sharding_strategy", choices=["fsdp", "hsdp"], default="fsdp"
        )
        parser.add_argument("--trace_dir", default="/gpfs/goon/prof/mamba/model")
        parser.add_argument("--act_ckpt", action="store_true")
        parser.add_argument("--act_ckpt_entire_blocks", action="store_true")
        parser.add_argument("--active", type=int, default=2)
        parser.add_argument("--bsz", type=int, default=2)
        parser.add_argument("--d_intermediate", type=int, default=1344)
        n_gpus = torch.cuda.device_count()
        parser.add_argument(
            "--ep_degrees", type=str, default=f"1,{n_gpus // 2},{n_gpus}"
        )
        parser.add_argument("--impls", type=str, default=",".join(EP_EXPERT_CLASSES))
        parser.add_argument("--in_features", type=int, default=3072)
        parser.add_argument("--n_activated_experts", type=int, default=8)
        parser.add_argument("--n_layer", type=int, default=4)
        parser.add_argument("--n_routed_experts", type=int, default=64)
        parser.add_argument("--n_shared_experts", type=int, default=0)
        parser.add_argument("--no_bwd", action="store_true")
        parser.add_argument("--repeat", type=int, default=2)
        parser.add_argument("--seqlen", type=int, default=4096)
        parser.add_argument("--wait", type=int, default=3)
        parser.add_argument("--warmup", type=int, default=3)
        parser.add_argument("--all_ranks", action="store_true")
        parser.add_argument("--vocab_size", type=int, default=128256)
        parser.add_argument("--attn_layer_rate", type=int, default=4)
        parser.add_argument("--low_cpu_fsdp", action="store_true")

        args = parser.parse_args()
        if rank == 0:
            print(f"--> running with these configs {args}")

        ssm_cfg = {"layer": "Mamba2"}
        attn_layer_idx = list(
            range(args.attn_layer_rate - 1, args.n_layer, args.attn_layer_rate)
        )
        attn_cfg = {
            "causal": True,
            "d_conv": 0,
            "head_dim": 128,
            "num_heads": 24,
            "num_heads_kv": 8,
            "out_proj_bias": False,
            "qkv_proj_bias": False,
            "rotary_emb_dim": 64,
        }
        moe_layer_idx = list(range(args.n_layer))
        moe_cfg = {
            "n_routed_experts": args.n_routed_experts,
            "n_activated_experts": args.n_activated_experts,
            "n_shared_experts": args.n_shared_experts,
            "d_intermediate": args.d_intermediate,
        }

        impls = args.impls.split(",")
        ep_degrees = [int(e) for e in args.ep_degrees.split(",")]

        torch.cuda.manual_seed(42)
        if args.sharding_strategy == "hsdp":
            fsdp_mesh = init_device_mesh(
                "cuda",
                (world_size // torch.cuda.device_count(), torch.cuda.device_count()),
                mesh_dim_names=("outer", "inner"),
            )
        else:
            fsdp_mesh = init_device_mesh(
                "cuda", (world_size,), mesh_dim_names=("fsdp",)
            )

        # Different inputs on different ranks for even tok distribution
        torch.manual_seed(42 + rank)
        iters_per_impl = args.repeat * (args.active + args.wait + args.warmup)
        device = "cuda"
        factory_kwargs = {"device": device}
        inputs = torch.randint(
            args.vocab_size,
            size=(iters_per_impl, args.bsz, args.seqlen),
            device=device,
        )
        for ep_degree in ep_degrees:
            ep_mesh = get_ep_mesh(ep_degree, world_size)
            for impl in impls:
                moe_cfg["moe_impl"] = impl
                cfg = MambaConfig(
                    d_model=args.in_features,
                    d_intermediate=args.d_intermediate,
                    n_layer=args.n_layer,
                    vocab_size=args.vocab_size,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    moe_layer_idx=moe_layer_idx,
                    moe_cfg=moe_cfg,
                    tie_embeddings=False,
                )
                if not rank:
                    print(f"{cfg=}")

                # Model building order:
                # 1. Create model, maybe on meta device.
                # 2. Activation checkpointing, if applicable
                # 3. fully_shard
                # 4. init weights, if meta device was used
                if args.low_cpu_fsdp:
                    if rank == 0:
                        print("Building model on meta device...")
                    with torch.device("meta"):
                        model = MambaLMHeadModel(
                            cfg,
                            ep_mesh=None if ep_mesh is None else ep_mesh["inner"],
                        )
                else:
                    model = MambaLMHeadModel(
                        config=cfg,
                        ep_mesh=None if ep_mesh is None else ep_mesh["inner"],
                        **factory_kwargs,
                    )

                if args.act_ckpt:
                    # Just ckpt mixer layers
                    if not rank:
                        print("Applying activation checkpointing")
                    act_ckpt_moe(model, mixer_only=args.act_ckpt_entire_blocks)

                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
                )
                fully_shard_moe(
                    model=model,
                    ep_degree=ep_degree,
                    world_size=world_size,
                    fsdp_mesh=fsdp_mesh,
                    ep_mesh=ep_mesh,
                    mp_policy=mp_policy,
                )

                if args.low_cpu_fsdp:
                    if rank == 0:
                        print("Moving meta model to CUDA...")
                    init_meta_moe(model)

                if not rank:
                    print(f"{model=}")
                # Assumption: no tied params
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    # record_shapes=True,
                    # with_stack=True,
                    # profile_memory=True,
                    schedule=torch.profiler.schedule(
                        wait=args.wait,
                        warmup=args.warmup,
                        active=args.active,
                        repeat=args.repeat,
                    ),
                    on_trace_ready=trace_handler,
                ) as prof:
                    for idx in range(iters_per_impl):
                        with record_function("fwd"):
                            out = model(inputs[idx])
                        with record_function("bwd"):
                            F.cross_entropy(
                                out.logits.view(-1, out.logits.size(-1)),
                                inputs[idx].view(-1).long(),
                            ).backward()
                        model.zero_grad()
                        prof.step()
                del model

    finally:
        dist.destroy_process_group()
