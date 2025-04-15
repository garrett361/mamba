import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.profiler import ProfilerActivity, profile, record_function

from mamba_ssm.modules.moe import MoE


class SequentialProfileModule(nn.Module):
    def __init__(self, modules_list) -> None:
        super().__init__()
        self.layers = nn.ModuleList(modules_list)

    def forward(self, inputs):
        outputs = inputs
        for idx, module in enumerate(self.layers):
            with record_function(f"fwd_{idx}"):
                outputs = module(outputs)
        return outputs


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
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        parser = ArgumentParser()
        parser.add_argument("--n_layer", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--seqlen", type=int, default=4096)
        parser.add_argument("--in_features", type=int, default=3072)
        parser.add_argument("--d_intermediate", type=int, default=1344)
        parser.add_argument("--n_routed_experts", type=int, default=64)
        parser.add_argument("--n_shared_experts", type=int, default=0)
        parser.add_argument("--n_activated_experts", type=int, default=8)
        parser.add_argument("--ep_degree", type=int, default=1)
        parser.add_argument("--act_ckpt", action="store_true")
        parser.add_argument("--warmups", type=int, default=3)
        parser.add_argument(
            "--sharding_strategy", choices=["fsdp", "hsdp"], default="fsdp"
        )
        parser.add_argument("--trace_dir", default=None)

        args = parser.parse_args()
        if rank == 0:
            print(f"--> running with these configs {args}")

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

        # Cases:
        # 1. ep_degree = 1: full replication, no ep_mesh
        # 2. ep_degree = world_size: ep_mesh is the world
        # 3. world_size > ep_degree > world_size: 2D mesh, experts distributed along slice .
        if args.ep_degree == 1:
            ep_mesh = None
        elif args.ep_degree == world_size:
            ep_mesh = init_device_mesh(
                "cuda",
                (world_size,),
                mesh_dim_names=("inner",),
            )
        else:
            ep_mesh = init_device_mesh(
                "cuda",
                (world_size // args.ep_degree, args.ep_degree),
                mesh_dim_names=("outer", "inner"),
            )

        model = SequentialProfileModule(
            [
                MoE(
                    in_features=args.in_features,
                    d_intermediate=args.d_intermediate,
                    n_routed_experts=args.n_routed_experts,
                    n_shared_experts=args.n_shared_experts,
                    n_activated_experts=args.n_activated_experts,
                    ep_mesh=None if ep_mesh is None else ep_mesh["inner"],
                )
                for _ in range(args.n_layer)
            ]
        )

        if args.act_ckpt:
            if not rank:
                print("Applying activation checkpointing")
            for layer_idx, moe in enumerate(model):
                model[layer_idx] = checkpoint_wrapper(moe, preserve_rng_state=False)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        for moe in model.layers:
            # Cases:
            # 1. ep_degree = 1: full replication, fully shard with the fsdp_mesh
            # 2. ep_degree = world_size: no expert replication at all. Ignore experts in fully_shard
            # 3. world_size > ep_degree > world_size: world_size // ep_degree expert replicas. Need
            #    to individually wrap experts using the ep_mesh because ModuleDict doesn't have a
            #    forward method.

            # The ignored_params arg requires torch nightly (> 2.6.0)
            ignored_params = set()
            if args.ep_degree == 1:
                pass
            elif args.ep_degree == world_size:
                # No replication in this case.
                ignored_params.add(moe.experts.parameters())
            else:
                for expert in moe.experts.values():
                    # Don't reshard due to comms costs
                    assert ep_mesh is not None
                    fully_shard(
                        expert,
                        mesh=ep_mesh["outer"],
                        mp_policy=mp_policy,
                        reshard_after_forward=False,
                    )
                    expert.set_reshard_after_backward(False)
            fully_shard(
                moe,
                mesh=fsdp_mesh,
                ignored_params=ignored_params,
                mp_policy=mp_policy,
                reshard_after_forward=True,
            )
        # The root unit doesn't own any params, so manually force the first layer to not shard after
        # bwd
        model.layers[0].set_reshard_after_backward(False)
        fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)

        if not rank:
            print(f"{model=}")

        # Different inputs on different ranks for even tok distribution
        torch.manual_seed(42 + rank)
        inputs = torch.randn(
            args.batch_size, args.seqlen, args.in_features, device="cuda"
        )
        for _ in range(args.warmups):
            model(inputs).sum().backward()
            model.zero_grad()

        dist.barrier()
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            with record_function("fwd"):
                out = model(inputs)
            with record_function("bwd"):
                out.sum().backward()
            model.zero_grad()
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10, header=f"{rank=}"
            )
        )
        if args.trace_dir is not None:
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)

            import json

            with open(trace_dir.joinpath("args.json"), "w") as fp:
                json.dump(vars(args), fp)
            prof.export_chrome_trace(str(trace_dir.joinpath(f"trace_rank_{rank}.json")))
        for layer_idx, moe in enumerate(model.layers):
            print(f"[{rank=}]: {layer_idx=} tok sum = {moe._tok_count}")
    finally:
        dist.destroy_process_group()
