import argparse
import csv
import datetime
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_cp import Mamba2CP


class BarrierEveryLayerStack(nn.Module):
    def __init__(self, layers: list[nn.Module], world_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if self.world_size > 1:
                dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cp_mamba_impl", type=str, default="allgather")
    parser.add_argument("--cp_mamba_recompute", action="store_true")
    parser.add_argument("--d_model", type=int, default=4096)  # bamba 9.8b default
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seq_len_per_gpu", type=int, default=65536)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_bwd", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--no_csv", action="store_true", default=False)
    parser.add_argument("--barrier_every_layer", action="store_true", default=False)
    args = parser.parse_args()
    if args.barrier_every_layer and "serial" not in args.cp_mamba_impl:
        raise ValueError("barrier_every_layer only relevant for serial impls")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not rank:
        print(f"{args=}")

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=60), device_id=device
        )
        mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

        mamba_cls = Mamba2CP if world_size > 1 else Mamba2
        mamba_kwargs = dict(
            d_model=args.d_model,
            device=device,
            dtype=dtype,
        )
        if world_size > 1:
            mamba_kwargs["cp_mesh"] = mesh
            mamba_kwargs["cp_mamba_impl"] = args.cp_mamba_impl
            mamba_kwargs["cp_mamba_recompute"] = args.cp_mamba_recompute

        if args.barrier_every_layer:
            mamba_stack = BarrierEveryLayerStack(
                [mamba_cls(**mamba_kwargs) for _ in range(args.n_layers)], world_size
            )
        else:
            mamba_stack = nn.Sequential(
                *[mamba_cls(**mamba_kwargs) for _ in range(args.n_layers)]
            )

        inputs = torch.randn(
            args.batch_size,
            args.seq_len_per_gpu,
            args.d_model,
            device=device,
            dtype=dtype,
        )

        # Initial barrier to avoid possible issues w/ P2P comms being the first ops.
        # https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.batch_isend_irecv
        dist.barrier()
        ctx = torch.no_grad if args.no_bwd else nullcontext
        with ctx():
            for _ in range(args.warmups):
                outputs = mamba_stack(inputs)
                if not args.no_bwd:
                    outputs.sum().backward()
                    mamba_stack.zero_grad()
                del outputs
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        with ctx():
            for _ in range(args.iters):
                outputs = mamba_stack(inputs)
                if not args.no_bwd:
                    outputs.sum().backward()
                    mamba_stack.zero_grad()
                del outputs
                dist.barrier()
        stop.record()
        torch.cuda.synchronize()

        secs = start.elapsed_time(stop) / 1e3

        toks_per_gpu = args.batch_size * args.seq_len_per_gpu * args.iters
        total_toks = toks_per_gpu * world_size
        toks_per_sec = total_toks / secs
        toks_per_sec_per_gpu = toks_per_sec / world_size
        if not rank:
            reserved_mem = (
                torch.cuda.max_memory_reserved(device=torch.cuda.current_device())
                / 2**30
            )
            allocated_mem = (
                torch.cuda.max_memory_allocated(device=torch.cuda.current_device())
                / 2**30
            )

            # Write results to CSV
            if not args.no_csv:
                csv_filename = "mamba_cp_layer_timing.csv"
                file_exists = os.path.isfile(csv_filename)

                with open(csv_filename, "a", newline="") as csvfile:
                    fieldnames = [
                        "batch_size",
                        "cp_mamba_impl",
                        "cp_mamba_recompute",
                        "d_model",
                        "iters",
                        "n_layers",
                        "seq_len_per_gpu",
                        "world_size",
                        "seq_len",
                        "no_bwd",
                        "warmups",
                        "total_toks",
                        "secs",
                        "toks_per_sec",
                        "toks_per_sec_per_gpu",
                        "reserved_mem_gib",
                        "allocated_mem_gib",
                        "barrier_every_layer",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    if not file_exists:
                        writer.writeheader()

                    writer.writerow(
                        {
                            "batch_size": args.batch_size,
                            "cp_mamba_impl": args.cp_mamba_impl,
                            "cp_mamba_recompute": args.cp_mamba_recompute,
                            "d_model": args.d_model,
                            "iters": args.iters,
                            "n_layers": args.n_layers,
                            "seq_len_per_gpu": args.seq_len_per_gpu,
                            "world_size": world_size,
                            "seq_len": args.seq_len_per_gpu * world_size,
                            "no_bwd": args.no_bwd,
                            "warmups": args.warmups,
                            "total_toks": total_toks,
                            "secs": secs,
                            "toks_per_sec": toks_per_sec,
                            "toks_per_sec_per_gpu": toks_per_sec_per_gpu,
                            "reserved_mem_gib": reserved_mem,
                            "allocated_mem_gib": allocated_mem,
                            "barrier_every_layer": args.barrier_every_layer,
                        }
                    )

            print(f"Total tokens: {total_toks}")
            print(f"Total Secs: {secs}")
            print(f"Tok/sec {toks_per_sec}")
            print(f"Tok/sec/gpu {toks_per_sec_per_gpu}")
            print(f"Reserved GiB {reserved_mem}")
            print(f"Allocated GiB {allocated_mem}")

            if args.wandb:
                import wandb

                config = vars(args)
                config["seq_len"] = args.seq_len_per_gpu * world_size
                wandb.init(project=args.project, id=args.run_id, config=config)
                wandb.log(
                    {
                        "tok_sec_gpu": toks_per_sec_per_gpu,
                        "sec": secs,
                        "reserved GiB": reserved_mem,
                        "allocated GiB": allocated_mem,
                    },
                    step=1,
                )

    finally:
        dist.destroy_process_group()
