import argparse
import datetime
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from mamba_ssm.modules.mamba2_cp import Mamba2CP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cp_mamba_impl", type=str, default="allgather")
    parser.add_argument("--cp_mamba_recompute", action="store_true")
    parser.add_argument("--d_model", type=int, default=4096)  # bamba 9.8b default
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seq_len_per_gpu", type=int, default=65536)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_bwd", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--no_barrier_after_iter", action="store_true", default=False)
    args = parser.parse_args()

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

        mamba_stack = nn.Sequential(
            *[
                Mamba2CP(
                    d_model=args.d_model,
                    cp_mesh=mesh,
                    cp_mamba_impl=args.cp_mamba_impl,
                    cp_mamba_recompute=args.cp_mamba_recompute,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(args.n_layers)
            ]
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
                # Barrier after each iteration to sync processes. Otherwise, we would amortize the
                # serial impl bubble over all iters, e.g. Mimics actual sync points expected in full
                # e2e code
                if not args.no_barrier_after_iter:
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
