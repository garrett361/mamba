import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from mamba_ssm.modules.mamba2_cp import Mamba2CP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cp_mamba_impl", type=str, default="allgather")
    parser.add_argument("--d_model", type=int, default=4096)  # bamba 9.8b default
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seq_len_per_gpu", type=int, default=8192)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
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

        with torch.no_grad():
            for _ in range(args.warmups):
                mamba_stack(inputs)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(args.iters):
                mamba_stack(inputs)
        dist.barrier()
        stop.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(stop) / 1e3

        toks_per_gpu = args.batch_size * args.seq_len_per_gpu * args.iters
        total_toks = toks_per_gpu * world_size
        toks_per_sec = total_toks / secs
        toks_per_sec_per_gpu = toks_per_sec / world_size
        if not rank:
            print(f"Total tokens: {total_toks}")
            print(f"Total Secs: {secs}")
            print(f"Tok/sec {toks_per_sec}")
            print(f"Tok/sec/gpu {toks_per_sec_per_gpu}")

            if args.wandb:
                import wandb

                config = vars(args)
                config["seq_len"] = args.seq_len_per_gpu * world_size
                wandb.init(project=args.project, id=args.run_id, config=config)
                wandb.log(
                    {
                        "tok_sec_gpu": toks_per_sec_per_gpu,
                        "sec": secs,
                    },
                    step=1,
                )

    finally:
        dist.destroy_process_group()
