from argparse import ArgumentParser

import torch
from timing_utils import CUDATimer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import NON_EP_EXPERT_CLASSES_AND_SIMPLE

moe_impls = list(NON_EP_EXPERT_CLASSES_AND_SIMPLE)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--d_intermediate", type=int, default=14336)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--d_intermediate_moe", type=int, default=1344)
    parser.add_argument("--n_routed_experts", type=int, default=64)
    parser.add_argument("--n_activated_experts", type=int, default=8)
    parser.add_argument("--n_shared_experts", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--no_bwd", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    attn_cfg = {
        "causal": True,
        "d_conv": 0,
        "head_dim": args.head_dim,
        "num_heads": 8,
        "num_heads_kv": 2,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": args.head_dim // 2,
    }
    ssm_cfg = {"layer": "Mamba2"}

    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    # Clear cache, like triton do_bench
    cache_size_bytes = 512 * 2**20  # 512k
    cache = torch.empty(int(cache_size_bytes // 4), dtype=torch.int, device="cuda")

    inputs = torch.randint(args.vocab_size, size=(args.bsz, args.seqlen), device=device)
    results = {}
    for moe_impl in moe_impls:
        moe_cfg = {
            "n_routed_experts": args.n_routed_experts,
            "n_activated_experts": args.n_activated_experts,
            "n_shared_experts": args.n_shared_experts,
            "d_intermediate": args.d_intermediate_moe,
            "moe_impl": moe_impl,
        }
        cfg = MambaConfig(
            d_model=args.in_features,
            d_intermediate=args.d_intermediate,
            n_layer=args.n_layer,
            vocab_size=args.vocab_size,
            tie_embeddings=False,
            attn_layer_idx=[args.n_layer - 1],
            attn_cfg=attn_cfg,
            moe_layer_idx=list(range(args.n_layer)),
            moe_cfg=moe_cfg,
            ssm_cfg=ssm_cfg,
        )
        model = MambaLMHeadModel(cfg, **factory_kwargs)
        print(f"{model=}")
        print(
            f"Number of parameters (B): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 2**30:.2f}"
        )

        for _ in range(args.warmups):
            out = model(inputs)
            if not args.no_bwd:
                out.logits.pow(2).sum().backward()
            cache.zero_()

        timer = CUDATimer()
        for _ in range(args.reps):
            with timer:
                out = model(inputs)
                if not args.no_bwd:
                    out.logits.pow(2).sum().backward()
            cache.zero_()
        time_s = timer.get_mean_time_s()
        time_std_s = timer.get_std_time_s()
        print(f"{moe_impl}: {time_s=:.2e}, {time_std_s=:.2e}, {time_std_s/time_s=:.2e}")
        results[moe_impl] = time_s

    print("Relative times:")
    min_time = min(results.values())
    for cls, time_s in results.items():
        print(f"\t{cls}: {time_s / min_time:.2e}")
