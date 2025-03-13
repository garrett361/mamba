import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

"""
torchrun --nproc-per-node 4 <path-to-this-file>
"""

if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        rank = int(os.environ["RANK"])
        batch_size = world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        dist.init_process_group(backend="nccl")

        d_model = 256
        expand = 2
        d_conv = 4
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
        seq_len = 32

        model = MambaLMHeadModel(config=cfg, device=device, dtype=dtype)
        inputs = torch.randint(vocab_size, size=(batch_size, seq_len), device=device)

        outputs = model(inputs).logits
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1).long()
        )
        loss.backward()

        # For some reason, different grads get populated on the different D's on different ranks?
        for n, p in model.named_parameters():
            if n.endswith(".D"):
                D_grad = p.grad

        all_losses = (
            [torch.empty_like(loss) for _ in range(world_size)] if not rank else None
        )
        dist.gather(loss, all_losses)
        all_inputs = (
            [torch.empty_like(inputs) for _ in range(world_size)] if not rank else None
        )
        dist.gather(inputs, all_inputs)
        all_D_grads = (
            [torch.empty_like(D_grad) for _ in range(world_size)] if not rank else None
        )
        dist.gather(D_grad, all_D_grads)

        if not rank:
            # Nice formatted printing:
            print("********** ALL LOSSES **********", flush=True)
            for rank, loss in enumerate(all_losses):
                print(f"[{rank=}]: {loss=}", flush=True)
            print("********** ALL INPUTS **********", flush=True)
            for rank, inputs in enumerate(all_inputs):
                print(f"[{rank=}]: {inputs=}", flush=True)
            print("********** ALL D_GRADS **********", flush=True)
            for rank, D_grad in enumerate(all_D_grads):
                print(f"[{rank=}]: {D_grad=}", flush=True)
            torch.cuda.synchronize()
            for rank in range(1, world_size):
                torch.testing.assert_close(all_losses[0], all_losses[rank])
                torch.testing.assert_close(all_inputs[0], all_inputs[rank])
                torch.testing.assert_close(all_D_grads[0], all_D_grads[rank])
    finally:
        dist.destroy_process_group()
