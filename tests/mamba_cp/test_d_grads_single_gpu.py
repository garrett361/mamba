import torch
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

"""
torchrun --nproc-per-node 4 <path-to-this-file>
"""

if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    device = torch.device("cuda")
    torch.cuda.set_device(0)

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
    grad_old = None
    try:
        for i in range(100):
            outputs = model(inputs).logits
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1).long()
            )
            loss.backward()
            grad_new = model.backbone.layers[0].mixer.D.grad.clone()
            if grad_old is not None:
                torch.testing.assert_close(grad_new, grad_old)
            model.zero_grad()
            torch.cuda.synchronize()
            print(f"PASSED on iteration {i}")
            grad_old = grad_new
        print(f"{grad_new=}")

    except:
        print(f"FAILED on iteration {i}\n{grad_old=}\n{grad_new=}")
        raise
