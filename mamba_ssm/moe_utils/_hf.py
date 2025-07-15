import os
import re

import torch
from safetensors.torch import save_file
from transformers.models.bamba import BambaConfig
from transformers.models.granitemoehybrid import GraniteMoeHybridConfig
from transformers.utils import SAFE_WEIGHTS_NAME


def convert_ssm_config_to_hf_config(
    config_ssm: dict,
    **kwargs,
) -> GraniteMoeHybridConfig:
    """Convert a config from mamba_ssm to a BambaConfig from here."""
    hf_config: BambaConfig = BambaConfig(**kwargs)

    hf_config.architectures = ["BambaForCausalLM"]

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_intermediate"]
    hf_config.mamba_n_heads = (
        hf_config.hidden_size * hf_config.mamba_expand
    ) // hf_config.mamba_d_head
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]

    # currently this script assumes config_ssm belongs to v2
    if config_ssm["ssm_cfg"].get("layer") != "Mamba2":
        raise ValueError("Conversion script only supports Mamba2")

    # Set attention values
    attn_cfg = config_ssm.get("attn_cfg")
    if attn_cfg:
        assert attn_cfg["causal"], "Only support non-causal attention."
        assert not attn_cfg["qkv_proj_bias"], "Only support no qkv bias."
        assert not attn_cfg["out_proj_bias"], "Only support no out bias."
        hf_config.attn_rotary_emb = attn_cfg["rotary_emb_dim"]
        hf_config.num_attention_heads = attn_cfg["num_heads"]
        hf_config.num_key_value_heads = attn_cfg["num_heads_kv"]
        hf_config.rope_theta = attn_cfg["rotary_emb_base"]

    attention_layer_indices = config_ssm.get("attn_layer_idx")
    if attention_layer_indices:
        hf_config.attn_layer_indices = attention_layer_indices

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def convert_state_dict_to_mamba_ssm(model):
    original_sd = model.state_dict()
    state_dict = {}

    for orig_k in list(original_sd.keys()):
        # k = orig_k.replace("model", "backbone")
        k = orig_k.replace("embed_tokens", "embedding")
        k = k.replace("mamba", "mixer")
        k = k.replace("final_layernorm", "norm_f")
        k = re.sub(r"(\d+)\.input_layernorm\.", r"\1.norm.", k)
        k = re.sub(r"(\d+)\.pre_ff_layernorm\.", r"\1.norm2.", k)
        k = k.replace("feed_forward.down_proj", "mlp.fc2")
        k = k.replace("self_attn.o_proj", "mixer.out_proj")
        if k != orig_k:
            state_dict[k.replace("model", "backbone")] = original_sd.pop(orig_k)
    for i in range(len(model.model.layers)):
        w1 = original_sd.pop(f"model.layers.{i}.feed_forward.up_proj.weight")
        w2 = original_sd.pop(f"model.layers.{i}.feed_forward.gate_proj.weight")
        state_dict[f"backbone.layers.{i}.mlp.fc1.weight"] = torch.cat([w1, w2], dim=0)
        if f"model.layers.{i}.self_attn.q_proj.weight" in original_sd:
            q = original_sd.pop(f"model.layers.{i}.self_attn.q_proj.weight")
            k = original_sd.pop(f"model.layers.{i}.self_attn.k_proj.weight")
            v = original_sd.pop(f"model.layers.{i}.self_attn.v_proj.weight")
            state_dict[f"backbone.layers.{i}.mixer.in_proj.weight"] = torch.cat(
                [q, k, v], dim=0
            )
    state_dict["lm_head.weight"] = original_sd.pop("lm_head.weight")
    assert len(original_sd) == 0, original_sd.keys()
    return {"model_state": state_dict}


def save_single_safetensor(
    state_dict: dict,
    save_directory: str,
    metadata: dict,
):
    save_file(
        state_dict,
        os.path.join(save_directory, SAFE_WEIGHTS_NAME),
        metadata,
    )


def convert_state_dict_from_mamba_ssm(original_sd: dict) -> dict[str, torch.Tensor]:
    state_dict = {}

    for orig_k, param in original_sd.items():
        k = orig_k.replace("backbone", "model")

        # for embeddings
        k = k.replace("embedding", "embed_tokens")

        # for mixer
        k = k.replace("mixer", "mamba")

        # for final layernorm
        k = k.replace("norm_f", "final_layernorm")

        # for block layernorm
        k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
        k = re.sub(r"(\d+)\.norm2\.", r"\1.pre_ff_layernorm.", k)

        # for mlp
        k = k.replace("mlp.fc2", "feed_forward.down_proj")

        if "mlp.fc1" in k:
            param, param2 = torch.chunk(param, 2, dim=0)
            k2 = k.replace("mlp.fc1", "feed_forward.gate_proj")
            state_dict[k2] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

        if ("in_proj" in k and orig_k.replace("in_proj", "conv1d") in original_sd) or (
            "out_proj" in k and orig_k.replace("out_proj", "conv1d") in original_sd
        ):
            # then this must be a mamba
            pass
        else:
            # for attn
            # - because mixer was replaced to mamba above
            k = k.replace("mamba.out_proj", "self_attn.o_proj")
            if "mamba.in_proj" in k:
                m, n = param.shape
                d = (m - n) // 2
                param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                k2 = k.replace("mamba.in_proj", "self_attn.k_proj")
                state_dict[k2] = param2
                k2 = k.replace("mamba.in_proj", "self_attn.v_proj")
                state_dict[k2] = param3
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict
