import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def get_small_cfg() -> MambaConfig:
    return MambaConfig(d_model=128, n_layer=2, vocab_size=512)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 16


class TestMamba:
    def test_cfg(self) -> None:
        cfg = get_small_cfg()
        assert cfg is not None

    def test_model(self) -> None:
        cfg = get_small_cfg()
        mamba = MambaLMHeadModel(cfg, device=DEVICE)
        inputs = torch.randint(cfg.vocab_size, size=(1, SEQ_LEN), device=DEVICE)
        outputs = mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, SEQ_LEN, cfg.vocab_size))
