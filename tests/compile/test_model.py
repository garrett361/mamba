import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import pytest


def get_small_cfg() -> MambaConfig:
    return MambaConfig(d_model=128, n_layer=2, vocab_size=512)


def get_small_model(device: torch.device) -> MambaLMHeadModel:
    return


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 16


class TestMamba:
    def setup_method(self, method):
        """
        Reset dynamo before every test.
        """
        torch._dynamo.reset()

    def teardown_method(self, method):
        pass

    def test_cfg(self) -> None:
        cfg = get_small_cfg()
        assert cfg is not None

    def test_model(self) -> None:
        cfg = get_small_cfg()
        mamba = MambaLMHeadModel(cfg, device=DEVICE)
        inputs = torch.randint(cfg.vocab_size, size=(1, SEQ_LEN), device=DEVICE)
        outputs = mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, SEQ_LEN, cfg.vocab_size))

    # TODO: @goon - Better test parametrization. Mostly left as independent methods for exploration
    # for now.
    @pytest.mark.parametrize("mode", ("default", "reduce-overhead", "max-autotune"))
    def test_compile(self, mode) -> None:
        cfg = get_small_cfg()
        mamba = MambaLMHeadModel(cfg, device=DEVICE)
        compiled_mamba = torch.compile(mamba, mode=mode)
        inputs = torch.randint(cfg.vocab_size, size=(1, SEQ_LEN), device=DEVICE)
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, SEQ_LEN, cfg.vocab_size))

    def test_compile_full_graph(self) -> None:
        cfg = get_small_cfg()
        mamba = MambaLMHeadModel(cfg, device=DEVICE)
        compiled_mamba = torch.compile(mamba, fullgraph=True)
        inputs = torch.randint(cfg.vocab_size, size=(1, SEQ_LEN), device=DEVICE)
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, SEQ_LEN, cfg.vocab_size))

    def test_compile_dynamic(self) -> None:
        cfg = get_small_cfg()
        mamba = MambaLMHeadModel(cfg, device=DEVICE)
        compiled_mamba = torch.compile(mamba, dynamic=True)
        inputs = torch.randint(cfg.vocab_size, size=(1, SEQ_LEN), device=DEVICE)
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, SEQ_LEN, cfg.vocab_size))
