from abc import abstractmethod, ABC
import pytest
import torch

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 16


class _TestMambaBase(ABC):
    seq_len = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_cfg(self) -> MambaConfig: ...

    def setup_method(self, method):
        """
        Reset dynamo before every test.
        """
        torch._dynamo.reset()

    def teardown_method(self, method):
        pass

    def test_cfg(self) -> None:
        cfg = self.get_cfg()
        assert cfg is not None

    # TODO: @goon - Better test parametrization. Mostly left as independent methods for exploration
    # for now.
    @pytest.mark.parametrize(
        "mode",
        ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    def test_compile(self, mode) -> None:
        cfg = self.get_cfg()
        mamba = MambaLMHeadModel(cfg, device=self.device)
        compiled_mamba = torch.compile(mamba, mode=mode)
        inputs = torch.randint(
            cfg.vocab_size, size=(1, self.seq_len), device=self.device
        )
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, self.seq_len, cfg.vocab_size))

    def test_compile_full_graph(self) -> None:
        cfg = self.get_cfg()
        mamba = MambaLMHeadModel(cfg, device=self.device)
        compiled_mamba = torch.compile(mamba, fullgraph=True)
        inputs = torch.randint(
            cfg.vocab_size, size=(1, self.seq_len), device=self.device
        )
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, self.seq_len, cfg.vocab_size))

    def test_compile_dynamic(self) -> None:
        cfg = self.get_cfg()
        mamba = MambaLMHeadModel(cfg, device=self.device)
        compiled_mamba = torch.compile(mamba, dynamic=True)
        inputs = torch.randint(
            cfg.vocab_size, size=(1, self.seq_len), device=self.device
        )
        outputs = compiled_mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, self.seq_len, cfg.vocab_size))


class TestMamba(_TestMambaBase):
    def get_cfg(self) -> MambaConfig:
        return MambaConfig(
            d_model=128, n_layer=2, vocab_size=512, ssm_cfg={"layer": "Mamba1"}
        )

    def test_model(self) -> None:
        """
        Sanity check. TODO: @goon - remove.
        """
        cfg = self.get_cfg()
        mamba = MambaLMHeadModel(cfg, device=self.device)
        for block in mamba.backbone.layers:
            assert isinstance(block.mixer, Mamba)
        inputs = torch.randint(
            cfg.vocab_size, size=(1, self.seq_len), device=self.device
        )
        outputs = mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, self.seq_len, cfg.vocab_size))


class TestMamba2(_TestMambaBase):
    def get_cfg(self) -> MambaConfig:
        # Important to have d_model sufficiently large to avoid this error:
        # RuntimeError: causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8
        # https://github.com/state-spaces/mamba/issues/345#issuecomment-2145035818
        return MambaConfig(
            d_model=512, n_layer=2, vocab_size=512, ssm_cfg={"layer": "Mamba2"}
        )

    def test_model(self) -> None:
        """
        Sanity check. TODO: @goon - remove.
        """
        cfg = self.get_cfg()
        mamba = MambaLMHeadModel(cfg, device=self.device)
        for block in mamba.backbone.layers:
            assert isinstance(block.mixer, Mamba2)
        inputs = torch.randint(
            cfg.vocab_size, size=(1, self.seq_len), device=self.device
        )
        outputs = mamba(inputs)
        assert outputs.logits.shape == torch.Size((1, self.seq_len, cfg.vocab_size))
