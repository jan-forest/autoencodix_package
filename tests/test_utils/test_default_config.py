import pytest
from pydantic import ValidationError
from autoencodix.configs.default_config import DefaultConfig


class TestDefaultConfig:
    @pytest.fixture
    def default_config(self):
        return DefaultConfig()

    def test_default_config_invalid_ratios(self):

        with pytest.raises(ValidationError):
            DefaultConfig(train_ratio=0.5, valid_ratio=0.5, test_ratio=0.5)

    def test_default_config_invalid_float_precision(self):
        with pytest.raises(ValidationError):
            DefaultConfig(device="mps", float_precision="16")

    def test_default_config_invalid_gpu_strategy(self):
        with pytest.raises(ValidationError):
            DefaultConfig(device="mps", gpu_strategy="ddp")

    def test_default_config_get_params(self):
        config = DefaultConfig()
        params = config.get_params()
        assert isinstance(params, dict)
        assert "latent_dim" in params
        assert params["latent_dim"]["default"] == 16

    def test_default_config_print_schema(self, capsys):
        config = DefaultConfig()
        config.print_schema()
        captured = capsys.readouterr()
        assert "DefaultConfig Configuration Parameters:" in captured.out
        assert "latent_dim:" in captured.out
