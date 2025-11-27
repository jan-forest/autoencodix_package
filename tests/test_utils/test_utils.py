from autoencodix.utils._utils import config_method
from typing import Optional, Union
from autoencodix.configs.default_config import DefaultConfig
import pytest


class MockClass:
    def __init__(self):
        self.config = DefaultConfig()

    @config_method(valid_params={"config", "epochs", "float_precision"})
    def mock_config_method(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ):
        all_kwargs = {**kwargs}
        return {"config": config, "kwargs": all_kwargs}

    @config_method(valid_params={"config", "epochs", "not_in_config"})
    def mock_config_method_invalid(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ):
        all_kwargs = {**kwargs}
        return {"config": config, "kwargs": all_kwargs}


class TestConfigMethod:

    @pytest.fixture
    def default_config(self):
        return DefaultConfig()

    @pytest.fixture
    def mock_class(self):
        return MockClass()

    def test_pass_class_config_when_none_is_given(self, mock_class, default_config):
        result = mock_class.mock_config_method()
        assert result["config"] == default_config

    def test_overwriting_config(self, mock_class, default_config):
        default_config.float_precision = "test"
        result = mock_class.mock_config_method(config=default_config)
        assert result["config"].float_precision == "test"

    def test_keyword_args_override_config(self, mock_class, default_config):
        result = mock_class.mock_config_method(epochs=123)
        assert result["config"].epochs == 123

    def test_invalid_default_config_params(self, mock_class):
        """
        Special case, when we allowed a keyword argument in a config_method
        that is not in the DefaultConfig class. We don't want to add it to the
        config object.

        """
        result = mock_class.mock_config_method_invalid(not_in_config="test")
        assert "not_in_config" not in result["config"].model_dump()

    def test_invalid_keyword_args(self, mock_class, default_config):
        """
        Of course we also don't allow keyword arguments that are not in the
        valid_params set to overwrite the config object, even if they would be
        valid parameters for the config
        """
        default_batch_size = default_config.batch_size
        new_batch_size = default_batch_size + 1
        result = mock_class.mock_config_method(epochs=123, batch_size=new_batch_size)
        assert result["config"].batch_size == default_batch_size
