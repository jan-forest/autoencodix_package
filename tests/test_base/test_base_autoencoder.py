import pytest
import torch
import torch.nn as nn
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils.default_config import DefaultConfig


class TestBaseAutoencoder:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAutoencoder(None, 10)

    def test_weight_initialization(self):
        layer = nn.Linear(10, 5)

        # Create a dummy concrete class just to test _init_weights
        class ConcreteAutoencoder(BaseAutoencoder):
            def _build_network(self):
                pass

            def encode(self, x):
                pass

            def decode(self, x):
                pass

            def forward(self, x):
                pass

        model = ConcreteAutoencoder(config=None, input_dim=10)

        model._init_weights(layer)
        assert torch.all(layer.bias == 0.01)

    def test_config_default_creation(self):
        class ConcreteAutoencoder(BaseAutoencoder):
            def _build_network(self):
                pass

            def encode(self, x):
                pass

            def decode(self, x):
                pass

            def forward(self, x):
                pass

        # test that default config is there when none is passed
        model = ConcreteAutoencoder(config=None, input_dim=10)
        assert isinstance(model.config, DefaultConfig)
