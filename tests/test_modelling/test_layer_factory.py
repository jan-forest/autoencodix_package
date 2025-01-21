import pytest
import torch

from autoencodix.modeling._layer_factory import LayerFactory

class TestLayerFactory:

    @pytest.fixture
    def layer_factory(self):
        return LayerFactory()

    def test_last_layer_creation_len(self, layer_factory):
        last_layer = layer_factory.create_layer(10, 5, last_layer=True)
        assert len(last_layer) == 1

    def test_last_layer_creation_type(self, layer_factory):
        last_layer = layer_factory.create_layer(10, 5, last_layer=True)
        assert type(last_layer[0]) is torch.nn.Linear

    def test_non_last_layer_creation_len(self, layer_factory):
        non_last_layer = layer_factory.create_layer(10, 5, last_layer=False)
        assert len(non_last_layer) == 4