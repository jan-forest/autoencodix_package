import pytest
import torch
import torch.nn as nn
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput


class TestBaseAutoencoder:
    # Helper for creating a concrete class for testing
    @pytest.fixture
    def concrete_autoencoder_class(self):
        class ConcreteAutoencoder(BaseAutoencoder):
            def _build_network(self):
                self._encoder = nn.Linear(10, 5)
                self._decoder = nn.Linear(5, 10)

            def encode(self, x):
                return self._encoder(x)

            def get_latent_space(self, x):
                return self.encode(x)

            def decode(self, x):
                return self._decoder(x)

            def forward(self, x):
                z = self.encode(x)
                recon = self.decode(z)
                return ModelOutput(reconstruction=recon, latentspace=z)

        return ConcreteAutoencoder

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAutoencoder(None, 10)

    def test_weight_initialization_sets_bias(self):
        layer = nn.Linear(10, 5)

        # Create a minimal concrete class
        class MinimalAutoencoder(BaseAutoencoder):
            def _build_network(self):
                pass

            def encode(self, x):
                pass

            def get_latent_space(self, x):
                pass

            def decode(self, x):
                pass

            def forward(self, x):
                pass

        model = MinimalAutoencoder(config=None, input_dim=10)
        model._init_weights(layer)
        assert torch.all(layer.bias == 0.01)

    def test_weight_initialization_uses_xavier_uniform(self):
        layer = nn.Linear(10, 5)
        original_weights = layer.weight.clone()

        class MinimalAutoencoder(BaseAutoencoder):
            def _build_network(self):
                pass

            def encode(self, x):
                pass

            def get_latent_space(self, x):
                pass

            def decode(self, x):
                pass

            def forward(self, x):
                pass

        model = MinimalAutoencoder(config=None, input_dim=10)
        model._init_weights(layer)
        # Check that weights changed (xavier uniform was applied)
        assert not torch.allclose(original_weights, layer.weight)

    def test_default_config_created_when_none_provided(
        self, concrete_autoencoder_class
    ):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        assert isinstance(model.config, DefaultConfig)

    def test_input_dim_attribute_set_correctly(self, concrete_autoencoder_class):
        input_dim = 10
        model = concrete_autoencoder_class(config=None, input_dim=input_dim)
        assert model.input_dim == input_dim

    def test_encoder_is_none_before_build_network(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        assert model._encoder is None

    def test_decoder_is_none_before_build_network(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        assert model._decoder is None

    def test_encoder_is_nn_module_after_build_network(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        assert isinstance(model._encoder, nn.Module)

    def test_decoder_is_nn_module_after_build_network(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        assert isinstance(model._decoder, nn.Module)

    def test_custom_config_is_used(self):
        class CustomConfig(DefaultConfig):
            def __init__(self):
                super().__init__()
                self.batch_size = 27

        class MinimalAutoencoder(BaseAutoencoder):
            def _build_network(self):
                pass

            def encode(self, x):
                pass

            def get_latent_space(self, x):
                pass

            def decode(self, x):
                pass

            def forward(self, x):
                pass

        custom_config = CustomConfig()
        model = MinimalAutoencoder(config=custom_config, input_dim=10)
        assert model.config.batch_size == 27

    def test_encode_returns_correct_shape(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        z = model.encode(x)
        assert z.shape == (3, 5)

    def test_decode_returns_correct_shape(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        z = torch.randn(3, 5)
        recon = model.decode(z)
        assert recon.shape == (3, 10)

    def test_get_latent_space_returns_correct_shape(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        latent = model.get_latent_space(x)
        assert latent.shape == (3, 5)

    def test_forward_returns_model_output_instance(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        output = model(x)
        assert isinstance(output, ModelOutput)

    def test_forward_output_has_reconstruction_attribute(
        self, concrete_autoencoder_class
    ):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        output = model(x)
        assert hasattr(output, "reconstruction")

    def test_forward_output_has_latent_space_attribute(
        self, concrete_autoencoder_class
    ):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        output = model(x)
        assert hasattr(output, "latentspace")

    def test_reconstruction_has_same_shape_as_input(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        output = model(x)
        assert output.reconstruction.shape == x.shape

    def test_latent_space_has_expected_shape(self, concrete_autoencoder_class):
        model = concrete_autoencoder_class(config=None, input_dim=10)
        model._build_network()
        x = torch.randn(3, 10)
        output = model(x)
        assert output.latentspace.shape == (3, 5)
