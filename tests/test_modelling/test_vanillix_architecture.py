import pytest
import torch

from autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.modeling._layer_factory import LayerFactory


class TestVanillixUnit:

    @pytest.fixture
    def defaults(self):
        latent_dim = 2
        n_layers = 3
        data_shape = (5, 10)
        return latent_dim, n_layers, data_shape

    @pytest.fixture
    def model(self, defaults):
        latent_dim, n_layers, data_shape = defaults
        config = DefaultConfig()
        config.latent_dim = latent_dim
        config.n_layers = n_layers
        return VanillixArchitecture(config=config, input_dim=data_shape[1])

    @pytest.fixture
    def sample_data(self, defaults):
        _, _, data_shape = defaults
        return torch.zeros(data_shape)

    def test_latent_shape_shape(self, model, sample_data, defaults):
        latent_dim, _, data_shape = defaults
        latent = model.encode(sample_data)
        assert latent.shape[0] == data_shape[0]
        assert latent.shape[1] == latent_dim

    def test_decoder_output_shape(self, model, defaults):
        latent_dim, _, data_shape = defaults
        latent_repr = torch.zeros(data_shape[0], latent_dim)
        output = model.decode(latent_repr)
        assert output.shape == data_shape

    def test_forward_pass_dtype(self, model, sample_data):
        output = model(sample_data)
        assert isinstance(output, ModelOutput)


class TestVanillixIntegration:
    @pytest.fixture
    def feature_dim(self):
        return 16

    @pytest.fixture
    def n_samples(self):
        return 3

    @pytest.fixture
    def config(self):
        config = DefaultConfig()
        config.latent_dim = 4
        config.n_layers = 3
        config.enc_factor = 2.0
        return config

    def test_layer_dimensions_integration(self, config, feature_dim):
        dims = LayerFactory.get_layer_dimensions(
            feature_dim=feature_dim,
            latent_dim=config.latent_dim,
            n_layers=config.n_layers,
            enc_factor=config.enc_factor,
        )
        assert dims[0] == feature_dim

    def test_layer_dimensions_deep(self, config, feature_dim):
        dims = LayerFactory.get_layer_dimensions(
            feature_dim=feature_dim,
            latent_dim=config.latent_dim,
            n_layers=config.n_layers,
            enc_factor=config.enc_factor,
        )
        assert dims[1] == feature_dim // config.enc_factor

    def test_full_forward_pass(self, config, feature_dim, n_samples):
        model = VanillixArchitecture(config=config, input_dim=feature_dim)
        x = torch.zeros(n_samples, feature_dim)
        output = model(x)
        assert output.reconstruction.shape == x.shape

    def test_architecture_layers_match_dimensions(self, config, feature_dim):
        model = VanillixArchitecture(config=config, input_dim=feature_dim)
        dims = LayerFactory.get_layer_dimensions(
            16, config.latent_dim, config.n_layers, config.enc_factor
        )
        assert model._encoder[0].in_features == dims[0]

    def test_forward_pass_output_types(self, config, feature_dim, n_samples):
        model = VanillixArchitecture(config=config, input_dim=feature_dim)
        x = torch.zeros(n_samples, feature_dim)
        output = model(x)
        assert isinstance(output.reconstruction, torch.Tensor)
        assert isinstance(output.latentspace, torch.Tensor)
