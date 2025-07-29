from typing import Literal, Tuple
import pytest
import torch

from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig


class TestVarixArchitecture:
    @pytest.fixture
    def defaults(self):
        latent_dim = 2
        n_layers = 3
        data_shape = (5, 10)
        return latent_dim, n_layers, data_shape

    @pytest.fixture
    def model(
        self, defaults: tuple[Literal[2], Literal[3], tuple[Literal[5], Literal[10]]]
    ):
        latent_dim, n_layers, data_shape = defaults
        config = DefaultConfig()
        config.latent_dim = latent_dim
        config.n_layers = n_layers
        return VarixArchitecture(config=config, input_dim=data_shape[1])

    @pytest.fixture
    def sample_data(
        self, defaults: tuple[Literal[2], Literal[3], tuple[Literal[5], Literal[10]]]
    ):
        _, _, data_shape = defaults
        return torch.zeros(data_shape)

    @pytest.mark.parametrize(
        "config",
        [
            DefaultConfig(n_layers=3, latent_dim=2, enc_factor=2),
            DefaultConfig(n_layers=5, latent_dim=4, enc_factor=4),
            DefaultConfig(n_layers=2, latent_dim=8, enc_factor=1),
            DefaultConfig(n_layers=1, latent_dim=16, enc_factor=1),
            DefaultConfig(n_layers=4, latent_dim=32, enc_factor=3),
        ],
    )
    @pytest.mark.parametrize(
        "data",
        [
            torch.zeros((5, 10)),
            torch.zeros((10, 5)),
            torch.zeros((2, 3)),
            torch.zeros((20, 5)),
            torch.zeros((13, 23)),
        ],
    )
    def test_shapes(
        self,
        config: DefaultConfig,
        data: torch.Tensor,
    ):
        model = VarixArchitecture(config=config, input_dim=data.shape[1])
        mu, logvar = model.encode(data)

        latent = model.reparameterize(mu, logvar)
        decoder_output = model.decode(latent)
        # multiple asserts because we don't want to calc the model output multiple times
        assert latent.shape == (data.shape[0], config.latent_dim)
        assert mu.shape == (data.shape[0], config.latent_dim)
        assert logvar.shape == (data.shape[0], config.latent_dim)
        assert decoder_output.shape == data.shape

    def test_forward_pass_dtype(
        self, model: VarixArchitecture, sample_data: torch.Tensor
    ):
        output = model(sample_data)
        assert isinstance(output, ModelOutput)

    def test_model_output_shape(
        self,
        model: VarixArchitecture,
        sample_data: torch.Tensor,
        defaults: Tuple[int, int, Tuple[int, int]],
    ):
        latent_dim, _, data_shape = defaults
        output = model(sample_data)
        assert output.reconstruction.shape == sample_data.shape
        assert output.latentspace.shape == (sample_data.shape[0], latent_dim)
        assert output.latent_mean.shape == (sample_data.shape[0], latent_dim)
        assert output.latent_logvar.shape == (sample_data.shape[0], latent_dim)

    @pytest.mark.parametrize(
        "config,input_dim",
        [
            # Edge cases that should work
            (DefaultConfig(n_layers=3, latent_dim=10, enc_factor=2), 10),  # Equal dims
            (DefaultConfig(n_layers=1, latent_dim=8, enc_factor=2), 16),  # Single layer
            (DefaultConfig(n_layers=3, latent_dim=8, enc_factor=1), 16),  # No reduction
            (
                DefaultConfig(n_layers=3, latent_dim=7, enc_factor=2),
                30,
            ),  # Uneven reduction
        ],
    )
    def test_edge_cases(self, config: DefaultConfig, input_dim: int):
        model = VarixArchitecture(config=config, input_dim=input_dim)
        # Test forward pass with dummy data
        data = torch.zeros((5, input_dim))
        output = model(data)
        assert output.reconstruction.shape == data.shape
        assert output.latentspace.shape == (5, config.latent_dim)

    @pytest.mark.parametrize(
        "config, input_dim, expected_mu_input",
        [
            (
                DefaultConfig(n_layers=0, enc_factor=1, batch_size=1, latent_dim=16),
                100,
                100,
            ),
            (
                DefaultConfig(n_layers=1, enc_factor=1, batch_size=1, latent_dim=16),
                100,
                100,
            ),
            (
                DefaultConfig(n_layers=4, enc_factor=1, batch_size=1, latent_dim=16),
                100,
                100,
            ),
            (
                DefaultConfig(n_layers=2, enc_factor=2, batch_size=1, latent_dim=16),
                100,
                25,
            ),
            (
                DefaultConfig(n_layers=3, enc_factor=2, batch_size=1, latent_dim=16),
                100,
                16,
            ),  # cant be smaller than latent_dim
            (
                DefaultConfig(n_layers=3, enc_factor=2, batch_size=1, latent_dim=2),
                100,
                12,
            ),
        ],
    )
    def test_mu_details(
        self, config: DefaultConfig, input_dim: int, expected_mu_input: int
    ):
        model = VarixArchitecture(config=config, input_dim=input_dim)
        mu_input = model._mu.in_features
        assert mu_input == expected_mu_input

    @pytest.mark.parametrize(
        "config",
        [
            DefaultConfig(n_layers=0, enc_factor=1, batch_size=1, latent_dim=16),
            DefaultConfig(n_layers=1, enc_factor=1, batch_size=1, latent_dim=16),
            DefaultConfig(n_layers=4, enc_factor=1, batch_size=1, latent_dim=16),
            DefaultConfig(n_layers=2, enc_factor=2, batch_size=1, latent_dim=16),
            DefaultConfig(n_layers=3, enc_factor=2, batch_size=1, latent_dim=16),
            DefaultConfig(n_layers=3, enc_factor=2, batch_size=1, latent_dim=2),
        ],
    )
    def test_symmetric(self, config: DefaultConfig):
        model = VarixArchitecture(config=config, input_dim=100)
        decoder = model._decoder
        # decrease by 1 because mu/logvar layers are counted for encoder, but are not technically inluded in the encoder
        decoder_length = len(decoder) - 1
        encoder_length = len(model._encoder)
        assert encoder_length == decoder_length

    @pytest.mark.parametrize(
        "config, input_dim",
        [
            (DefaultConfig(n_layers=2, enc_factor=2, batch_size=1, latent_dim=8), 64),
            (DefaultConfig(n_layers=3, enc_factor=3, batch_size=1, latent_dim=4), 81),
            (DefaultConfig(n_layers=1, enc_factor=1, batch_size=1, latent_dim=16), 32),
        ],
    )
    def test_latent_space_size(self, config: DefaultConfig, input_dim: int):
        model = VarixArchitecture(config=config, input_dim=input_dim)
        assert model._mu.out_features == config.latent_dim
        assert model._logvar.out_features == config.latent_dim
