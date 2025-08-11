import torch
import torch.nn as nn
import pytest

from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.utils._model_output import ModelOutput


@pytest.fixture
def default_config():
    return DefaultConfig()


@pytest.fixture
def varix_loss_module(default_config):
    return VarixLoss(config=default_config)


@pytest.fixture
def sample_data():
    batch_size = 10
    feature_dim = 5
    return {
        "input": torch.zeros(batch_size, feature_dim),
        "z": torch.zeros(batch_size, feature_dim),
        "mu": torch.zeros(batch_size, feature_dim),
        "logvar": torch.zeros(batch_size, feature_dim),
        "reconstruction": torch.zeros(batch_size, feature_dim),
    }


@pytest.fixture
def model_output(sample_data):
    return ModelOutput(
        reconstruction=sample_data["reconstruction"],
        latentspace=sample_data["z"],
        latent_mean=sample_data["mu"],
        latent_logvar=sample_data["logvar"],
    )


class TestVarixLoss:
    def test_mse_loss_initialization(self, default_config, varix_loss_module):
        """
        Test that MSE loss is correctly initialized when specified in config.
        """
        loss = varix_loss_module
        default_config.reconstruction_loss = "mse"
        assert isinstance(loss.recon_loss, nn.MSELoss)

    def test_bce_loss_initialization(self, default_config):
        """
        Test that BCE loss is correctly initialized when specified in config.
        """
        default_config.reconstruction_loss = "bce"
        loss = VarixLoss(config=default_config)
        assert isinstance(loss.recon_loss, nn.BCEWithLogitsLoss)

    def test_invalid_reduction_config(self, default_config):
        """
        Test that invalid reduction method raises NotImplementedError.
        """
        config = default_config
        config.loss_reduction = "invalid"
        with pytest.raises(NotImplementedError):
            VarixLoss(config=config)

    def test_invalid_reconstruction_loss_config(self, default_config):
        """
        Test that invalid reconstruction loss type raises NotImplementedError.
        """
        config = default_config
        config.reconstruction_loss = "invalid"
        with pytest.raises(NotImplementedError):
            VarixLoss(config)

    def test_mmd_kernel_shape(self, varix_loss_module, sample_data):
        """
        Test that MMD kernel output has correct shape (batch_size x batch_size).
        """
        x = sample_data["z"]
        y = torch.randn_like(x)
        kernel = varix_loss_module._mmd_kernel(x, y)
        assert kernel.shape == (x.size(0), y.size(0))

    def test_mmd_kernel_bounds(self, varix_loss_module, sample_data):
        """
        Test that MMD kernel values are bounded between 0 and 1.
        """
        x = sample_data["z"]
        y = torch.randn_like(x)
        kernel = varix_loss_module._mmd_kernel(x, y)
        assert ((kernel >= 0) & (kernel <= 1)).all()

    def test_mmd_loss_type(self, varix_loss_module, sample_data):
        """
        Test that MMD loss returns a scalar tensor.
        """
        z = sample_data["z"]
        true_samples = torch.randn_like(z)
        mmd_loss = varix_loss_module.compute_mmd_loss(z, true_samples)
        assert isinstance(mmd_loss, torch.Tensor)

    def test_mmd_loss_shape(self, varix_loss_module, sample_data):
        """
        Test that MMD loss is a scalar (0-dimensional tensor).
        """
        z = sample_data["z"]
        true_samples = torch.randn_like(z)
        mmd_loss = varix_loss_module.compute_mmd_loss(z, true_samples)
        assert mmd_loss.ndim == 0

    def test_kl_loss_type(self, varix_loss_module, sample_data):
        """
        Test that KL loss returns a scalar tensor.
        """
        kl_loss = varix_loss_module.compute_kl_loss(
            sample_data["mu"], sample_data["logvar"]
        )
        assert isinstance(kl_loss, torch.Tensor)

    def test_kl_loss_shape(self, varix_loss_module, sample_data):
        """
        Test that KL loss is a scalar (0-dimensional tensor).
        """
        kl_loss = varix_loss_module.compute_kl_loss(
            sample_data["mu"], sample_data["logvar"]
        )
        assert kl_loss.ndim == 0

    def test_variational_loss_kl_type(self, varix_loss_module, sample_data):
        """
        Test that variational KL loss returns a scalar tensor.
        """
        loss = varix_loss_module.compute_variational_loss(
            sample_data["mu"], sample_data["logvar"]
        )
        assert isinstance(loss, torch.Tensor)

    def test_variational_loss_mmd_type(self, sample_data):
        """
        Test that variational MMD loss returns a scalar tensor.
        """
        config = DefaultConfig(default_vae_loss="mmd")
        loss_module = VarixLoss(config)
        loss = loss_module.compute_variational_loss(
            sample_data["mu"],
            sample_data["logvar"],
            sample_data["z"],
            torch.randn_like(sample_data["z"]),
        )
        assert isinstance(loss, torch.Tensor)

    def test_forward_total_loss_type(
        self, varix_loss_module, sample_data, model_output
    ):
        """
        Test that forward pass returns total loss as a scalar tensor.
        """
        total_loss, _ = varix_loss_module(model_output, sample_data["input"])
        assert isinstance(total_loss, torch.Tensor)

    def test_forward_dict_type(self, varix_loss_module, sample_data, model_output):
        """
        Test that forward pass returns dictionary with loss components as tensors.
        """
        _, loss_dict = varix_loss_module(model_output, sample_data["input"])
        assert isinstance(loss_dict, dict)

    def test_forward_loss_dict_structure(
        self, varix_loss_module, sample_data, model_output
    ):
        """
        Test that forward pass returns dictionary with required loss components.
        """
        _, loss_dict = varix_loss_module(model_output, sample_data["input"])
        assert len(loss_dict) == 2

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_methods(self, reduction, sample_data, model_output):
        """
        Test that different reduction methods (mean/sum) produce valid losses.
        """
        config = DefaultConfig(loss_reduction=reduction)
        loss_module = VarixLoss(config)
        total_loss, _ = loss_module(model_output, sample_data["input"])
        assert not torch.isnan(total_loss)

    def test_missing_mu_error(self, varix_loss_module):
        """
        Test that computing variational loss with missing mu raises ValueError.
        """
        with pytest.raises(ValueError):
            varix_loss_module.compute_variational_loss(None, torch.randn(10, 5))

    def test_missing_logvar_error(self, varix_loss_module):
        """
        Test that computing variational loss with missing logvar raises ValueError.
        """
        with pytest.raises(ValueError):
            varix_loss_module.compute_variational_loss(torch.randn(10, 5), None)

    def test_missing_z_error_mmd(self):
        """
        Test that computing MMD loss with missing z raises ValueError.
        """
        config = DefaultConfig(default_vae_loss="mmd")
        loss_module = VarixLoss(config)
        with pytest.raises(ValueError):
            loss_module.compute_variational_loss(
                torch.randn(10, 5), torch.randn(10, 5), None
            )
