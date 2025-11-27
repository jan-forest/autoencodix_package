from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from autoencodix.base._base_loss import BaseLoss


@dataclass
class DefaultConfig:
    loss_reduction: str = "mean"
    reconstruction_loss: str = "mse"
    default_vae_loss: str = "kl"


@dataclass
class ModelOutput:
    recon_x: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


# Create a concrete implementation for testing
class LossImplementation(BaseLoss):
    def forward(self, model_output, targets):
        recon_loss = self.recon_loss(model_output.recon_x, targets)
        var_loss = self.compute_variational_loss(
            model_output.mu, model_output.logvar, model_output.z
        )
        return recon_loss + var_loss, {"recon_loss": recon_loss, "var_loss": var_loss}


@pytest.fixture
def mock_config():
    mock_config = Mock(spec=DefaultConfig)
    mock_config.loss_reduction = "mean"
    mock_config.reconstruction_loss = "mse"
    mock_config.default_vae_loss = "kl"
    return mock_config


@pytest.fixture
def loss_module(mock_config):
    return LossImplementation(mock_config)


@pytest.fixture
def sample_data():
    batch_size = 10
    feature_dim = 5
    return {
        "input": torch.randn(batch_size, feature_dim),
        "z": torch.randn(batch_size, feature_dim),
        "mu": torch.zeros(batch_size, feature_dim),
        "logvar": torch.zeros(batch_size, feature_dim),
    }


class TestBaseLoss:
    def test_mse_loss_initialization(self, mock_config):
        """
        Test that MSE loss is correctly initialized when specified in config.
        """
        loss = LossImplementation(mock_config)
        assert isinstance(loss.recon_loss, nn.MSELoss)

    def test_bce_loss_initialization(self):
        """
        Test that BCE loss is correctly initialized when specified in config.
        """
        config = DefaultConfig(reconstruction_loss="bce")
        loss = LossImplementation(config)
        assert isinstance(loss.recon_loss, nn.BCEWithLogitsLoss)

    def test_invalid_reduction_config(self):
        """
        Test that invalid reduction method raises NotImplementedError.
        """
        config = DefaultConfig(loss_reduction="invalid")
        with pytest.raises(NotImplementedError):
            LossImplementation(config)

    def test_invalid_reconstruction_loss_config(self):
        """
        Test that invalid reconstruction loss type raises NotImplementedError.
        """
        config = DefaultConfig(reconstruction_loss="invalid")
        with pytest.raises(NotImplementedError):
            LossImplementation(config)

    def test_mmd_kernel_shape(self, loss_module, sample_data):
        """
        Test that MMD kernel output has correct shape (batch_size x batch_size).
        """
        x = sample_data["z"]
        y = torch.randn_like(x)
        kernel = loss_module._mmd_kernel(x, y)
        assert kernel.shape == (x.size(0), y.size(0))

    def test_mmd_kernel_bounds(self, loss_module, sample_data):
        """
        Test that MMD kernel values are bounded between 0 and 1.
        """
        x = sample_data["z"]
        y = torch.randn_like(x)
        kernel = loss_module._mmd_kernel(x, y)
        assert ((kernel >= 0) & (kernel <= 1)).all()

    def test_mmd_loss_type(self, loss_module, sample_data):
        """
        Test that MMD loss returns a scalar tensor.
        """
        z = sample_data["z"]
        true_samples = torch.randn_like(z)
        mmd_loss = loss_module.compute_mmd_loss(z, true_samples)
        assert isinstance(mmd_loss, torch.Tensor)

    def test_mmd_loss_shape(self, loss_module, sample_data):
        """
        Test that MMD loss is a scalar (0-dimensional tensor).
        """
        z = sample_data["z"]
        true_samples = torch.randn_like(z)
        mmd_loss = loss_module.compute_mmd_loss(z, true_samples)
        assert mmd_loss.ndim == 0

    def test_kl_loss_type(self, loss_module, sample_data):
        """
        Test that KL loss returns a scalar tensor.
        """
        kl_loss = loss_module.compute_kl_loss(sample_data["mu"], sample_data["logvar"])
        assert isinstance(kl_loss, torch.Tensor)

    def test_kl_loss_shape(self, loss_module, sample_data):
        """
        Test that KL loss is a scalar (0-dimensional tensor).
        """
        kl_loss = loss_module.compute_kl_loss(sample_data["mu"], sample_data["logvar"])
        assert kl_loss.ndim == 0

    def test_variational_loss_kl_type(self, loss_module, sample_data):
        """
        Test that variational KL loss returns a scalar tensor.
        """
        loss = loss_module.compute_variational_loss(
            sample_data["mu"], sample_data["logvar"]
        )
        assert isinstance(loss, torch.Tensor)

    def test_variational_loss_mmd_type(self, sample_data):
        """
        Test that variational MMD loss returns a scalar tensor.
        """
        config = DefaultConfig(default_vae_loss="mmd")
        loss_module = LossImplementation(config)
        loss = loss_module.compute_variational_loss(
            sample_data["mu"],
            sample_data["logvar"],
            sample_data["z"],
            torch.randn_like(sample_data["z"]),
        )
        assert isinstance(loss, torch.Tensor)

    def test_forward_total_loss_type(self, loss_module, sample_data):
        """
        Test that forward pass returns total loss as a scalar tensor.
        """
        model_output = ModelOutput(
            recon_x=torch.randn_like(sample_data["input"]),
            z=sample_data["z"],
            mu=sample_data["mu"],
            logvar=sample_data["logvar"],
        )
        total_loss, _ = loss_module(model_output, sample_data["input"])
        assert isinstance(total_loss, torch.Tensor)

    def test_forward_loss_dict_structure(self, loss_module, sample_data):
        """
        Test that forward pass returns dictionary with required loss components.
        """
        model_output = ModelOutput(
            recon_x=torch.randn_like(sample_data["input"]),
            z=sample_data["z"],
            mu=sample_data["mu"],
            logvar=sample_data["logvar"],
        )
        _, loss_dict = loss_module(model_output, sample_data["input"])
        assert set(loss_dict.keys()) == {"recon_loss", "var_loss"}

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_methods(self, reduction, sample_data):
        """
        Test that different reduction methods (mean/sum) produce valid losses.
        """
        config = DefaultConfig(loss_reduction=reduction)
        loss_module = LossImplementation(config)
        model_output = ModelOutput(
            recon_x=torch.randn_like(sample_data["input"]),
            z=sample_data["z"],
            mu=sample_data["mu"],
            logvar=sample_data["logvar"],
        )
        total_loss, _ = loss_module(model_output, sample_data["input"])
        assert not torch.isnan(total_loss)

    def test_missing_mu_error(self, loss_module):
        """
        Test that computing variational loss with missing mu raises ValueError.
        """
        with pytest.raises(ValueError):
            loss_module.compute_variational_loss(None, torch.randn(10, 5))

    def test_missing_logvar_error(self, loss_module):
        """
        Test that computing variational loss with missing logvar raises ValueError.
        """
        with pytest.raises(ValueError):
            loss_module.compute_variational_loss(torch.randn(10, 5), None)

    def test_missing_z_error_mmd(self):
        """
        Test that computing MMD loss with missing z raises ValueError.
        """
        config = DefaultConfig(default_vae_loss="mmd")
        loss_module = LossImplementation(config)
        with pytest.raises(ValueError):
            loss_module.compute_variational_loss(
                torch.randn(10, 5), torch.randn(10, 5), None
            )
