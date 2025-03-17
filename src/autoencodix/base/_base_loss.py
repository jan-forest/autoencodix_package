from abc import abstractmethod, ABC
from autoencodix.utils.default_config import DefaultConfig
import torch
from torch import nn
from autoencodix.utils._model_output import ModelOutput

from typing import Optional, Tuple, Dict


class BaseLoss(nn.Module, ABC):
    """Base class for autoencoder loss functions."""

    def __init__(self, config: DefaultConfig):
        super().__init__()
        self.config = config
        self.recon_loss: nn.Module
        if self.config.loss_reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.config.loss_reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            raise NotImplementedError(
                f"Invalid loss reduction type: {self.config.loss_reduction}. Only 'mean' and 'sum' are supported. This makes sure the reconstrunction loss and the variational loss are calculated correctly."
            )  # TODO maybe implement other reduction types later

        if self.config.reconstruction_loss == "mse":
            self.recon_loss = nn.MSELoss(reduction=config.loss_reduction)
        elif self.config.reconstruction_loss == "bce":
            self.recon_loss = nn.BCEWithLogitsLoss(reduction=config.loss_reduction)
        else:
            raise NotImplementedError(
                f"Invalid reconstruction loss type: {self.config.reconstruction_loss}. Only 'mse' and 'bce' are supported."
            )

        # Setup compute kernel for MMD, pass here to be able to change kernel function in init
        self.compute_kernel = self._mmd_kernel

    def _mmd_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        computes kernel for maximum mean discrepancy (MMD) calculation
        Parameters:
            x - (torch.tensor): input data
            y - (torch.tensor): input data
        Returns:
            kernel - (torch.tensor): kernel matrix (x.shape[0]  y.shape[0])

        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)

        return torch.exp(-kernel_input)

    def compute_mmd_loss(
        self, z: torch.Tensor, true_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss.

        Parameters:
            z: Samples from the encoded distribution
            true_samples: Samples from the prior distribution (if None, generates from standard normal)
        """

        true_samples_kernel = self.compute_kernel(true_samples, true_samples)
        z_kernel = self.compute_kernel(z, z)
        ztr_kernel = self.compute_kernel(true_samples, z)

        if self.config.loss_reduction == "mean":
            return true_samples_kernel.mean() + z_kernel.mean() - 2 * ztr_kernel.mean()
        elif self.config.loss_reduction == "sum":
            return true_samples_kernel.sum() + z_kernel.sum() - 2 * ztr_kernel.sum()
        else:
            raise NotImplementedError(
                f"Invalid loss reduction type: {self.config.loss_reduction}. Only 'mean' and 'sum' are supported."
            )

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss between N(mu, logvar) and N(0, 1)."""
        if self.config.loss_reduction == "mean":
            return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:  # sum
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def compute_variational_loss(
        self,
        mu: Optional[torch.Tensor],
        logvar: Optional[torch.Tensor],
        z: Optional[torch.Tensor] = None,
        true_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute either KL or MMD loss based on config."""
        if mu is None:
            raise ValueError("mu must be provided for VAE loss")
        if logvar is None:
            raise ValueError("logvar must be provided for VAE loss")
        if self.config.default_vae_loss == "kl":
            return self.compute_kl_loss(mu=mu, logvar=logvar)
        elif self.config.default_vae_loss == "mmd":
            if z is None:
                raise ValueError("z must be provided for MMD loss")
            if true_samples is None:
                raise ValueError("true_samples must be provided for MMD loss")
            return self.compute_mmd_loss(z=z, true_samples=true_samples)
        else:
            raise NotImplementedError(
                f"VAE loss type {self.config.default_vae_loss} is not implemented. Only 'kl' and 'mmd' are supported."
            )

    @abstractmethod
    def forward(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate the loss for the autoencoder."""
        pass
