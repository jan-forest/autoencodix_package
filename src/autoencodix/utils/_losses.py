import torch
from typing import Dict, Tuple

from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput


class VanillixLoss(BaseLoss):
    """Loss function for vanilla autoencoder."""

    def forward(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = self.recon_loss(model_output.reconstruction, targets)
        return total_loss, {"recon_loss": total_loss}


class VarixLoss(BaseLoss):
    """Loss function for variational autoencoder."""

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        true_samples = torch.randn(
            self.config.batch_size, self.config.latent_dim, requires_grad=False
        )
        recon_loss = self.recon_loss(model_output.reconstruction, targets)
        var_loss = self.compute_variational_loss(
            mu=model_output.latent_mean,
            logvar=model_output.latent_logvar,
            z=model_output.latentspace,
            true_samples=true_samples,
        )
        total_loss = recon_loss + self.config.beta * var_loss
        return total_loss, {"recon_loss": recon_loss, "var_loss": var_loss}
