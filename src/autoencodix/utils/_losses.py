import torch
from typing import Dict, Tuple, Optional
from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig


class VanillixLoss(BaseLoss):
    """Loss function for vanilla autoencoder."""

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for vanilla autoencoder - ignores epoch."""
        total_loss = self.recon_loss(model_output.reconstruction, targets)
        return total_loss, {"recon_loss": total_loss}


class VarixLoss(BaseLoss):
    """Loss function for variational autoencoder with unified interface."""

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        super().__init__(config)

    def _compute_losses(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction and variational losses."""
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

        return recon_loss, var_loss

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with conditional annealing."""
        recon_loss, var_loss = self._compute_losses(model_output, targets)

        # if are pretraining, we pass total_epochs, otherwise, we use 'epochs' from config
        calc_epochs: int = self.config.epochs
        if total_epochs:
            calc_epochs = total_epochs

        if self.config.anneal_function == "no-annealing":
            # Use constant beta
            effective_beta = self.config.beta
            anneal_factor = 1.0
        else:
            anneal_factor = self.annealing_scheduler.get_weight(
                epoch_current=epoch,
                total_epoch=calc_epochs,
                func=self.config.anneal_function,
            )
            effective_beta = self.config.beta * anneal_factor

        total_loss = recon_loss + effective_beta * var_loss

        return total_loss, {
            "recon_loss": recon_loss,
            "var_loss": var_loss,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
        }


class XModalLoss(BaseLoss):
    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        super().__init__(config)

    def forward():
        pass

    def _calc_paired_loss():
        pass

    def _calc_class_loss():
        pass

    def _weight_sub_losses():
        pass

    def _weight_adversial_loss():
        pass
