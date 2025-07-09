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

        # Determine the forward function strategy at initialization
        if self.config.anneal_function == "no-annealing":
            self._forward_impl = self._forward_with_annealing
        else:
            self._forward_impl = self._forward_with_annealing

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Unified forward interface - delegates to appropriate implementation."""
        return self._forward_impl(model_output, targets, epoch)

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

    def _forward_without_annealing(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass without annealing - uses constant beta (ignores epoch)."""
        recon_loss, var_loss = self._compute_losses(model_output, targets)
        total_loss = recon_loss + self.config.beta * var_loss
        return total_loss, {"recon_loss": recon_loss, "var_loss": var_loss}

    def _forward_with_annealing(
        self, model_output: ModelOutput, targets: torch.Tensor, epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with annealing - calculates annealing factor from epoch."""
        recon_loss, var_loss = self._compute_losses(model_output, targets)

        annealing_epoch = self.annealing_scheduler.get_annealing_epoch(
            anneal_pretraining=self.config.anneal_pretraining,
            n_epochs_pretrain=self.config.pretrain_epochs,
            current_epoch=epoch,
        )

        # Get annealing weight
        anneal_factor = self.annealing_scheduler.get_weight(
            epoch_current=annealing_epoch,
            total_epoch=self.config.epochs,
            func=self.config.anneal_function,
        )

        # Apply annealed beta
        effective_beta = self.config.beta * anneal_factor
        total_loss = recon_loss + effective_beta * var_loss

        return total_loss, {
            "recon_loss": recon_loss,
            "var_loss": var_loss,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
        }
