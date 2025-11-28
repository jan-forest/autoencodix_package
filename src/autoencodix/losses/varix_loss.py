import torch
from typing import Tuple, Optional

from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig


class VarixLoss(BaseLoss):
    """Implements loss for variational autoencoder with unified interface.
    Attributes:
        config: Configuration object
    """

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        """Inits VarixLoss
        Args:
            config: Configuraion object.Any
            annealing_scheduler: Enables passing a custom annealer class, defaults to our implementation of an annealer
        """
        super().__init__(config, annealing_scheduler=annealing_scheduler)

    def _compute_losses(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction and variational losses.

        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            targets: original data to compare with reconstruction

        Returns:
            Tuple of torch.Tensors: reconstruction loss and variational loss

        """
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
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass with conditional annealing.
        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            targets: original data to compare with reconstruction
            epoch: current training epoch
            total_epochs: number of total epochs
            **kwargs
        Returns:
            Tuple consisting of:
            - tensor of the total loss
        Returns:
            Tuple consisting of:
            - tensor of the total loss
            - Dict with loss_type as key and sub_loss value.
            - Dict with loss_type as key and sub_loss value.

        """

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
            "var_loss": var_loss * effective_beta,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
        }
