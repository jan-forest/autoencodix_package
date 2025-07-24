import torch
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._utils import flip_labels


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

    def forward(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
        clf_scores: torch.Tensor,
        labels: torch.Tensor,
        clf_loss_fn: torch.nn.Module,
    ):
        adver_loss = self._calc_adversial_loss(
            labels=labels, clf_loss_fn=clf_loss_fn, clf_scores=clf_scores
        )
        aggregated_sub_losses = self._sum_sub_losses(
            modality_dynamics=modality_dynamics
        )
        paired_loss = self._calc_paired_loss(
            batch=batch, modality_dynamics=modality_dynamics
        )
        class_loss = self._calc_class_loss(
            batch=batch, modality_dynamics=modality_dynamics
        )
        total_loss = adver_loss + aggregated_sub_losses + paired_loss + class_loss
        return total_loss, {
            "total_loss": total_loss,
            "adver_loss": adver_loss,
            "aggregated_sub_losses": aggregated_sub_losses,
            "paired_loss": paired_loss,
            "class_loss": class_loss,
        }

    def _calc_paired_loss(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
    ):
        return torch.tensor(0)  # TODO

    def _calc_class_loss(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
    ):
        return torch.tensor(0)  # TODO

    def _sum_sub_losses(
        self, modality_dynamics: Dict[str, Dict[str, Any]]
    ) -> torch.Tensor:
        """Computes the average total loss for all modalities."""
        losses = [helper["loss"] for helper in modality_dynamics.values()]
        return torch.stack(losses).mean()

    def _calc_adversial_loss(
        self,
        labels: torch.Tensor,
        clf_scores: torch.Tensor,
        clf_loss_fn: torch.nn.Module,
    ):
        flipped_labels = flip_labels(labels=labels)
        adversarial_loss = clf_loss_fn(clf_scores, flipped_labels)
        return adversarial_loss
