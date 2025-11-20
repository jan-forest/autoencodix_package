import torch
from typing import Dict, Optional, Tuple

from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput


class VanillixLoss(BaseLoss):
    """Implements loss for vanilla autoencoder."""

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates reconstruction loss as specified in config (BCE, MSE, etc.).

        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            targets: original data to compare with reconstruction
            epoch: not used for Vanillix loss.
            **kwargs: addtional keyword args.

        """
        total_loss = self.recon_loss(model_output.reconstruction, targets)
        return total_loss, {"recon_loss": total_loss}
