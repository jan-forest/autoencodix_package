import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss
from autoencodix.configs import DefaultConfig
from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from typing import Tuple, Dict


class MaskixLoss(BaseLoss):
    """Loss of the masked autoencoder ``Maskix``, following scMAE (Fang et al. 2024).

    Combines a mask-prediction loss (binary cross-entropy: which entries were
    corrupted) with a weighted MSE reconstruction loss that gives corrupted
    entries more weight:

        total = delta_mask_predictor * mask_loss
                + (1 - delta_mask_predictor) * weighted_recon_loss

    Corrupted entries are weighted with ``delta_mask_corrupted``, uncorrupted
    ones with ``1 - delta_mask_corrupted``.

    (LLM-generated docstring, pending human review.)
    """

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        """Inits MaskixLoss
        Args:
            config: Configuration object.Any
            annealing_scheduler: Enables passing a custom annealer class, defaults to our implementation of an annealer
        """
        super().__init__(config, annealing_scheduler=annealing_scheduler)
        self._validate_loss_config()

    def _validate_model_output(self, model_output: ModelOutput):
        """Ensure the model output contains the predicted mask.

        Args:
            model_output: Output of the Maskix architecture.

        Raises:
            ValueError: If ``additional_info`` is missing or has no
                "predicted_mask".
            TypeError: If ``additional_info`` is not a dict.

        (LLM-generated docstring, pending human review.)
        """
        if model_output.additional_info is None:
            raise ValueError(
                "For Maskix, we need to provide an 'additional_info' attribute in the ModelOutput, this likely went wrong in the architecture's forward method"
            )
        if not isinstance(model_output.additional_info, dict):
            raise TypeError(
                f"The `additional_info` attribute of ModelOutput needs to be of type dict, got {type(model_output.additional_info)}"
            )
        if "predicted_mask" not in model_output.additional_info.keys():
            raise ValueError(
                f"For Maskix, we require 'predicted_mask' to be in the additional_info attribute of ModelOutput, got: {model_output.additional_info.keys()}"
            )

    def _validate_loss_config(self):
        """Warn if the config deviates from the published setup (mean reduction, MSE).

        (LLM-generated docstring, pending human review.)
        """
        if not self.config.loss_reduction == "mean":
            import warnings

            warnings.warn(
                f"You chose loss reduction: {self.config.loss_reduction}, this deviates from the implementation in the literature for this architecture, the authors used 'mean'"
            )
        if not self.config.reconstruction_loss == "mse":
            import warnings

            warnings.warn(
                f"You chose {self.config.reconstruction_loss}, however we support only 'mse' for Maskix, we will use 'mse'"
            )

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        corrupted_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total Maskix loss and its components.

        Args:
            model_output: Model output with ``reconstruction`` and
                ``additional_info["predicted_mask"]`` (logits).
            targets: Original, uncorrupted input.
            corrupted_input: Masked input fed to the model; entries that differ
                from ``targets`` count as masked.
            **kwargs: Ignored; accepted for a uniform trainer interface.

        Returns:
            Tuple of the total loss and a dict with "recon_loss" (unweighted, for
            monitoring), "recon_loss_weighted" and "mask_loss" (both as weighted
            in the total).

        Raises:
            ValueError: If the predicted mask and the true mask differ in shape.

        (LLM-generated docstring, pending human review.)
        """
        self._validate_model_output(model_output)
        predicted_mask = model_output.additional_info["predicted_mask"]  # ty: ignore

        is_masked = (corrupted_input != targets).float()
        if not predicted_mask.shape == is_masked.shape:
            raise ValueError(
                f"Shape mismatch: predicted_mask {predicted_mask.shape} vs mask {is_masked.shape}"
            )
        mask_loss: torch.Tensor = bce_logits(
            predicted_mask, is_masked, reduction=self.config.loss_reduction
        )

        # this is from the publication
        # the authors want to give higher incentives to reconstruct corrupted features correctly
        corrupted_weight_matrix: torch.Tensor = (
            self.config.delta_mask_corrupted * is_masked
            + (1 - is_masked) * (1 - self.config.delta_mask_corrupted)
        )
        recon_loss: torch.Tensor = mse_loss(
            model_output.reconstruction, targets, reduction="none"
        )
        recon_loss_weighted = self.reduction_fn(
            torch.mul(corrupted_weight_matrix, recon_loss)
        )
        total_loss: torch.Tensor = (
            mask_loss * self.config.delta_mask_predictor
            + (1 - self.config.delta_mask_predictor) * recon_loss_weighted
        )

        recon_loss: torch.Tensor = mse_loss(
            model_output.reconstruction, targets, reduction=self.config.loss_reduction
        )

        # predicted_mask: torch.tensor
        return total_loss, {
            "recon_loss": recon_loss,
            "recon_loss_weighted": (1 - self.config.delta_mask_predictor)
            * recon_loss_weighted,
            "mask_loss": mask_loss * self.config.delta_mask_predictor,
        }
