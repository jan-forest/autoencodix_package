import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from collections import defaultdict
from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._annealer import AnnealingScheduler
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._utils import flip_labels


class XModalLoss(BaseLoss):
    """Implements Loss for XModalix.
    The loss of the XModalix consists of 4 parts:
    - Combined (mean) Reconstruction loss over all sub modalities.
    - Combined Distribution loss over all sub modalities (KL or MMD)
    - Class loss: When we have metadata information about a sample e.g.
        cancer type, we calculate the mean of all samples in the latent
        space of each class and then we calc the distance between the mean
        individual sample.
    - Paired loss: When samples of different modalities are paired, then
        their latens space representation should be proximal.
    - Advers loss: Forces the latentspaaces of different data modalities to
        be similar.
    The reconstruction and distribution loss are calculated, combined, and weighted
    (with hyperparam beta) for each sub-modality first. In this class we grab this
    loss from the modality_dynamics and combine them for all sub-modalities with the mean.
    Attributes:
        class_means_train:
        class_means_valid:
        sample_to_class_map:
    """

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        super().__init__(config)
        self.class_means_train: Dict[str, torch.Tensor] = {}
        self.class_means_valid: Dict[str, torch.Tensor] = {}
        self.sample_to_class_map: Dict[str, Any] = {}
        self.class_mean_momentum = 0.75

    def forward(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
        clf_scores: torch.Tensor,
        labels: torch.Tensor,
        clf_loss_fn: torch.nn.Module,
        is_training: bool = True,
        epoch: Optional[int] = None,
        **kwargs,
    ):
        """Forward pass of XModal loss.
        Args:
            batch: data from custom dataset
            modality_dynamics: trainingdynamics such as losses, reconstructions for each data modality.
            clf_scores: output of latent classifier for advers loss.
            labels: indicator to which datamodality which latentspace belongs (for adver. loss).
            clf_loss_fn: loss for clf, passed in xmodal_trainer.py, defaults Crossentropy.
            is_training: indicator if we're in training, false for valid and test loop. Used for class loss calc.
            **kwargs: addtional keyword arguments.
        """
        adver_loss = self._calc_adversial_loss(
            labels=labels, clf_loss_fn=clf_loss_fn, clf_scores=clf_scores
        )
        aggregated_sub_losses = self._combine_sub_losses(
            modality_dynamics=modality_dynamics
        )
        sub_losses = self._store_sub_losses(modality_dynamics=modality_dynamics)
        paired_loss = self._calc_paired_loss(
            batch=batch, modality_dynamics=modality_dynamics
        )
        class_loss = self._calc_class_loss(
            batch=batch,
            modality_dynamics=modality_dynamics,
            is_training=is_training,
        )
        total_loss = (
            self.config.gamma * adver_loss
            + aggregated_sub_losses  # beta already applied when calcing sub_loss in train loop (works because of distributive law.)
            + self.config.delta_pair * paired_loss
            + self.config.delta_class * class_loss
        )
        loss_dict = {
            "adver_loss": self.config.gamma * adver_loss,
            "aggregated_sub_losses": aggregated_sub_losses,
            "paired_loss": self.config.delta_pair * paired_loss,
            "class_loss": self.config.delta_class * class_loss,
        }
        loss_dict.update(sub_losses)
        return total_loss, loss_dict

    def _calc_class_loss(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
        is_training: bool,
    ) -> torch.Tensor:
        """
        Optimized, vectorized version of the original _calc_class_loss.

        Notes:
        - Handles arbitrary (hashable) class labels (e.g., strings or ints).
        - Groups latents by label across modalities without looping over samples.
        - Initializes new class means from the batch means (detached).
        - Updates existing class means using in-place EMA under torch.no_grad().
        - Preserves original return semantics (avg across modalities that have metadata).
        """

        if not self.config.class_param:
            # choose a device if possible
            first_mp = next(iter(modality_dynamics.values()), None)
            device = (
                first_mp["mp"].latentspace.device
                if first_mp is not None
                else torch.device("cpu")
            )
            return torch.tensor(0.0, device=device)

        any_mp = next(iter(modality_dynamics.values()))
        device = any_mp["mp"].latentspace.device

        class_means_dict = (
            self.class_means_train if is_training else self.class_means_valid
        )

        # 1) Build global lists: all_latents (tensor), all_labels (python list), and modality slices
        latents_list = []
        labels_list = []  # python objects (strings/ints)
        modality_slices = {}  # mod_name -> (start_idx, end_idx)
        start = 0

        for mod_name, mod_data in batch.items():
            sample_ids: Optional[List[str]] = mod_data.get("sample_ids")
            mod_labels: Optional[List[str]] = mod_data.get("class_labels")
            if not sample_ids or not mod_labels:
                import warnings

                warnings.warn(f"No metadata for modality {mod_name}")
                continue
            self.sample_to_class_map.update(
                {
                    sample_id: mod_label
                    for sample_id, mod_label in zip(sample_ids, mod_labels)
                }
            )
            latents = modality_dynamics[mod_name]["mp"].latentspace  # (N_mod, D)
            n_mod = latents.shape[0]


            if len(mod_labels) != n_mod:
                raise ValueError(
                    f"Mismatch between number of latents ({n_mod}) and labels ({len(mod_labels)}) for modality {mod_name}"
                )

            latents_list.append(latents)
            labels_list.extend(mod_labels)
            modality_slices[mod_name] = (start, start + n_mod)
            start += n_mod

        if len(latents_list) == 0:
            return torch.tensor(0.0, device=device)
        all_latents = torch.cat(latents_list, dim=0)  # shape (N_total, D)

        # 2) Determine unique labels (preserve order) - small (<=20) so Python-level unique is fine
        unique_labels = list(
            dict.fromkeys(labels_list)
        )  # preserves first-occurrence order

        # 3) Compute batch means per label (vectorized per label w/o per-sample loops)
        batch_means: Dict[Any, torch.Tensor] = {}
        for lbl in unique_labels:
            # find indices for this label (python-level)
            # create a tensor of indices for selection to avoid many small GPU->CPU ops
            indices = [i for i, l in enumerate(labels_list) if l == lbl]
            if len(indices) == 0:
                continue
            idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            # gather latents and compute mean
            lbl_latents = all_latents.index_select(0, idx_tensor)
            batch_mean = lbl_latents.mean(dim=0)
            batch_means[lbl] = batch_mean  # kept on device

        # 4) Initialize new classes if needed (mean of current batch samples)
        for lbl, bmean in batch_means.items():
            if lbl not in class_means_dict:
                # store detached clone to avoid accidental linkage to computation graph
                class_means_dict[lbl] = bmean.detach().clone()

        # 5) Compute loss per modality using current class means
        total_class_loss = torch.tensor(0.0, device=device)
        num_modalities_with_metadata = 0

        for mod_name, (start_idx, end_idx) in modality_slices.items():
            num_modalities_with_metadata += 1
            latents = all_latents[start_idx:end_idx]  # view into concatenated tensor
            mod_labels = labels_list[start_idx:end_idx]

            # Build target_means_tensor by stacking class_means_dict entries (ensures device alignment)
            # This is vectorized: (N_mod, D)
            target_means = torch.stack(
                [class_means_dict[lbl].to(device) for lbl in mod_labels], dim=0
            )

            distances = torch.linalg.norm(latents - target_means, dim=1)

            # Apply the configured reduction to distances
            total_class_loss = total_class_loss + self.reduction_fn(distances)

        # Average across modalities (same semantic as original)
        avg_class_loss = (
            total_class_loss / num_modalities_with_metadata
            if num_modalities_with_metadata > 0
            else total_class_loss
        )

        # 6) If training, update class means using EMA with batch statistics (in-place, no_grad)
        # if is_training:
        with torch.no_grad():
            for lbl, bmean in batch_means.items():
                if lbl in class_means_dict:
                    # ensure same device, then update in-place
                    cm = class_means_dict[lbl]
                    if cm.device != bmean.device:
                        cm = cm.to(bmean.device)
                        class_means_dict[lbl] = cm
                    cm.mul_(self.class_mean_momentum).add_(
                        bmean, alpha=(1 - self.class_mean_momentum)
                    )
                else:
                    # this branch shouldn't run due to initialization above, but keep for safety
                    class_means_dict[lbl] = bmean.detach().clone()

        return avg_class_loss

    def _calc_paired_loss(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the paired loss across modalities in the current batch.
        This method prepares latent spaces and sample IDs for each modality and
        computes a paired loss if at least two modalities are present. If fewer
        than two modalities exist, it returns a zero tensor that still requires
        gradients to preserve the computation graph.
        Args:
            batch: A dictionary mapping modality names to modality data.
                Each entry contains sample identifiers under the key
                `"sample_ids"`.
            modality_dynamics: A dictionary mapping modality names to their
                dynamics, where each entry contains a `"mp"` object with a
                `latentspace` tensor.
        Returns:
            A scalar tensor representing the paired loss. Returns a zero tensor
            requiring gradients if fewer than two modalities are available.
        """
        latentspaces = {
            mod_name: dynamics["mp"].latentspace
            for mod_name, dynamics in modality_dynamics.items()
        }
        sample_ids = {
            mod_name: mod_data["sample_ids"] for mod_name, mod_data in batch.items()
        }
        if len(latentspaces) < 2:
            # Return a zero tensor that requires gradients to avoid issues in the graph
            # Assuming at least one tensor exists to get the device
            any_latent_tensor = next(iter(latentspaces.values()))
            return torch.tensor(
                0.0, device=any_latent_tensor.device, requires_grad=True
            )
        return self.compute_paired_loss(
            latentspaces=latentspaces, sample_ids=sample_ids
        )

    def _combine_sub_losses(
        self, modality_dynamics: Dict[str, Dict[str, Any]]
    ) -> torch.Tensor:
        """Combines the sub losses total loss for all modalities."""
        losses = [helper["loss"] for helper in modality_dynamics.values()]
        if self.config.loss_reduction == "mean":
            return torch.stack(losses).mean()
        return torch.stack(losses).sum()

    def _store_sub_losses(
        self, modality_dynamics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        sub_losses = {}
        for k, v in modality_dynamics.items():
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        new_key = f"{k}.{k3}"
                        sub_losses[new_key] = v3.item()
                if k2 == "loss":
                    sub_losses[f"{k}.loss"] = v2.item()
        return sub_losses

    def _calc_adversial_loss(
        self,
        labels: torch.Tensor,
        clf_scores: torch.Tensor,
        clf_loss_fn: torch.nn.Module,
    ):
        flipped_labels = flip_labels(labels=labels)
        adversarial_loss = clf_loss_fn(clf_scores, flipped_labels)
        return adversarial_loss
