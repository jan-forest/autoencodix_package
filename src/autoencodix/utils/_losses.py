import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict
from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._utils import flip_labels


class VanillixLoss(BaseLoss):
    """Loss function for vanilla autoencoder."""

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        **kwargs,
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
        **kwargs,
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
            "var_loss": var_loss * effective_beta,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
        }


class XModalLoss(BaseLoss):
    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        super().__init__(config)

        self.class_means_train: Dict[str, torch.Tensor] = {}
        self.class_means_valid: Dict[str, torch.Tensor] = {}
        self.sample_to_class_map: Dict[str, Any] = {}

    def forward(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
        clf_scores: torch.Tensor,
        labels: torch.Tensor,
        clf_loss_fn: torch.nn.Module,
        is_training: bool = True,
        **kwargs,
    ):
        adver_loss = self._calc_adversial_loss(
            labels=labels, clf_loss_fn=clf_loss_fn, clf_scores=clf_scores
        )
        aggregated_sub_losses = self._sum_sub_losses(
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
            "total_loss": total_loss,
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
        device = next(iter(modality_dynamics.values()))["mp"].latentspace.device

        if not self.config.class_param:
            return torch.tensor(0.0, device=device)
        class_means_dict = (
            self.class_means_train if is_training else self.class_means_valid
        )
        # Step 1: Dynamically update maps and discover new classes from the batch
        for mod_name, mod_data in batch.items():
            metadata_df = mod_data.get("metadata")
            if metadata_df is None:
                continue

            batch_map = metadata_df[self.config.class_param]
            self.sample_to_class_map.update(batch_map.to_dict())

            for label in batch_map.unique():
                if label not in class_means_dict:
                    class_means_dict[label] = torch.zeros(
                        self.config.latent_dim, device=device
                    )

        # Step 2: Calculate the loss for the current batch
        total_class_loss = torch.tensor(0.0, device=device)
        for mod_name, mod_data in batch.items():
            if mod_data.get("metadata") is None:
                continue

            latents = modality_dynamics[mod_name]["mp"].latentspace
            metadata_df = mod_data["metadata"]
            batch_class_labels = metadata_df[self.config.class_param].tolist()

            target_means_list = [
                self.class_means_train[label] for label in batch_class_labels
            ]
            target_means_tensor = torch.stack(target_means_list).to(latents.device)

            distance = torch.abs(latents - target_means_tensor).mean(dim=1)
            total_class_loss += self.reduction_fn(distance)

        num_modalities_in_batch = len(batch)
        return (
            total_class_loss / num_modalities_in_batch
            if num_modalities_in_batch > 0
            else total_class_loss
        )

    def update_class_means(
        self, epoch_dynamics: List[Dict], device: str, is_training: bool
    ):
        """
        Method to be called by the Trainer at the end of an epoch to update the
        target class mean vectors.
        """
        if not self.config.class_param or not epoch_dynamics:
            return
        class_means_dict = (
            self.class_means_train if is_training else self.class_means_valid
        )
        final_latents = defaultdict(list)
        final_sample_ids = defaultdict(list)
        for batch_data in epoch_dynamics:
            for mod_name, data in batch_data["latentspaces"].items():
                final_latents[mod_name].append(data)
            for mod_name, data in batch_data["sample_ids"].items():
                final_sample_ids[mod_name].append(data)

        all_latents_df_list = []
        for mod_name in final_latents.keys():
            mod_latents = np.concatenate(final_latents[mod_name])
            mod_ids = np.concatenate(final_sample_ids[mod_name])
            all_latents_df_list.append(pd.DataFrame(mod_latents, index=mod_ids))

        if not all_latents_df_list:
            return

        all_latents_df = pd.concat(all_latents_df_list)
        all_latents_df["class_label"] = all_latents_df.index.map(
            self.sample_to_class_map
        )

        new_means_df = all_latents_df.groupby("class_label").mean()

        for label, mean_values in new_means_df.iterrows():
            class_means_dict[label] = torch.tensor(
                mean_values.values, dtype=torch.float32, device=device
            )

    def _calc_paired_loss(
        self,
        batch: Dict[str, Dict[str, Any]],
        modality_dynamics: Dict[str, Dict[str, Any]],
    ) -> torch.Tensor:
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

    def _sum_sub_losses(
        self, modality_dynamics: Dict[str, Dict[str, Any]]
    ) -> torch.Tensor:
        """Computes the average total loss for all modalities."""
        losses = [helper["loss"] for helper in modality_dynamics.values()]
        # return torch.stack(losses).mean()
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


class DisentanglixLoss(BaseLoss):
    """Loss function for VAE with disentanglement, Disentanglix."""

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        super().__init__(config)

        # Determine the forward function strategy at initialization
        if self.config.anneal_function == "no-annealing":
            self._forward_impl = self._forward_without_annealing
        else:
            self._forward_impl = self._forward_with_annealing

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        epoch: int,
        n_samples: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Unified forward interface - delegates to appropriate implementation."""
        return self._forward_impl(
            model_output=model_output, targets=targets, n_samples=n_samples, epoch=epoch
        )

    def _compute_losses(
        self, model_output: ModelOutput, targets: torch.Tensor, n_samples: int
    ) -> Tuple[torch.Tensor, ...]:
        """Compute reconstruction, mutual information, total correlation and dimension-wise KL loss terms."""
        # true_samples = torch.randn(
        #     self.config.batch_size, self.config.latent_dim, requires_grad=False
        # )

        recon_loss: torch.Tensor = self.recon_loss(model_output.reconstruction, targets)
        mut_info_loss, tot_corr_loss, dimwise_kl_loss = (
            self._compute_decomposed_vae_loss(
                z=model_output.latentspace,  # Latent space of batch samples (shape: [batch_size, latent_dim])
                mu=model_output.latent_mean,  # Mean of latent space (shape: [batch_size, latent_dim])
                logvar=model_output.latent_logvar,  # Log variance of latent space (shape: [batch_size, latent_dim])
                n_samples=n_samples,  # Number of samples of whole dataset
                use_mss=self.config.use_mss,
            )
        )
        # Clip losses to avoid negative values
        mut_info_loss: torch.Tensor = torch.clamp(mut_info_loss, min=0.0)
        tot_corr_loss: torch.Tensor = torch.clamp(tot_corr_loss, min=0.0)
        dimwise_kl_loss: torch.Tensor = torch.clamp(dimwise_kl_loss, min=0.0)

        return recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss

    def _compute_decomposed_vae_loss(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n_samples: int,
        use_mss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mutual information, total correlation and dimension-wise KL loss terms."""

        log_q_z_given_x = self._compute_log_gauss_dense(z, mu, logvar).sum(
            dim=1
        )  # Dim [batch_size]
        log_prior = self._compute_log_gauss_dense(
            z, torch.zeros_like(z), torch.zeros_like(z)
        ).sum(dim=1)  # Dim [batch_size]

        log_q_batch_perm = self._compute_log_gauss_dense(
            z.reshape(z.shape[0], 1, -1),
            mu.reshape(1, z.shape[0], -1),
            logvar.reshape(1, z.shape[0], -1),
        )  # Dim [batch_size, batch_size, latent_dim]

        if use_mss:
            logiw_mat = self._compute_log_import_weight_mat(z.shape[0], n_samples).to(
                z.device
            )
            log_q_z = torch.logsumexp(
                logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1
            )  # Dim [batch_size]

            log_product_q_z = torch.logsumexp(
                logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm,
                dim=1,
            ).sum(dim=-1)  # Dim [batch_size]
        else:
            log_q_z = torch.logsumexp(log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(
                torch.tensor([z.shape[0] * n_samples]).to(z.device)
            )  # Dim [batch_size]
            log_product_q_z = torch.logsumexp(log_q_batch_perm, dim=1) - torch.log(
                torch.tensor([z.shape[0] * n_samples]).to(z.device)
            ).sum(dim=-1)  # Dim [batch_size]

        mut_info_loss = self.reduction_fn(
            log_q_z_given_x - log_q_z
        )  ## Reduction: mean or sum over batch
        tot_corr_loss = self.reduction_fn(log_q_z - log_product_q_z)
        dimwise_kl_loss = self.reduction_fn(log_product_q_z - log_prior)

        return mut_info_loss, tot_corr_loss, dimwise_kl_loss

    def _forward_without_annealing(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        n_samples: int,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass without annealing - uses constant beta (ignores epoch)."""
        recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss = (
            self._compute_losses(model_output, targets, n_samples)
        )
        total_loss = (
            recon_loss
            + self.config.beta_mi * mut_info_loss
            + self.config.beta_tc * tot_corr_loss
            + self.config.beta_dimKL * dimwise_kl_loss
        )
        return total_loss, {
            "recon_loss": recon_loss,
            "mut_info_loss": mut_info_loss * self.config.beta_mi,
            "tot_corr_loss": tot_corr_loss * self.config.beta_tc,
            "dimwise_kl_loss": dimwise_kl_loss * self.config.beta_dimKL,
        }

    def _forward_with_annealing(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        n_samples: int,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with annealing - calculates annealing factor from epoch."""
        recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss = (
            self._compute_losses(model_output, targets, n_samples)
        )

        # Get annealing weight
        anneal_factor = self.annealing_scheduler.get_weight(
            # epoch_current=annealing_epoch,
            epoch_current=epoch,
            total_epoch=self.config.epochs,
            func=self.config.anneal_function,
        )

        # Apply annealed beta
        effective_beta_mi = self.config.beta_mi * anneal_factor
        effective_beta_tc = self.config.beta_tc * anneal_factor
        effective_beta_dimKL = self.config.beta_dimKL * anneal_factor
        # Calculate total loss
        total_loss = (
            recon_loss
            + effective_beta_mi * mut_info_loss
            + effective_beta_tc * tot_corr_loss
            + effective_beta_dimKL * dimwise_kl_loss
        )

        return total_loss, {
            "recon_loss": recon_loss,
            "mut_info_loss": mut_info_loss * effective_beta_mi,
            "tot_corr_loss": tot_corr_loss * effective_beta_tc,
            "dimwise_kl_loss": dimwise_kl_loss * effective_beta_dimKL,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_mi_factor": torch.tensor(effective_beta_mi),
            "effective_beta_tc_factor": torch.tensor(effective_beta_tc),
            "effective_beta_dimKL_factor": torch.tensor(effective_beta_dimKL),
        }
