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
        return total_loss, {"recon_loss": recon_loss, "var_loss": var_loss * self.config.beta}

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
        # print(f"Annealing epoch: {annealing_epoch}")
        # print(f"Total epochs: {self.config.epochs}")
        # print(f"Anneal function: {self.config.anneal_function}")
        # Get annealing weight
        anneal_factor = self.annealing_scheduler.get_weight(
            epoch_current=annealing_epoch,
            total_epoch=self.config.epochs,
            func=self.config.anneal_function,
        )
        # print(f"Anneal factor: {anneal_factor}")

        # Apply annealed beta
        effective_beta = self.config.beta * anneal_factor
        total_loss = recon_loss + effective_beta * var_loss

        return total_loss, {
            "recon_loss": recon_loss,
            "var_loss": var_loss * effective_beta,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
        }

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
            model_output=model_output,
            targets=targets,
            n_samples=n_samples,
            epoch=epoch)

    def _compute_losses(
        self, model_output: ModelOutput, targets: torch.Tensor, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction, mutual information, total correlation and dimension-wise KL loss terms."""
        # true_samples = torch.randn(
        #     self.config.batch_size, self.config.latent_dim, requires_grad=False
        # )

        recon_loss = self.recon_loss(model_output.reconstruction, targets)
        # var_loss = self.compute_variational_loss(
        #     mu=model_output.latent_mean,
        #     logvar=model_output.latent_logvar,
        #     z=model_output.latentspace,
        #     true_samples=true_samples,
        # )
        mut_info_loss, tot_corr_loss, dimwise_kl_loss = self._compute_decomposed_vae_loss(
            z =model_output.latentspace, # Latent space of batch samples (shape: [batch_size, latent_dim])
            mu = model_output.latent_mean, # Mean of latent space (shape: [batch_size, latent_dim])
            logvar = model_output.latent_logvar, # Log variance of latent space (shape: [batch_size, latent_dim])
            n_samples = n_samples, # Number of samples of whole dataset
            use_mss = self.config.use_mss,
        )
        # Clip losses to avoid negative values
        mut_info_loss = torch.clamp(mut_info_loss, min=0.0)
        tot_corr_loss = torch.clamp(tot_corr_loss, min=0.0)
        dimwise_kl_loss = torch.clamp(dimwise_kl_loss, min=0.0)

        return recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss
    
    def _compute_decomposed_vae_loss(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n_samples: int,
        use_mss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mutual information, total correlation and dimension-wise KL loss terms."""

        log_q_z_given_x = self._compute_log_gauss_dense(z, mu, logvar).sum(dim=1) # Dim [batch_size]
        log_prior = self._compute_log_gauss_dense(z,
                                                  torch.zeros_like(z), torch.zeros_like(z)
                                                    ).sum(dim=1)    # Dim [batch_size]
        
        log_q_batch_perm = self._compute_log_gauss_dense(
                        z.reshape(z.shape[0], 1, -1),
                        mu.reshape(1, z.shape[0], -1),
                        logvar.reshape(1, z.shape[0], -1),
        ) # Dim [batch_size, batch_size, latent_dim]

        if use_mss:
            logiw_mat = self._compute_log_import_weight_mat(
                z.shape[0],
                n_samples).to(z.device)
            log_q_z = torch.logsumexp(
                logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1
            ) # Dim [batch_size]

            log_product_q_z = (
                torch.logsumexp(
                logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm,
                dim=1,
            ).sum(dim=-1)
            ) # Dim [batch_size]
        else:
            log_q_z = torch.logsumexp(
                log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(
                    torch.tensor([z.shape[0] * n_samples]).to(z.device)
                ) # Dim [batch_size]
            log_product_q_z = (
                torch.logsumexp(log_q_batch_perm, dim=1) - torch.log(
                    torch.tensor([z.shape[0] * n_samples]).to(z.device)
                ).sum(dim=-1)
            ) # Dim [batch_size]
        

        mut_info_loss = self.reduction_fn(log_q_z_given_x - log_q_z) ## Reduction: mean or sum over batch
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
        recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss = self._compute_losses(model_output, targets, n_samples)
        total_loss =    recon_loss + \
                        self.config.beta_mi * mut_info_loss + \
                        self.config.beta_tc * tot_corr_loss + \
                        self.config.beta_dimKL * dimwise_kl_loss
        return total_loss, {"recon_loss": recon_loss,
                            "mut_info_loss": mut_info_loss * self.config.beta_mi,
                            "tot_corr_loss": tot_corr_loss * self.config.beta_tc,
                            "dimwise_kl_loss": dimwise_kl_loss * self.config.beta_dimKL}

    def _forward_with_annealing(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        n_samples: int,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with annealing - calculates annealing factor from epoch."""
        recon_loss, mut_info_loss, tot_corr_loss, dimwise_kl_loss = self._compute_losses(model_output, targets, n_samples)

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
        effective_beta_mi = self.config.beta_mi * anneal_factor
        effective_beta_tc = self.config.beta_tc * anneal_factor
        effective_beta_dimKL = self.config.beta_dimKL * anneal_factor
        # Calculate total loss
        total_loss = recon_loss + \
                        effective_beta_mi * mut_info_loss + \
                        effective_beta_tc * tot_corr_loss + \
                        effective_beta_dimKL * dimwise_kl_loss
        
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
