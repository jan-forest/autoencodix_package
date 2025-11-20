from abc import abstractmethod, ABC
import itertools
from autoencodix.configs.default_config import DefaultConfig
import torch
from torch import nn
from autoencodix.utils._annealer import AnnealingScheduler

from typing import Optional, Any


class BaseLoss(nn.Module, ABC):
    """Provides common loss computation functionality for autoencoders.

    Implements standard loss calculations including reconstruction loss,
    KL divergence, and Maximum Mean Discrepancy (MMD), while requiring
    subclasses to implement the specific forward method.

    Attributes:
        config: Configuration parameters for the loss function.
        recon_loss: Module for computing reconstruction loss (MSE or BCE).
        reduction_fn: Function to apply reduction (mean or sum).
        compute_kernel: Function to compute kernel for MMD loss.
        annealing_scheduler: Helper for loss calculation with annealing.
    """

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        """Initializes the loss module with the specified configuration.

        Args:
            config: Configuration parameters for the loss function.
            annealing_scheduler: Helper class for loss calculation with annealing.

        Raises:
            NotImplementedError: If unsupported loss reduction or reconstruction
                loss type is specified.
        """
        super().__init__()
        self.annealing_scheduler = annealing_scheduler or AnnealingScheduler()
        self.config = config
        self.recon_loss: nn.Module

        if self.config.loss_reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.config.loss_reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            raise NotImplementedError(
                f"Invalid loss reduction type: {self.config.loss_reduction}. "
                f"Only 'mean' and 'sum' are supported."
            )

        if self.config.reconstruction_loss == "mse":
            self.recon_loss = nn.MSELoss(reduction=config.loss_reduction)
        elif self.config.reconstruction_loss == "bce":
            self.recon_loss = nn.BCEWithLogitsLoss(reduction=config.loss_reduction)
        else:
            raise NotImplementedError(
                f"Invalid reconstruction loss type: {self.config.reconstruction_loss}. "
                f"Only 'mse' and 'bce' are supported. Please check the value of "
                f"'config.reconstruction_loss' for typos or unsupported types."
            )

        self.compute_kernel = self._mmd_kernel

    def _mmd_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian kernel for Maximum Mean Discrepancy calculation.

        Calculates the kernel matrix between two sets of samples, using a
        Gaussian kernel with normalization by feature dimension.

        Args:
            x: First set of input samples.
            y: Second set of input samples.

        Returns:
            Kernel matrix of shape (x.shape[0], y.shape[0]).
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)

        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)

    def compute_mmd_loss(
        self, z: torch.Tensor, true_samples: torch.Tensor
    ) -> torch.Tensor:
        """Computes Maximum Mean Discrepancy loss.

        Args:
            z: Samples from the encoded distribution.
            true_samples: Samples from the prior distribution.

        Returns:
            The MMD loss value.

        Raises:
            NotImplementedError: If unsupported loss reduction type is specified.
        """
        true_samples_kernel = self.compute_kernel(x=true_samples, y=true_samples)
        z_device = z.device
        true_samples = true_samples.to(z_device)
        z_kernel = self.compute_kernel(z, z)
        ztr_kernel = self.compute_kernel(x=true_samples, y=z)

        if self.config.loss_reduction == "mean":
            return true_samples_kernel.mean() + z_kernel.mean() - 2 * ztr_kernel.mean()
        elif self.config.loss_reduction == "sum":
            return true_samples_kernel.sum() + z_kernel.sum() - 2 * ztr_kernel.sum()
        else:
            raise NotImplementedError(
                f"Invalid loss reduction type: {self.config.loss_reduction}. "
                f"Only 'mean' and 'sum' are supported."
            )

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Computes KL divergence loss between N(mu, logvar) and N(0, 1).

        Args:
            mu: Mean tensor.
            logvar: Log variance tensor.

        Returns:
            The KL divergence loss value.

        Raises:
            ValueError: If mu and logvar do not have the same shape.
        """
        if mu.shape != logvar.shape:
            raise ValueError(
                f"Shape mismatch: mu has shape {mu.shape}, but logvar has shape {logvar.shape}."
            )
        return -0.5 * self.reduction_fn(1 + logvar - mu.pow(2) - logvar.exp())

    def compute_variational_loss(
        self,
        mu: Optional[torch.Tensor],
        logvar: Optional[torch.Tensor],
        z: Optional[torch.Tensor] = None,
        true_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes either KL or MMD loss based on configuration.

        Args:
            mu: Mean tensor for variational loss.
            logvar: Log variance tensor for variational loss.
            z: Encoded samples for MMD loss.
            true_samples: Prior samples for MMD loss.

        Returns:
            The computed variational loss.

        Raises:
            ValueError: If required parameters are missing or if mu and logvar have shape mismatch.
            NotImplementedError: If unsupported VAE loss type is specified.
        """

        if self.config.default_vae_loss == "kl":
            if mu is None:
                raise ValueError("mu must be provided for VAE loss")
            if logvar is None:
                raise ValueError("logvar must be provided for VAE loss")
            if mu.shape != logvar.shape:
                raise ValueError(
                    f"Shape mismatch: mu has shape {mu.shape}, but logvar has shape {logvar.shape}"
                )

            return self.compute_kl_loss(mu=mu, logvar=logvar)

        elif self.config.default_vae_loss == "mmd":
            if z is None:
                raise ValueError("z must be provided for MMD loss")
            if true_samples is None:
                raise ValueError("true_samples must be provided for MMD loss")
            return self.compute_mmd_loss(z=z, true_samples=true_samples)
        else:
            raise NotImplementedError(
                f"VAE loss type {self.config.default_vae_loss} is not implemented. "
                f"Only 'kl' and 'mmd' are supported."
            )

    def compute_paired_loss(
        self,
        latentspaces: dict[str, torch.Tensor],
        sample_ids: dict[str, list],
    ) -> torch.Tensor:
        """
        Calculates the paired distance loss across all pairs of modalities in a batch.

        Args:
            latentspaces: A dictionary mapping modality names to their latent space tensors.
                        e.g., {'RNA': tensor_rna, 'ATAC': tensor_atac}
            sample_ids: A dictionary mapping modality names to their list of sample IDs.

        Returns:
            A single scalar tensor representing the total paired loss.
        """

        loss_helper = []
        modality_names = list(latentspaces.keys())

        # 1. Iterate through all unique pairs of modalities
        for mod_a, mod_b in itertools.combinations(modality_names, 2):
            ids_a = sample_ids[mod_a]
            ids_b = sample_ids[mod_b]

            # 2. Find the intersection of sample IDs
            common_ids = set(ids_a) & set(ids_b)

            if not common_ids:
                print("no common ids")
                continue

            # 3. Create a mapping from sample ID to index for efficient lookup
            id_to_idx_a = {sample_id: i for i, sample_id in enumerate(ids_a)}
            id_to_idx_b = {sample_id: i for i, sample_id in enumerate(ids_b)}

            # Get the corresponding indices for the common samples
            indices_a = [id_to_idx_a[common_id] for common_id in common_ids]
            indices_b = [id_to_idx_b[common_id] for common_id in common_ids]

            # 4. Select the latent vectors for the paired samples
            paired_latents_a = latentspaces[mod_a][indices_a]
            paired_latents_b = latentspaces[mod_b][indices_b]

            # 5. Calculate the distance between the aligned latent vectors

            distance = torch.linalg.norm(paired_latents_a - paired_latents_b, dim=1)
            pair_loss = self.reduction_fn(distance)
            loss_helper.append(pair_loss)
        if not loss_helper:
            return torch.tensor(0.0)
        if self.config.loss_reduction == "mean":
            return torch.stack(loss_helper).mean()
        return torch.stack(loss_helper).sum()

    @staticmethod
    def _compute_log_gauss_dense(
        z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Computes the log probability of a Gaussian distribution.

        Args:
            z: Latent variable tensor.
            mu: Mean tensor.
            logvar: Log variance tensor.

        Returns:
            Log probability of the Gaussian distribution.
        """
        return -0.5 * (
            torch.log(torch.tensor([2 * torch.pi]).to(z.device))
            + logvar
            + (z - mu) ** 2 * torch.exp(-logvar)
        )

    @staticmethod
    def _compute_log_import_weight_mat(batch_size: int, n_samples: int) -> torch.Tensor:
        """Computes the log import weight matrix for disentangled loss.
           Similar to: https://github.com/rtqichen/beta-tcvae
        Args:
            batch_size: Number of samples in the batch.
            n_samples: Total number of samples in the dataset.

        Returns:
            Log import weight matrix of shape (batch_size, n_samples).
        """

        N = n_samples
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Calculates the loss for the autoencoder.

        This method must be implemented by subclasses to define the specific
        loss computation logic for the autoencoder. The implementation should
        compute the total loss as well as any individual loss components
        (e.g., reconstruction loss, KL divergence, etc.) based on the model's
        output and the provided targets.

        Args:
            *kwargs depending on the loss type and pipeline


        Returns:
            - The total loss value as a scalar tensor.
            - A dictionary of individual loss components, where the keys are
                descriptive strings (e.g., "reconstruction_loss", "kl_loss") and
                the values are the corresponding loss tensors.
            - Implementation in subclasses is flexible, so for new loss classes this can differ.

        Note:
            Subclasses must implement this method to define the specific loss
            computation logic for their use case.
        """
        # TODO maybe standardize the return types more i.e. request a scalar and a dict
        pass
