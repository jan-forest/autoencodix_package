import torch
from typing import Tuple, Optional, Union, Dict
import pandas as pd
import numpy as np

from autoencodix.base._base_loss import BaseLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig


class SupervisixLoss(BaseLoss):
    """
    Implements loss for supervised variational autoencoder with unified interface.
    
    Attributes:
        config: Configuration object
    """

    def __init__(self, config: DefaultConfig, annealing_scheduler=None):
        """
        Inits SupervisixLoss.

        Args:
            config: Configuraion object.Any
            annealing_scheduler: Enables passing a custom annealer class, defaults to our implementation of an annealer
        """
        super().__init__(config, annealing_scheduler=annealing_scheduler)

    def _compute_losses(
        self, 
        model_output: ModelOutput, 
        targets: torch.Tensor,
        sample_ids: Tuple[str],
        metadata: Union[pd.DataFrame, pd.Series],
        last_epoc_class_means: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction, variational and class-based losses.

        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            targets: original data to compare with reconstruction

        Returns:
            Tuple of torch.Tensors: reconstruction loss, variational loss, class separation loss and 
            class cohesion loss
        """
        # Get random folating-point sample from a standard normal distribution in the
        # shape of batch_size x Latent_dim
        true_samples = torch.randn( 
            self.config.batch_size, self.config.latent_dim, requires_grad=False
        )

        recon_loss = self.recon_loss(model_output.reconstruction, targets)
        # compute_variational_loss uses true_samples when the configured variational
        # loss is set to MMD -> true_samples represents samples from the prior distribution
        # and compares it to z (latent codes produced by the endcoder)
        # Mismatch ist then penalized. This encourages encoder latents to look like the Gaussian prior
        var_loss = self.compute_variational_loss(
            mu=model_output.latent_mean,
            logvar=model_output.latent_logvar,
            z=model_output.latentspace,
            true_samples=true_samples,
        )

        class_sep_loss = self._compute_class_separation_loss(
            model_output=model_output,
            sample_ids=sample_ids,
            metadata=metadata,
            last_epoch_class_means=last_epoc_class_means
        )

        class_cohesion_loss = self._compute_class_cohesion_loss(
            model_output=model_output,
            sample_ids=sample_ids,
            metadata=metadata,
            last_epoch_class_means=last_epoc_class_means
        )

        return recon_loss, var_loss, class_sep_loss, class_cohesion_loss

    def _distance_latent(
        self,
        latent_a,
        latent_b,
    ) -> torch.Tensor:
        """
        Function as defined in utils.py of the autoencodix pipeline package.

        Computes the euclidean distance between two vectors in the latent space, then
        calculates the mean of the absolute differences across all latent dimensions and
        applies the reduction function (mean or sum) to get a single value representing
        the distance between the two vectors.

        Args:
            latent_a: The first latent vector.
            latent_b: The second latent vector.
        
        Returns:
            torch.Tensor: A single value representing the distance between the two latent vectors.
        """
        # Calculate vector a - vector b, get the absoltue value and apply the reduction function
        return self.reduction_fn( # Mean or sum across samples
            torch.mean(torch.abs(latent_a - latent_b), dim=1) # Mean across latent dimensions
        )
        
    def _compute_class_separation_loss(
            self,
            model_output: ModelOutput,
            sample_ids: Tuple[str],
            metadata: Union[pd.DataFrame, pd.Series],
            last_epoch_class_means: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute class separation loss.

        Formula: L_sep = ((-||latents samples class 1 - latent mean class 2|| - ||latents samples class 1 - latent mean class 3|| - ...) / n_classes) 
                         - ((-||latents samples class 2 - latent mean class 1|| - ||latents samples class 2 - latent mean class 3|| - ...) / n_classes)
        
        The latent mean of a class is calcuated by summing up the mean value over
        samples of a group per latent dimension. For this the mean of the last
        training epoch per class is used, if available. If not, the loss for this class is set to 0.

        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            sample_ids: tuple of sample IDs corresponding to the batch, used to identify class labels
            metadata: DataFrame or Series containing metadata for each sample
            last_epoch_class_means: Dictionary containing the mean latent representations for each class from the last epoch.

        Returns:
            torch.Tensor: class separation loss

        NOTE: Averaging over number of classes twice OK? -> Yes
        """
        # Get the column name that contains the class labels
        class_col = self.config.class_param # Has to be set in config

        # NOTE: In config auslagern
        if class_col is None:
            raise ValueError("config.class_param must be set for Supervisix losses")

        # Get the class labels for all classes in the data set
        classes_all = metadata.loc[:, class_col].unique()

        # Get the class labels for all samples in the batch
        class_labels_batch = metadata.loc[list(sample_ids), class_col].to_numpy()

        # Get the classes and the class indexes
        classes_batch, group_idxs = np.unique(class_labels_batch, return_inverse=True)

        # Get the latent representation of the samples in the batch
        latent_representation = model_output.latentspace

        # Set up class separation loss for the batch
        class_sep_loss_batch = torch.zeros((), device=latent_representation.device, 
                                           dtype=latent_representation.dtype)

        for class_label_a, idx in zip(classes_batch, np.unique(group_idxs)):
            # Get indices for samples belonging to current group/class
            class_a_idxs = group_idxs == idx

            # Get latent space representation for current class
            class_a_latent = latent_representation[class_a_idxs]

            # Get classes to iterate over for distance calcuation
            # Takes classes missing in current batch into account
            classes = classes_batch if len(classes_batch) == len(classes_all) else classes_all

            # Set loss for class a
            class_a_loss = torch.zeros((), device=latent_representation.device, 
                                       dtype=latent_representation.dtype)

            for class_label_b in classes:
                if class_label_b != class_label_a:
                    # Calculate distance between class a latent representation and
                    # last epoch class mean of class b
                    if class_label_b in last_epoch_class_means.keys():
                        class_a_loss = class_a_loss - self._distance_latent(latent_a=class_a_latent, 
                                                                            latent_b=last_epoch_class_means[class_label_b])

            # Average over all classes in the dataset - 1 before adding to the batch loss
            class_sep_loss_batch = class_sep_loss_batch + (class_a_loss / (len(classes_all) - 1)) 

        return class_sep_loss_batch / len(classes_all) # Average over all classes in the data set

    def _compute_class_cohesion_loss(
            self,
            model_output: ModelOutput,
            sample_ids: Tuple[str],
            metadata: Union[pd.DataFrame, pd.Series],
            last_epoch_class_means: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute class cohesion loss.

        Args:
            model_output: custom class that stores model output like latentspaces and reconstructions.
            sample_ids: tuple of sample IDs corresponding to the batch, used to identify class labels
            metadata: DataFrame or Series containing metadata for each sample
            last_epoch_class_means: Dictionary containing the mean latent representations for each class from the last epoch.

        Returns:
            torch.Tensor: class cohesion loss

        TODO: Deviding through number of classes in dataset OK? Or should it be number of classes in batch?
        """
        # Get the column name that contains the class labels
        class_col = self.config.class_param # Has to be set in config

        if class_col is None:
            raise ValueError("config.class_param must be set for Supervisix losses")
        
        # Get the class labels for all classes in the data set
        classes_all = metadata.loc[:, class_col].unique()

        # Get the class labels for all samples in the batch
        class_labels_batch = metadata.loc[list(sample_ids), class_col].to_numpy()

        # Get the classes and the class indexes
        classes_batch, group_idxs = np.unique(class_labels_batch, return_inverse=True)

        # Get the latent representation of the samples in the batch
        latent_representation = model_output.latentspace

        # Set up class cohesion loss for the batch
        class_cohesion_loss_batch = torch.zeros((), device=latent_representation.device, 
                                           dtype=latent_representation.dtype)

        for class_label, idx in zip(classes_batch, np.unique(group_idxs)):
            if class_label in last_epoch_class_means.keys():
                # Get indices for samples belonging to current group/class
                class_idxs = group_idxs == idx

                # Get latent space representation for current class
                class_latent = latent_representation[class_idxs]

                # Get loss for class a
                class_loss = self._distance_latent(latent_a=class_latent, 
                                                   latent_b=last_epoch_class_means[class_label])

                # Add current class loss to batch loss
                class_cohesion_loss_batch = class_cohesion_loss_batch + class_loss

        return class_cohesion_loss_batch / len(classes_all) # Average over all classes in the data set

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        sample_ids: Tuple[str], # Is torch.LongTensor of ints (row indices) if data has no real sample IDs
        metadata: Union[pd.DataFrame, pd.Series],
        last_epoch_class_means: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        """
        # NOTE: Add handling for sample_ids int torch case?
        recon_loss, var_loss, class_sep_loss, class_cohesion_loss = self._compute_losses(model_output, 
                                                                                         targets, 
                                                                                         sample_ids, 
                                                                                         metadata, 
                                                                                         last_epoch_class_means)

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

        total_loss = recon_loss + effective_beta * var_loss + self.config.gamma_class_separation * class_sep_loss + self.config.delta_class_cohesion * class_cohesion_loss

        # NOTE: Do I need to have gamma and delta in there?
        return total_loss, {
            "recon_loss": recon_loss,
            "var_loss": var_loss * effective_beta,
            "class_sep_loss": class_sep_loss * self.config.gamma_class_separation,
            "class_cohesion_loss": class_cohesion_loss * self.config.delta_class_cohesion,
            "anneal_factor": torch.tensor(anneal_factor),
            "effective_beta_factor": torch.tensor(effective_beta),
            #"gamma_factor": torch.tensor(self.config.gamma_class_separation),
            #"delta_factor": torch.tensor(self.config.delta_class_cohesion),
        }
