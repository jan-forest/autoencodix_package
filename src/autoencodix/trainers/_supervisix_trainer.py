import torch
import numpy as np
from typing import Optional, Type, Dict, List
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput

class SupervisixTrainer(GeneralTrainer):
    """Specialized trainer for Supervisix (supervised) autoencoders.

    Handles retaining and passing on the mean loss per class in the data set for previous epochs. 
    Uses most of the functionality from the GeneralTrainer class and supervisix-specific functionality 
    is added via a hook.

    Attributes:
        Inherits all attributes from GeneralTrainer.
    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        **kwargs,
    ):
        """Initializes the SupervisixTrainer with the given datasets, model, and configuration.


        Args:
            trainset: The dataset used for training.
            validset: The dataset used for validation, if provided.
            result: An object to store and manage training results.
            config: Configuration object containing training hyperparameters and settings.
            model_type: The autoencoder model class to be trained.
            loss_type: The loss function class specific to the model.
        """
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
        )

        self.epoch_class_means_train: Dict[str, torch.Tensor] = {}
        self.epoch_class_means_valid: Dict[str, torch.Tensor] = {}

    def supervisix_capture_hook(
            self, 
            model_output: ModelOutput, 
            sample_ids: List[str], 
            dataset_type: str,
            batch_class_means: Dict[str, torch.Tensor]
    ):
        """Capture latent representations for each class in the dataset per batch in the current epoch.

        Args:
            model_outputs: The output from the model's forward pass.
            sample_ids: The identifiers for the samples in the current batch.
            dataset_type: A string indicating whether the data is from 'train' or 'valid'.
            batch_class_means: A dictionary to store the mean latent representation for each class in the current batch.
            batch: The current batch number.
            dataset_type: A string indicating whether the data is from 'train' or 'valid'.
        """

        # Get the column name that contains the class labels
        class_col = self._config.class_param # Has to be set in config

        if class_col is None:
            raise ValueError("config.class_param must be set for Supervisix losses")
        
        # Get the class labels for all samples in the batch
        class_labels = (self._trainset.metadata.loc[list(sample_ids), class_col].to_numpy() if dataset_type == "train" 
                        else self._validset.metadata.loc[list(sample_ids), class_col].to_numpy())

        # Get the classes and the class indexes
        classes, group_idxs = np.unique(class_labels, return_inverse=True)

        # Get the latent representation of the samples in the batch
        latent_representation = model_output.latentspace

        for class_label, idx in zip(classes, np.unique(group_idxs)):
            # Get the indices of the samples belonging to each class
            class_idx = group_idxs == idx

            # Separate the samples by class
            class_latents = latent_representation[class_idx]

            # Calculate the mean latent representation
            class_mean = class_latents.mean(dim=0).cpu().detach() if self._config.save_vram else class_latents.mean(dim=0).detach()

            if class_label not in batch_class_means.keys():
                batch_class_means[class_label] = class_mean.unsqueeze(0)  # Add a new dimension for stacking
            else:
                batch_class_means[class_label] = torch.cat([batch_class_means[class_label], class_mean.unsqueeze(0)], dim=0)
                
    def supervisix_get_hook(
            self,
            dataset_type: str
    ) -> Dict[str, torch.Tensor]:
        """Get the last epoch class means for the current dataset type.
        Args:
            dataset_type: A string indicating whether the data is from 'train' or 'valid'.

        Returns:
            A dictionary containing the mean latent representations for each class from the last epoch.
            During the first training epoch, this will return an emtpy dictionary, wich is handled by the loss function.
        """
        if dataset_type == "train":
            return {key: value.to(self.device) for key, value in self.epoch_class_means_train.items()}
        elif dataset_type == "valid":
            return {key: value.to(self.device) for key, value in self.epoch_class_means_valid.items()}
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def supervisix_update_hook(
            self, 
            batch_class_means: Dict[str, torch.Tensor], 
            dataset_type: str
    ):
        """Update epoch class means at the end of each epoch.
        
        Args:
            batch_class_means: A dictionary containing the mean latent representations for each class in the current batch.
            dataset_type: A string indicating whether the data is from 'train' or 'valid'.
        """
        # Get per class means per epoch and store them in the epoch_class_means_train or epoch_class_means_valid
        for class_label, means in batch_class_means.items():
            if dataset_type == "train":
                self.epoch_class_means_train[class_label] = means.mean(dim=0)
            elif dataset_type == "valid":
                self.epoch_class_means_valid[class_label] = means.mean(dim=0)
