import torch
from typing import Optional, Type, Union, Tuple, Dict, Any
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class OntixTrainer(GeneralTrainer):
    """Specialized trainer for Ontix (ontology-based) autoencoders.

    Handles ontology-specific weight masking and positive weight constraints. Uses most of the
    functionality from the GeneralTrainer class and ontology-specific functionality is added via a hook
    after the backward pass.

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
        ontologies: Optional[Union[Tuple, Dict[Any, Any]]],
    ):
        """Initializes the OntixTrainer with the given datasets, model, and configuration.


        Args:
            trainset: The dataset used for training.
            validset: The dataset used for validation, if provided.
            result: An object to store and manage training results.
            config: Configuration object containing training hyperparameters and settings.
            model_type: The autoencoder model class to be trained.
            loss_type: The loss function class specific to the model.
            ontologies: Ontology information required for Ontix.
        """
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            ontologies=ontologies,
        )

    def _apply_post_backward_processing(self):
        """Apply ontology-specific processing after backward pass."""
        # Apply positive weight constraint to decoder
        self._model._decoder.apply(self._model._positive_dec)

        # Apply ontology-based weight masking
        with torch.no_grad():
            self._validate_masks()
            self._apply_weight_masks()

    def _validate_masks(self):
        """Validate that number of masks matches decoder layers."""
        if len(self._model.masks) != len(self._model._decoder):
            raise ValueError(
                f"Number of masks ({len(self._model.masks)}) does not match "
                f"number of decoder layers ({len(self._model._decoder)})"
            )

    def _apply_weight_masks(self):
        """Apply weight masks to decoder layers."""
        for i, mask in enumerate(self._model.masks):
            mask = mask.to(self._fabric.device)
            self._model._decoder[i].weight.mul_(mask)
