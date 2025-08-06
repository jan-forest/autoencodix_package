import torch
from typing import Optional, Type, Union, Tuple, Dict, Any
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class OntixTrainer(GeneralTrainer):
    """
    Specialized trainer for Ontix (ontology-based) autoencoders.
    Handles ontology-specific weight masking and positive weight constraints.
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
