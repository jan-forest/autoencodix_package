from typing import Dict, Optional, Type, Union, List
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_loss import BaseLoss
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._model_output import ModelOutput


class StackixTrainer(GeneralTrainer):
    """
    StackixTrainer is a wrapper class that conforms to the BaseTrainer interface
    while internally using the StackixOrchestrator and StackixModalityTrainer to perform the
    actual training process.

    This trainer maintains compatibility with the BasePipeline interface while
    leveraging the more modular and well-designed StackixOrchestrator and StackixModalityTrainer
    classes for the actual implementation.

    Attributes
    ----------
    _orchestrator : StackixOrchestrator
        The orchestrator that manages modality model training and latent space preparation
    _modality_trainer : Optional[StackixModalityTrainer]
        The trainer for the final stacked model (created during training)
    _workdir : str
        Directory for saving intermediate models and results
    """

    def __init__(
        self,
        trainset: Optional[StackixDataset],
        validset: Optional[StackixDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        workdir: str = "./stackix_work",
    ):
        """
        Initialize the StackixTrainer with datasets and configuration.

        Parameters
        ----------
        trainset : Optional[StackixDataset]
            Training dataset containing multiple modalities
        validset : Optional[StackixDataset]
            Validation dataset containing multiple modalities
        result : Result
            Result object to store training outcomes
        config : DefaultConfig
            Configuration parameters for training and model architecture
        model_type : Type[BaseAutoencoder]
            Type of autoencoder model to use for each modality
        loss_type : Type[BaseLoss]
            Type of loss function to use for training
        dataset_class : Type[BaseDataset], optional
            Class to use for creating datasets (default is NumericDataset)
        workdir : str, optional
            Directory to save intermediate models and results (default is "./stackix_work")
        """
        # Initialize the parent class
        self._workdir = workdir
        self.result = result
        self._config = config
        self._model_type = model_type
        self._loss_type = loss_type
        self._trainset = trainset
        self._validset = validset

        # Import the classes here to avoid circular imports
        from autoencodix.trainers._stackix_orchestrator import StackixOrchestrator
        from autoencodix.trainers._stackix_modality_trainer import (
            StackixModalityTrainer,
        )

        # Keep references to the actual implementation classes
        self._orchestrator_class = StackixOrchestrator
        self._modality_trainer_class = StackixModalityTrainer

        # Initialize the orchestrator
        self._orchestrator = self._orchestrator_class(
            trainset=trainset,
            validset=validset,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            workdir=workdir,
        )

        # Modality trainer will be created during the training process
        self._modality_trainer = None

    def train(self) -> Result:
        """
        Train the Stackix model using the orchestrator and trainer.

        This method orchestrates the complete training process:
        1. Train individual modality models
        2. Extract and concatenate latent spaces
        3. Train the final stacked model

        Returns
        -------
        Result
            Training results including losses, latent spaces, and other metrics
        """
        # Step 1: Train individual modality models
        modality_models = self._orchestrator.train_modalities()

        # Step 2: Prepare concatenated latent space datasets
        train_latent_ds, valid_latent_ds = self._orchestrator.prepare_latent_datasets()
        modality_dimensions = self._orchestrator.get_modality_dimensions()

        # Step 3: Create and train the stacked model
        self._modality_trainer = self._modality_trainer_class(
            trainset=train_latent_ds,
            validset=valid_latent_ds,
            config=self._config,
            model_type=self._model_type,
            loss_type=self._loss_type,
            modality_models=modality_models,
            modality_dimensions=modality_dimensions,
        )

        # Train the stacked model
        final_result = self._modality_trainer.train()

        # Store a reference to the model in the result
        self._result = final_result
        self._model = self._modality_trainer._model

        # Store modality models in the result for reference
        self._result.modality_models = modality_models
        self._result.modality_dimensions = modality_dimensions

        return self._result

    def predict(self, data: Union[StackixDataset, torch.Tensor]) -> Result:
        """
        Generate predictions using the trained model.

        Parameters
        ----------
        data : Union[StackixDataset, torch.Tensor]
            Input data for prediction, either a multi-modal dataset or concatenated latent tensor

        Returns
        -------
        Result
            Prediction metrics

        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        if self._modality_trainer is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        return self._modality_trainer.predict_with_reconstruction(data)

    def _capture_dynamics(self, epoch: int, model_output:List[ModelOutput], split: str) -> None:
        return super()._capture_dynamics(epoch, model_output, split)
