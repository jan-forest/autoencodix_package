from typing import Dict, Optional, Tuple, Type

import torch
from lightning_fabric import Fabric

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.trainers._stackix_orchestrator import StackixOrchestrator
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class StackixTrainer(GeneralTrainer):
    """StackixTrainer is a wrapper for StackixOrchestrator that conforms to the BaseTrainer interface.

    This trainer maintains compatibility with the BasePipeline interface while
    leveraging the more modular and well-designed StackixOrchestrator
    classes for the actual implementation.

    Attributes:
    _workdir: Directory for saving intermediate models and results
    _result: Result object to store training outcomes
    _config: Configuration parameters for training and model architecture
    _model_type: Type of autoencoder model to use for each modality
    _loss_type: Type of loss function to use for training
    _trainset: Training dataset containing multiple modalities
    _validset: Validation dataset containing multiple modalities
    _orchestrator_type: Type to use for orchestrating modality training (default is StackixOrchestrator)
    _trainer_type: Type to use for training each modality model (default is GeneralTrainer)
    _modality_trainers: Dictionary of trained models for each modality
    _modality_results: Dictionary of training results for each modality
    _trainer: Trainer for the stacked model
    _fabric: Lightning Fabric wrapper for device and precision management
    _train_latent_ds: Training dataset with concatenated latent spaces
    _valid_latent_ds: Validation dataset with concatenated latent spaces
    concat_idx: Indices used for concatenating latent spaces
    _model: The instantiated stacked model architecture
    _optimizer: The optimizer used for training
    _orchestrator: The orchestrator that manages modality model training and latent space preparation
    _workdir: Directory for saving intermediate models and results
    """

    def __init__(
        self,
        trainset: Optional[StackixDataset],
        validset: Optional[StackixDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        orchestrator_type: Type[StackixOrchestrator] = StackixOrchestrator,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        workdir: str = "./stackix_work",
        ontologies: Optional[Tuple] = None,
        **kwargs,
    ) -> None:
        """Initialize the StackixTrainer with datasets and configuration.

        Args:
            trainset: Training dataset containing multiple modalities
            validset: Validation dataset containing multiple modalities
            result: Result object to store training outcomes
            config: Configuration parameters for training and model architecture
            model_type: Type of autoencoder model to use for each modality
            loss_type: Type of loss function to use for training
            orchestrator_type: Type to use for orchestrating modality training (default is StackixOrchestrator)
            trainer_type: Type to use for training each modality model (default is GeneralTrainer)
            workdir: Directory to save intermediate models and results (default is "./stackix_work")
            onotologies: Ontology information, if provided for Ontix compatibility
        """
        self._workdir = workdir
        self._result = result
        self._config = config
        self._model_type = model_type
        self._loss_type = loss_type
        self._trainset = trainset
        self._validset = validset
        self._orchestrator_type = orchestrator_type
        self._trainer_type = trainer_type

        self._orchestrator = self._orchestrator_type(
            trainset=trainset,
            validset=validset,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            workdir=workdir,
        )
        self._modality_trainers: Optional[Dict[str, BaseAutoencoder]] = None
        self._modality_results: Optional[Dict[str, Result]] = None

        self._fabric = Fabric(
            accelerator=self._config.device,
            devices=self._config.n_gpus,
            precision=self._config.float_precision,
            strategy=self._config.gpu_strategy,
        )

        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
        )

    def get_model(self) -> torch.nn.Module:
        """Getter for the the trained model.

        Returns:
            The trained model
        """
        return self._model

    def train(self) -> Result:
        """Train the stacked model on the concatenated latent space.

        Uses the standard BaseTrainer training process but with the stacked model.

        Returns:
            Training results including losses, latent spaces, and other metrics
        """
        print("Training each modality model...")
        self._train_modalities()
        print("finished training each modality model")
        self._trainer = self._trainer_type(
            trainset=self._train_latent_ds,
            validset=self._valid_latent_ds,
            result=self._result,
            config=self._config,
            model_type=self._model_type,
            loss_type=self._loss_type,
        )

        self._model = self._trainer._model
        self._result = self._trainer.train()

        return self._result

    def _train_modalities(self) -> None:
        """Trains a Autoencoder for each modality in the dataset.
        This method orchestrates the training of individual modality models

        This method orchestrates the complete training process:
        1. Train individual modality models
        2. Extract and concatenate latent spaces
        3. Populates the self._modality_models and self._modality_results attributes

        """
        # Step 1: Train individual modality models
        self._modality_trainers, self._modality_results = (
            self._orchestrator.train_modalities()
        )

        self._result.sub_results = self._modality_results
        # Step 2: Prepare concatenated latent space datasets
        self._train_latent_ds = self._orchestrator.prepare_latent_datasets(
            split="train"
        )
        self._valid_latent_ds = self._orchestrator.prepare_latent_datasets(
            split="valid"
        )
        self.concat_idx = self._orchestrator.concat_idx

    def _reconstruct(self, split: str) -> None:
        """Orchestrates the reconstruction by delegating the task to the StackixOrchestrator.

        Args:
            split: The data split to reconstruct ('train', 'valid', 'test').
        """
        stacked_recon = self._result.reconstructions.get(epoch=-1, split=split)
        if stacked_recon is None:
            print(f"Warning: No reconstruction found for split '{split}'. Skipping.")
            return

        # The orchestrator handles the de-concatenation, re-assembly, and decoding.
        modality_reconstructions = self._orchestrator.reconstruct_from_stack(
            reconstructed_stack=torch.from_numpy(stacked_recon).to(self._fabric.device)
        )

        # The result is the final, full-sized data reconstructions.
        self._result.sub_reconstructions = modality_reconstructions

    def predict(
        self, data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs
    ) -> Result:
        """Make predictions on the given dataset.

        Args:
            data: The dataset to make predictions on.
            model: The model to use for predictions. If None, uses the trained model.
            **kwargs: Additional keyword arguments.
        Returns:
            Result: The prediction results including reconstructions and latent spaces.
        """
        self.n_test = len(data) if data is not None else 0
        self._orchestrator.set_testset(testset=data)
        test_ds = self._orchestrator.prepare_latent_datasets(split="test")
        self.testset = test_ds
        print(test_ds)
        pred_result = super().predict(data=test_ds, model=model)
        self._result.update(other=pred_result)
        self._reconstruct(split="test")
        return self._result
