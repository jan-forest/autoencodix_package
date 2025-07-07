from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.trainers._stackix_orchestrator import StackixOrchestrator
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class StackixTrainer(GeneralTrainer):
    """
    StackixTrainer is a wrapper class that conforms to the BaseTrainer interface
    while internally using the StackixOrchestrator to perform the
    actual training process.

    This trainer maintains compatibility with the BasePipeline interface while
    leveraging the more modular and well-designed StackixOrchestrator
    classes for the actual implementation.

    Attributes
    ----------
    _orchestrator : StackixOrchestrator
        The orchestrator that manages modality model training and latent space preparation
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
        orchestrator_type: Type[StackixOrchestrator] = StackixOrchestrator,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        workdir: str = "./stackix_work",
        ontologies: Optional[Tuple] = None,
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
        """
        Get the trained model.

        Returns
        -------
        torch.nn.Module
            The trained model
        """
        return self._model

    def train(self) -> Result:
        """
        Train the stacked model on the concatenated latent space.

        Uses the standard BaseTrainer training process but with the stacked model.

        Returns
        -------
        Result
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
        """
        Trains a Autoencoder for each modality in the dataset.
        This method orchestrates the training of individual modality models

        This method orchestrates the complete training process:
        1. Train individual modality models
        2. Extract and concatenate latent spaces
        3. Populates the self._modality_models and self._modality_results attributes

        Returns
        -------
            None
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
        stacked_recon = self._result.reconstructions.get(epoch=-1, split=split)

        modality_reconstructions = {}
        for name, (start_idx, end_idx) in self.concat_idx.items():
            stacked_input = stacked_recon[:, start_idx:end_idx]
            stacked_tensor = torch.tensor(stacked_input, dtype=torch.float32)
            with self._fabric.autocast() and torch.no_grad():
                model = self._modality_results[name].model
                stacked_tensor = self._fabric.to_device(stacked_tensor)
                model = self._fabric.to_device(model)
                model.eval()
                modality_reconstructions[name] = model.decode(stacked_tensor).cpu()

        self._result.sub_reconstructions = modality_reconstructions

    def predict(self, data: BaseDataset, model: torch.nn.Module) -> Result:
        """ """
        self.n_test = len(data) if data is not None else 0
        self._orchestrator.set_testset(testset=data)
        test_ds = self._orchestrator.prepare_latent_datasets(split="test")
        pred_result = super().predict(data=test_ds, model=model)
        self._result.update(other=pred_result)
        self._reconstruct(split="test")
        return self._result

    def _capture_dynamics(
        self,
        epoch: int,
        model_output: List[ModelOutput],
        split: str,
        sample_ids: Optional[List[int]] = None,
    ) -> None:
        return super()._capture_dynamics(epoch, model_output, split, sample_ids)
