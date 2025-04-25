from typing import Dict, Optional, Type, Union, List
import torch
from torch.utils.data import DataLoader


from autoencodix.trainers._stackix_orchestrator import StackixOrchestrator
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_loss import BaseLoss
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._model_output import ModelOutput


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
        self._modality_models: Optional[Dict[str, BaseAutoencoder]] = None
        self._modality_results: Optional[Dict[str, Result]] = None

    def train(self) -> Result:
        """
        Train the stacked model on the concatenated latent space.

        Uses the standard BaseTrainer training process but with the stacked model.

        Returns
        -------
        Result
            Training results including losses, latent spaces, and other metrics
        """
        self._train_modalities()
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
        self._modality_models, self._modality_results = (
            self._orchestrator.train_modalities()
        )
        self._result.sub_results = self._modality_results

        # Step 2: Prepare concatenated latent space datasets
        train_latent_ds, valid_latent_ds = self._orchestrator.prepare_latent_datasets()

        # Step 3: Create and train the stacked model
        self._trainer = self._trainer_type(
            trainset=train_latent_ds,
            validset=valid_latent_ds,
            result=self._result,
            config=self._config,
            model_type=self._model_type,
            loss_type=self._loss_type,
        )

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

        return self.predict_with_reconstruction(data)

    def _capture_dynamics(
        self, epoch: int, model_output: List[ModelOutput], split: str
    ) -> None:
        return super()._capture_dynamics(epoch, model_output, split)

    def predict_with_reconstruction(
        self,
        data: Union[StackixDataset, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions and reconstruct original modality data.

        This method takes either a StackixDataset or a pre-encoded latent tensor,
        passes it through the stacked model, and reconstructs each original modality.

        Parameters
        ----------
        data : Union[StackixDataset, torch.Tensor]
            Input data for prediction, either a multi-modal dataset or concatenated latent tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of reconstructed data for each modality

        Raises
        ------
        ValueError
            If no model is available or an unsupported data type is provided
        """
        if self._model is None:
            raise ValueError("No model available for prediction")

        # Build concatenated latent representation if needed
        if isinstance(data, StackixDataset):
            # Handle multi-modal dataset
            modality_latents = []

            for modality in data.modality_keys:
                model = self._modality_models.get(modality)
                if model is None:
                    raise ValueError(f"Model for modality {modality} not trained")

                modality_dataset = data.datasets_dict[modality]

                # Create dataloader
                dataloader = DataLoader(
                    dataset=modality_dataset,
                    batch_size=self._config.batch_size,
                    shuffle=False,
                    num_workers=self._config.n_workers,
                    pin_memory=self._config.device in ["cuda", "gpu"],
                )

                # Set up dataloader with fabric
                dataloader = self._fabric.setup_dataloaders(dataloader)

                # Extract latent representations
                latent_parts = []
                with self._fabric.autocast(), torch.no_grad():
                    model.eval()
                    for batch, _ in dataloader:
                        mu, _ = model.encode(x=batch)
                        latent_parts.append(mu)

                modality_latents.append(torch.cat(tensors=latent_parts, dim=0))

            # Concatenate all modality latents
            concatenated = torch.cat(tensors=modality_latents, dim=1)

        elif torch.is_tensor(data):
            # Handle pre-concatenated latent tensor
            concatenated = data
        else:
            raise ValueError("Unsupported data type for predict_with_reconstruction")

        # Process through stacked model
        self._model.eval()
        with self._fabric.autocast(), torch.no_grad():
            stacked_out = self._model(x=concatenated)
            recon_latent = stacked_out.latentspace

        # Reconstruct each modality
        recon: Dict[str, torch.Tensor] = {}
        idx = 0

        for modality, dim in self._modality_latent_dims.items():
            # Extract the portion of latent space for this modality
            slice_z = recon_latent[:, idx : idx + dim]
            idx += dim

            # Decode using the modality-specific decoder
            recon[modality] = self._modality_models[modality].decode(x=slice_z)

        return recon
