import torch
import os
import numpy as np
from typing import Optional, Type, Union, Tuple, Any, Dict, List
from collections import defaultdict
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput


class GeneralTrainer(BaseTrainer):
    """Handles general training logic for autoencoder models.

    Attributes:
         _trainset: The dataset used for training.
         _validset: The dataset used for validation, if provided.
         _result: An object to store and manage training results.
         _config: Configuration object containing training hyperparameters and settings.
         _model_type: The autoencoder model class to be trained.
         _loss_fn: Instantiated loss function specific to the model.
         _trainloader: DataLoader for the training dataset.
         _validloader: DataLoader for the validation dataset, if provided.
         _model: The instantiated model architecture.
         _optimizer: The optimizer used for training.
         _fabric: Lightning Fabric wrapper for device and precision management.
         n_train: Number of training samples.
         n_valid: Number of validation samples.
         n_test: Number of test samples (set during prediction).
         n_features: Number of input features.
         latent_dim: Dimensionality of the latent space.
         device: Device on which the model is located.
         _n_cpus: Number of CPU cores available.
         _reconstruction_buffer: Buffer to store reconstructions during training/validation/testing.
         _latentspace_buffer: Buffer to store latent representations during training/validation/testing.
         _mu_buffer: Buffer to store latent means (for VAE) during training/validation
         _sigma_buffer: Buffer to store latent log-variances (for VAE) during training/validation/testing.
         _sample_ids_buffer: Buffer to store sample IDs during training/validation/testing.

    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        ontologies: Optional[Union[Tuple, List]] = None,
    ):
        """Initializes the GeneralTrainer.

        Args:
                trainset: The dataset used for training.
                validset: The dataset used for validation, if provided.
                result: An object to store and manage training results.
                config: Configuration object containing training hyperparameters and settings.
                model_type: The autoencoder model class to be trained.
                loss_type: The loss function class to be used for training.
                ontologies: Ontology information, if provided for Ontix
        """
        self._n_cpus = os.cpu_count()
        if self._n_cpus is None:
            self._n_cpus = 0

        super().__init__(
            trainset, validset, result, config, model_type, loss_type, ontologies
        )

        self.latent_dim = config.latent_dim
        if ontologies is not None:
            if not hasattr(self._model, "latent_dim"):
                raise ValueError(
                    "Model must have a 'latent_dim' attribute when ontologies are provided."
                )
            self.latent_dim = self._model.latent_dim

        # we will set this later, in the predict method
        self.n_test: Optional[int] = None
        self.n_train = len(trainset.data) if trainset else 0
        self.n_valid = len(validset.data) if validset else 0
        self.n_features = trainset.get_input_dim() if trainset else 0
        self.device = next(self._model.parameters()).device

        self._init_buffers()

    def _init_buffers(self, input_data: Optional[BaseDataset] = None):
        if input_data:
            self.n_features = input_data.get_input_dim()

        def make_tensor_buffer(size: int, dim: Union[int, Tuple[int, ...]]):
            if isinstance(dim, int):
                return torch.zeros((size, dim), device=self.device)
            else:
                return torch.zeros((size, *dim), device=self.device)

        def make_numpy_buffer(size: int):
            return np.empty((size,), dtype=object)

        self._latentspace_buffer = {
            "train": make_tensor_buffer(self.n_train, self.latent_dim),
            "valid": (
                make_tensor_buffer(self.n_valid, self.latent_dim)
                if self.n_valid
                else None
            ),
            "test": (
                make_tensor_buffer(self.n_test, self.latent_dim)
                if self.n_test
                else None
            ),
        }
        self._reconstruction_buffer = {
            "train": make_tensor_buffer(self.n_train, self.n_features),
            "valid": (
                make_tensor_buffer(self.n_valid, self.n_features)
                if self.n_valid
                else None
            ),
            "test": (
                make_tensor_buffer(self.n_test, self.n_features)
                if self.n_test
                else None
            ),
        }
        self._mu_buffer = {
            "train": make_tensor_buffer(self.n_train, self.latent_dim),
            "valid": (
                make_tensor_buffer(self.n_valid, self.latent_dim)
                if self.n_valid
                else None
            ),
            "test": (
                make_tensor_buffer(self.n_test, self.latent_dim)
                if self.n_test
                else None
            ),
        }
        self._sigma_buffer = {
            "train": make_tensor_buffer(self.n_train, self.latent_dim),
            "valid": (
                make_tensor_buffer(self.n_valid, self.latent_dim)
                if self.n_valid
                else None
            ),
            "test": (
                make_tensor_buffer(self.n_test, self.latent_dim)
                if self.n_test
                else None
            ),
        }
        self._sample_ids_buffer = {
            "train": make_numpy_buffer(self.n_train),
            "valid": make_numpy_buffer(self.n_valid) if self.n_valid else None,
            "test": make_numpy_buffer(self.n_test) if self.n_test else None,
        }

    def _apply_post_backward_processing(self):
        pass

    def train(self, epochs_overwrite=None) -> Result:
        """Orchestrates training over multiple epochs, including training and validation phases.

        Args:
            epochs_overwrite: If provided, overrides the number of epochs specified in the config.
                This is only there so we can use the train method for pretraining.Any
        Returns:
            Result object containing training results and dynamics like latent spaces and reconstructions.
        """
        epochs = self._config.epochs
        if epochs_overwrite:
            epochs = epochs_overwrite
        with self._fabric.autocast():
            for epoch in range(epochs):
                self._init_buffers()
                should_checkpoint: bool = self._should_checkpoint(epoch)
                self._model.train()

                epoch_loss, epoch_sub_losses = self._train_epoch(
                    should_checkpoint=should_checkpoint, epoch=epoch
                )
                self._log_losses(epoch, "train", epoch_loss, epoch_sub_losses)

                if self._validset:
                    valid_loss, valid_sub_losses = self._validate_epoch(
                        should_checkpoint=should_checkpoint, epoch=epoch
                    )
                    self._log_losses(epoch, "valid", valid_loss, valid_sub_losses)

                if should_checkpoint:
                    self._store_checkpoint(epoch)

        self._result.model = next(self._model.children())
        return self._result

    def _train_epoch(
        self, should_checkpoint: bool, epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Handles loss computation, backwards pass and checkpointing for one epoch.

        Args:
            should_checkpoint: Whether to checkpoint this epoch.
            epoch: The current epoch number.
        Returns:
            total_loss: The total loss for the epoch.
            sub_losses: A dictionary of sub-losses accumulated over the epoch.
        """
        total_loss = 0.0
        sub_losses: Dict[str, float] = defaultdict(float)

        for indices, features, sample_ids in self._trainloader:
            self._optimizer.zero_grad()
            model_outputs = self._model(features)
            loss, batch_sub_losses = self._loss_fn(
                # model_output=model_outputs, targets=features, epoch=epoch
                model_output=model_outputs,
                targets=features,
                epoch=epoch,
                n_samples=len(
                    self._trainloader.dataset
                ),  # Pass n_samples for disentangled loss calculations
            )

            self._fabric.backward(loss)
            self._apply_post_backward_processing()
            self._optimizer.step()

            total_loss += loss.item()
            for k, v in batch_sub_losses.items():
                if "_factor" not in k:  # Skip factor losses
                    sub_losses[k] += v.item()
                else:
                    sub_losses[k] = v.item()

            if should_checkpoint:
                self._capture_dynamics(model_outputs, "train", indices, sample_ids)

        return total_loss, sub_losses

    def _validate_epoch(
        self, should_checkpoint: bool, epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Handles validation for one epoch.

        Args:
            should_checkpoint: Whether to checkpoint this epoch.
            epoch: The current epoch number.
        Returns:
            total_loss: The total loss for the epoch.
            sub_losses: A dictionary of sub-losses accumulated over the epoch.
        """
        total_loss = 0.0
        sub_losses: Dict[str, float] = defaultdict(float)
        self._model.eval()

        with torch.no_grad():
            for indices, features, sample_ids in self._validloader:
                model_outputs = self._model(features)
                loss, batch_sub_losses = self._loss_fn(
                    # model_output=model_outputs, targets=features, epoch=epoch
                    model_output=model_outputs,
                    targets=features,
                    epoch=epoch,
                    n_samples=len(
                        self._validloader.dataset
                    ),  # Pass n_samples for disentangled loss calculations
                )
                total_loss += loss.item()
                for k, v in batch_sub_losses.items():
                    if "_factor" not in k:  # Skip factor losses
                        sub_losses[k] += v.item()
                    else:
                        sub_losses[k] = v.item()
                if should_checkpoint:
                    self._capture_dynamics(model_outputs, "valid", indices, sample_ids)

        return total_loss, sub_losses

    def _log_losses(
        self, epoch: int, split: str, total_loss: float, sub_losses: Dict[str, float]
    ) -> None:
        """Logs the total and sub-losses for an epoch and stores them in the Result object.

        Args:
            epoch: The current epoch number.
            split: The data split ("train" or "valid").
            total_loss: The total loss for the epoch.
            sub_losses: A dictionary of sub-losses for the epoch.
        """
        dataset_len = len(
            self._trainloader.dataset if split == "train" else self._validloader.dataset
        )
        self._result.losses.add(epoch=epoch, split=split, data=total_loss / dataset_len)
        self._result.sub_losses.add(
            epoch=epoch,
            split=split,
            data={
                k: v / dataset_len if "_factor" not in k else v
                for k, v in sub_losses.items()
            },
        )

        self._fabric.print(
            f"Epoch {epoch + 1} - {split.capitalize()} Loss: {total_loss:.4f}"
        )
        self._fabric.print(
            f"Sub-losses: {', '.join([f'{k}: {v:.4f}' for k, v in sub_losses.items()])}"
        )

    def _store_checkpoint(self, epoch: int) -> None:
        """Stores model checkpoints and training dynamics to result object.

        Args:
            epoch: The current epoch number.
        """
        self._result.model_checkpoints.add(epoch=epoch, data=self._model.state_dict())
        self._dynamics_to_result(epoch, "train")
        if self._validset:
            self._dynamics_to_result(epoch, "valid")

    def _capture_dynamics(
        self,
        model_output: Union[ModelOutput, Any],
        split: str,
        indices: torch.Tensor,
        sample_ids: Any,
        **kwargs,
    ) -> None:
        """Writes model dynamics (latent space, reconstructions, etc.) to buffers.

        Args:
            model_output: The output from the model containing dynamics information.
            split: The data split ("train", "valid", or "test").
            indices: The indices of the samples in the current batch.
            sample_ids: The sample IDs corresponding to the current batch.
            **kwargs: Additional arguments (not used here).

        """
        indices_np = (
            indices.cpu().numpy()
            if isinstance(indices, torch.Tensor)
            else np.array(indices)
        )

        self._sample_ids_buffer[split][indices_np] = np.array(sample_ids)

        self._latentspace_buffer[split][indices_np] = model_output.latentspace.detach()
        self._reconstruction_buffer[split][indices_np] = (
            model_output.reconstruction.detach()
        )

        if model_output.latent_logvar is not None:
            self._sigma_buffer[split][indices_np] = model_output.latent_logvar.detach()

        if model_output.latent_mean is not None:
            self._mu_buffer[split][indices_np] = model_output.latent_mean.detach()

    def _dynamics_to_result(self, epoch: int, split: str) -> None:
        """Transfers buffered dynamics to the Result object.

        Args:
            epoch: The current epoch number.
            split: The data split ("train", "valid", or "test").
        """

        def maybe_add(buffer, target):
            if buffer[split] is not None and buffer[split].sum() != 0:
                target.add(
                    epoch=epoch, split=split, data=buffer[split].cpu().detach().numpy()
                )

        self._result.latentspaces.add(
            epoch=epoch,
            split=split,
            data=self._latentspace_buffer[split].cpu().detach().numpy(),
        )
        self._result.reconstructions.add(
            epoch=epoch,
            split=split,
            data=self._reconstruction_buffer[split].cpu().detach().numpy(),
        )
        self._result.sample_ids.add(
            epoch=epoch, split=split, data=self._sample_ids_buffer[split]
        )
        maybe_add(self._mu_buffer, self._result.mus)
        maybe_add(self._sigma_buffer, self._result.sigmas)

    def predict(
        self, data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs
    ) -> Result:
        """Decided to add predict method to the trainer class.

        This violates SRP, but the trainer class has a lot of attributes and methods
        that are needed for prediction. So this way we don't need to write so much duplicate code

        Args:
            data: BaseDataset unseen data to run inference on
            model: torch.nn.Module model to run inference with
            **kwargs: Additional arguments (not used here).

        Returns:
            self._result: Result object containing the inference results

        """
        if model is None:
            model: torch.nn.Module = self._model
            import warnings

            warnings.warn(
                "No model provided for prediction, using the trained model from the trainer."
            )
        model.eval()
        inference_loader = DataLoader(
            data,
            batch_size=self._config.batch_size,
            shuffle=False,
        )
        self.n_test = len(data)
        self._init_buffers(input_data=data)
        inference_loader = self._fabric.setup_dataloaders(inference_loader)  # type: ignore
        with self._fabric.autocast(), torch.inference_mode():
            for idx, data, sample_ids in inference_loader:
                model_output = model(data)
                self._capture_dynamics(
                    model_output=model_output,
                    split="test",
                    sample_ids=sample_ids,
                    indices=idx,
                )
        self._dynamics_to_result(epoch=-1, split="test")

        return self._result

    def decode(self, x: torch.Tensor):
        """Decodes the input tensor x using the trained model.

        Args:
            x: Input tensor to be decoded.
        Returns:
            Decoded tensor.
        """
        with self._fabric.autocast(), torch.no_grad():
            x = self._fabric.to_device(obj=x)
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Expected input to be a torch.Tensor, got {type(x)} instead."
                )
            return self._model.decode(x=x)

    def get_model(self) -> torch.nn.Module:
        """Getter method for the trained model.

        Returns:

            The trained model as a torch.nn.Module.
        """
        return self._model
