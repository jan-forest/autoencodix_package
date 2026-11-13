import abc
from packaging.version import Version
import random
import numpy as np
import os
import warnings
from typing import Optional, Type, List, cast, Tuple, Union

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class BaseTrainer(abc.ABC):
    """General training logic for all autoencoder models.

    This class sets up the model, optimizer, and data loaders. It also handles
    reproducibility and model-specific configurations. Subclasses must implement
    model training and prediction logic.

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
        ontologies: Ontology information, if provided for Ontix
    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        ontologies: Optional[
            Union[Tuple, List]
        ] = None,  # Addition to Varix, mandotory for Ontix
        **kwargs,
    ):
        """Store the training components and set up model, optimizer and loaders.

        Args:
            trainset: Dataset used for training; must not be None.
            validset: Dataset used for validation, or None to train without
                validation.
            result: Result container that collects losses, latent spaces and the
                trained model.
            config: Configuration with training hyperparameters and device settings.
            model_type: Autoencoder architecture class to instantiate.
            loss_type: Loss class to instantiate with ``config``.
            ontologies: Ontology definitions; required for ``Ontix``, ignored by
                other architectures.
            **kwargs: Additional keyword arguments for subclasses.

        (LLM-generated docstring, pending human review.)
        """
        self._trainset = trainset
        self._model_type = model_type
        self._validset = validset
        self._result = result
        self._config = config
        self._loss_type = loss_type
        self.ontologies = ontologies
        self.setup_trainer()

    def setup_trainer(self, old_model=None):
        """Create loss, Lightning Fabric, model, optimizer and data loaders.

        Called from ``__init__``. Also called before prediction with an already
        trained model, in which case input validation and data loader creation are
        skipped and the given model is reused.

        Args:
            old_model: Previously trained model to reuse (e.g. after ``load``).
                If None, a new model is built from ``model_type``.

        (LLM-generated docstring, pending human review.)
        """
        if old_model is None:
            self._input_validation()
            self._init_loaders()

        self._loss_fn = self._loss_type(config=self._config)

        self._handle_reproducibility()
        # Internal data handling
        self._model: BaseAutoencoder
        self._fabric = Fabric(
            accelerator=self._config.device,
            devices=self._config.n_gpus,
            precision=self._config.float_precision,
            strategy=self._config.gpu_strategy,
        )

        self._fabric.launch()
        self._setup_fabric(old_model=old_model)

        self._n_cpus = os.cpu_count()
        if self._n_cpus is None:
            self._n_cpus = 0

    def _setup_fabric(self, old_model=None):
        """Sets up the model, optimizer, and data loaders with Lightning Fabric.

        Args:
            old_model: Previously trained model to reuse instead of building a new
                one. Data loaders are only wrapped and the model is only compiled
                when this is None.
        """
        self._init_model_architecture(
            ontologies=self.ontologies, old_model=old_model
        )  # Ontix

        self._optimizer = torch.optim.AdamW(
            params=self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        if old_model is None:
            self._trainloader = self._fabric.setup_dataloaders(self._trainloader)  # type: ignore
            if self._validloader is not None:
                self._validloader = self._fabric.setup_dataloaders(self._validloader)  # type: ignore

        if (
            Version(torch.__version__) >= Version("2.0")
            and torch.cuda.is_available()
            and old_model is None
            and self._config.compile_model
        ):  # dont compile twice
            self._model = torch.compile(self._model)

        self._model, self._optimizer = self._fabric.setup(self._model, self._optimizer)

    def _init_loaders(self):
        """Initializes the DataLoaders for training and validation datasets."""
        # g = torch.Generator()
        # g.manual_seed(self._config.global_seed)
        last_batch_is_one_sample = len(self._trainset) % self._config.batch_size == 1
        corrected_bs = (
            self._config.batch_size + 1
            if last_batch_is_one_sample
            else self._config.batch_size
        )
        if last_batch_is_one_sample:
            warnings.warn(
                f"increased batch_size to {corrected_bs} for trainset, to avoid dropping samples and having batches (makes trainingdynamics messy with missing samples per epoch) of size one (fails for Models with BachNorm)"
            )

        self._trainloader = DataLoader(
            cast(BaseDataset, self._trainset),
            shuffle=True,
            batch_size=corrected_bs,
            worker_init_fn=self._seed_worker,
            pin_memory=self._config.pin_memory,
            # generator=g,
        )
        if self._validset:
            last_batch_is_one_sample = (
                len(self._validset) % self._config.batch_size == 1
            )
            corrected_bs = (
                self._config.batch_size + 1
                if last_batch_is_one_sample
                else self._config.batch_size
            )
            if last_batch_is_one_sample:
                warnings.warn(
                    f"increased batch_size to {corrected_bs} for validset, to avoid dropping samples and having batches (makes trainingdynamics messy with missing samples per epoch) of size one (fails for Models with BachNorm)"
                )

            self._validloader = DataLoader(
                dataset=self._validset,
                batch_size=self._config.batch_size,
                shuffle=False,
                pin_memory=self._config.pin_memory,
            )
        else:
            self._validloader = None  # type: ignore

    def _input_validation(self) -> None:
        """Check that train/validation datasets and config are usable for training.

        Raises:
            ValueError: If the train set or config is None.
            TypeError: If a dataset is not a ``BaseDataset``.

        (LLM-generated docstring, pending human review.)
        """
        if self._trainset is None:
            raise ValueError(
                "Trainset cannot be None. Check the indices you provided with a custom split or be sure that the train_ratio attribute of the config is >0."
            )
        if not isinstance(self._trainset, BaseDataset):
            raise TypeError(
                f"Expected train type to be an instance of BaseDataset, got {type(self._trainset)}."
            )
        if self._validset is None:
            print("training without validation")
        elif not isinstance(self._validset, BaseDataset):
            raise TypeError(
                f"Expected valid type to be an instance of BaseDataset, got {type(self._validset)}."
            )
        if self._config is None:
            raise ValueError("Config cannot be None.")

    def _handle_reproducibility(self) -> None:
        """Sets all relevant seeds for reproducibility

        Raises:
            NotImplementedError: If the device is set to "mps" (Apple Silicon).
        """
        if self._config.reproducible:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed=self._config.global_seed)
            random.seed(self._config.global_seed)
            np.random.seed(self._config.global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed=self._config.global_seed)
                torch.cuda.manual_seed_all(seed=self._config.global_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed=self._config.global_seed)

                # torch.use_deterministic_algorithms(True, warn_only=True)

                print(
                    "Warning: MPS backend has limited support for deterministic algorithms. "
                    "Seeding is active, but full reproducibility is not guaranteed."
                )
            elif not torch.cuda.is_available():
                print(
                    f"Reproducibility settings for device {self._config.device} are not implemented or necessary i.e. for cpu."
                )

    def _seed_worker(self, worker_id):
        worker_seed = self._config.global_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _init_model_architecture(self, ontologies: tuple, old_model=None) -> None:
        """Initializes the model architecture, based on the model type and input dimension.

        Args:
            ontologies: Ontology definitions passed to the model (``Ontix``); if
                None, the model is built from config and input dimension only.
            old_model: Previously trained model; if given, it is used as is.
        """
        if old_model:
            self._model = old_model
            return

        self._input_dim = cast(BaseDataset, self._trainset).get_input_dim()
        self.feature_order = self._trainset.feature_ids
        if ontologies is None:
            self._model = self._model_type(
                config=self._config, input_dim=self._input_dim
            )

        else:
            self._model = self._model_type(
                config=self._config,
                input_dim=self._input_dim,
                ontologies=ontologies,
                feature_order=self._trainset.feature_ids,  # type: ignore
            )

    def _should_checkpoint(self, epoch: int) -> bool:
        """Return whether results should be stored for this epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            True every ``checkpoint_interval`` epochs and at the last epoch.

        (LLM-generated docstring, pending human review.)
        """
        return (
            (epoch + 1) % self._config.checkpoint_interval == 0
            or epoch == self._config.epochs - 1
        )

    @abc.abstractmethod
    def train(self, epochs_overwrite: Optional[int] = None) -> Result:
        """Train the model and return the populated result container.

        Args:
            epochs_overwrite: Number of epochs to train instead of
                ``config.epochs``.

        Returns:
            The ``Result`` object with losses, latent spaces and the trained model.

        (LLM-generated docstring, pending human review.)
        """
        pass

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into the input space with the trained model.

        Args:
            x: Latent vectors of shape (n_samples, latent_dim).

        Returns:
            Reconstructed data in the input space.

        (LLM-generated docstring, pending human review.)
        """
        pass

    @abc.abstractmethod
    def predict(
        self, data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs
    ) -> Result:
        """Encode and reconstruct a dataset with a trained model.

        Args:
            data: Dataset to predict on.
            model: Trained model to use; defaults to the trainer's own model.
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            The ``Result`` object updated with latent spaces and reconstructions.

        (LLM-generated docstring, pending human review.)
        """
        pass

    @abc.abstractmethod
    def purge(self) -> None:
        """Cleans up any resources used during training, such as cached data or large attributes."""
        pass
