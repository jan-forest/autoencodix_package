import abc
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
from autoencodix.utils.default_config import DefaultConfig


class BaseTrainer(abc.ABC):
    """General training logic for all autoencoder models.

    This class sets up the model, optimizer, and data loaders. It also handles
    reproducibility and model-specific configurations. Subclasses must implement
    model training and prediction logic.

    Attributes:
        trainset: The dataset used for training.
        validset: The dataset used for validation, if provided.
        result: An object to store and manage training results.
        config: Configuration object containing training hyperparameters and settings.
        model_type: The autoencoder model class to be trained.
        loss_fn: Instantiated loss function specific to the model.
        trainloader: DataLoader for the training dataset.
        validloader: DataLoader for the validation dataset, if provided.
        model: The instantiated model architecture.
        optimizer: The optimizer used for training.
        fabric: Lightning Fabric wrapper for device and precision management.
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
    ):
        self._trainset = trainset
        self._model_type = model_type
        self._validset = validset
        self._result = result
        self._config = config
        self.ontologies = ontologies

        # Call this first for tests to work
        self._input_validation()
        self._handle_reproducibility()

        self._loss_fn = loss_type(config=self._config)

        # Internal data handling
        self._model: BaseAutoencoder
        self._fabric = Fabric(
            accelerator=self._config.device,
            devices=self._config.n_gpus,
            precision=self._config.float_precision,
            strategy=self._config.gpu_strategy,
        )

        self._init_loaders()
        self._setup_fabric()
        self._n_cpus = os.cpu_count()
        if self._n_cpus is None:
            self._n_cpus = 0

    def _setup_fabric(self):
        self._input_dim = cast(BaseDataset, self._trainset).get_input_dim()
        self._init_model_architecture(ontologies=self.ontologies)  # Ontix

        self._optimizer = torch.optim.AdamW(
            params=self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        self._model, self._optimizer = self._fabric.setup(self._model, self._optimizer)
        self._trainloader = self._fabric.setup_dataloaders(self._trainloader)  # type: ignore
        if self._validloader is not None:
            self._validloader = self._fabric.setup_dataloaders(self._validloader)  # type: ignore
        self._fabric.launch()

    def _init_loaders(self):
        self._trainloader = DataLoader(
            cast(BaseDataset, self._trainset),
            batch_size=self._config.batch_size,
            shuffle=True,  # best practice to shuffle in training
        )
        if self._validset:
            self._validloader = DataLoader(
                dataset=self._validset,
                batch_size=self._config.batch_size,
                shuffle=False,

            )
        else:
            self._validloader = None  # type: ignore

    def _input_validation(self) -> None:
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
            if self._config.device == "cuda":
                torch.cuda.manual_seed(seed=self._config.global_seed)
                torch.cuda.manual_seed_all(seed=self._config.global_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            elif self._config.device == "mps":
                raise NotImplementedError(
                    "MPS backend does not support reproducibility settings."
                )
            else:
                print("cpu not relevant here")

    def _init_model_architecture(self, ontologies: tuple) -> None:
        if ontologies is None:
            self._model = self._model_type(
                config=self._config, input_dim=self._input_dim
            )
        else:
            ## Ontix specific
            self._model = self._model_type(
                config=self._config,
                input_dim=self._input_dim,
                ontologies=ontologies,
                feature_order=self._trainset.feature_ids,  # type: ignore
            )

    def _should_checkpoint(self, epoch: int):
        return (
            epoch + 1
        ) % self._config.checkpoint_interval == 0 or epoch == self._config.epochs - 1

    @abc.abstractmethod
    def train(self, epochs_overwrite: Optional[int]) -> Result:
        pass

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def predict(self, data: BaseDataset, model: torch.nn.Module) -> Result:
        pass
