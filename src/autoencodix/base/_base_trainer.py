import abc
from typing import Optional, Union, Type

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


# internal check done
# write tests: done
class BaseTrainer(abc.ABC):
    """
    Abstract base class for all trainer implementations. It sets up the general components
    needed for training, such as the model, optimizer, and data loaders. Additionally,
    it provides the structure for handling reproducibility and model-specific configurations.

    Attributes
    ----------
    _trainset : BaseDataset
        The dataset used for training.
    _validset : Optional[BaseDataset]
        The dataset used for validation, if provided.
    _result : Result
        An object to store and manage training results.
    _config : DefaultConfig
        Configuration object containing training hyperparameters and settings.
    _model_type : Type[BaseAutoencoder]
        The autoencoder model class to be trained.
    _trainloader : DataLoader
        DataLoader for the training dataset.
    _validloader : Optional[DataLoader]
        DataLoader for the validation dataset, if validation data is provided.
    _model : BaseAutoencoder
        The instantiated model architecture.
    _optimizer : torch.optim.Optimizer
        The optimizer used for training.
    _fabric : Fabric
        A wrapper for distributed training and precision management.

    Methods
    -------
    train() -> Result
        Abstract method that must be implemented to define the training loop.
    _capture_dynamics(epoch: int, model_output: torch.Tensor) -> None
        Abstract method for capturing model dynamics during training.
    _loss_fn(model_output: ModelOutput, targets: torch.Tensor) -> torch.Tensor
        Abstract method for defining the loss computation logic.
    _input_validation() -> None
        Validates input arguments and ensures the proper configuration.
    _handle_reproducibility() -> None
        Sets seeds and configurations for reproducibility.
    _init_model_architecture() -> None
        Initializes the model architecture based on the configuration.

    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[Union[BaseDataset, None]],
        result: Optional[Result],
        config: Optional[Union[None, DefaultConfig]],
        model_type: Type[BaseAutoencoder],
    ):
        # passed attributes --------------------------
        self._trainset = trainset
        self._model_type = model_type
        self._validset = validset
        self._result = result
        self._config = config
        # call this first for tests to work
        self._input_validation()
        self._handle_reproducibility()

        # internal data handling ----------------------
        self._trainloader = DataLoader(
            self._trainset,
            batch_size=self._config.batch_size,
            shuffle=True,  # best practice to shuffle in training
            num_workers=self._config.n_workers,
        )
        if self._validset:
            self._validloader = DataLoader(
                self._validset,
                batch_size=self._config.batch_size,
                shuffle=False,
                num_workers=self._config.n_workers,
            )

        # model and optimizer setup --------------------------------
        self._input_dim = self._trainset.get_input_dim()
        self._init_model_architecture()

        self._fabric = Fabric(
            accelerator=self._config.device,
            devices=self._config.n_gpus,
            precision=self._config.float_precision,  # TODO see issue github
            strategy=self._config.gpu_strategy,  # TODO allow non-auto and handle based on available devices
        )

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        self._model, self._optimizer = self._fabric.setup(self._model, self._optimizer)
        self._trainloader = self._fabric.setup_dataloaders(self._trainloader)
        if self._validset:
            self._validloader = self._fabric.setup_dataloaders(self._validloader)
        self._fabric.launch()

    def _input_validation(self):
        if self._trainset is None:
            raise ValueError(
                "Trainset cannot be None. Check the indices you provided with a custom split or be sure that the train_ratio attribute of the config is >0."
            )
        if not isinstance(self._trainset, (BaseDataset)):
            raise TypeError(
                f"Expected train type to be an instance of BaseDataset, got {type(self._trainset)}."
            )
        if self._validset is None:
            print("training without validation")
        if self._validset:
            if not isinstance(self._validset, (BaseDataset)):
                raise TypeError(
                    f"Expected valid type to be an instance of BaseDataset, got {type(self._validset)}."
                )
        if self._config is None:
            raise ValueError("Config cannot be None.")

    def _handle_reproducibility(self):
        """Sets all relevant seeds for reproducibility"""
        if self._config.reproducible:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed=self._config.global_seed)
            if self._config.device == "cuda":
                torch.cuda.manual_seed(seed=self._config.global_seed)
                torch.cuda.manual_seed_all(seed=self._config.global_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            elif self._config.device == "mps":
                print("warning: MPS backend does not support reproducibility")
            else:
                print("cpu not relevant here")

    def _init_model_architecture(self):

        self._model = self._model_type(config=self._config, input_dim=self._input_dim)

    @abc.abstractmethod
    def train(self) -> Result:
        pass

    @abc.abstractmethod
    def _capture_dynamics(self, epoch: int, model_output: torch.Tensor) -> None:
        pass

    @abc.abstractmethod
    def _loss_fn(
        sefl, model_output: ModelOutput, targets: torch.Tensor
    ) -> torch.Tensor:
        pass
