import abc
from typing import Optional, Union

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


# internal check done
# write tests: TODO
class BaseTrainer(abc.ABC):
    """
    Parent class for all trainer classes. Here we initialize all general training components
    such as model, optimizer, dataloaders, etc.

    Attributes
    ----------
    _result : Result
        Result object to store the training results

    Methods
    -------
    train():
        Abstract method to train the model


    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[Union[BaseDataset, None]],
        result: Result,
        config: Optional[Union[None, DefaultConfig]],
        called_from: str,
    ):
        # passed attributes --------------------------
        self._trainset = trainset
        self._called_from = called_from
        self._validset = validset
        self._result = Result()
        self._config = config
        self._handle_reproducibility()
        self._input_validation()

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
        self._get_model_architecture()

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
        if self._config.reproducible in ["all", "training"]:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed=self._config.global_seed)
            if self._device == "cuda":
                torch.cuda.manual_seed(seed=self._config.global_seed)
                torch.cuda.manual_seed_all(seed=self._config.global_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            elif self._device == "mps":
                print("warning: MPS backend does not support reproducibility")
            else:
                print("cpu not relevant here")

    def _get_model_architecture(self):
        """Populates the model attribute with the appropriate model architecture"""
        if self._called_from == "Vanillix":
            from autoencodix.modeling._vanillix_architecture import VanillixArchitecture

            print(f"getting model for {self._called_from}")

            self._model = VanillixArchitecture(
                config=self._config, input_dim=self._input_dim
            )

        else:
            raise NotImplementedError(
                f"Model architecture for {self._called_from} is not implemented, only Vanillix is supported."
            )

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
