from typing import Dict, Optional, Type, Union
import torch
import numpy as np

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._utils import config_method
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_evaluator import BaseEvaluator
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.visualize._general_visualizer import GeneralVisualizer


class Varix(BasePipeline):
    """Varix specific version of the BasePipeline class.

    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.
    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to VarixConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[BaseAutoencoder] = VarixArchitecture,
        loss_type: Type[BaseLoss] = VarixLoss,
        preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor,
        visualizer: Type[BaseVisualizer] = GeneralVisualizer,
        evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """Initialize Varix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        See parent class for full list of Args.

        """
        self._default_config = VarixConfig()
        super().__init__(
            data=data,
            dataset_type=dataset_type,
            trainer_type=trainer_type,
            model_type=model_type,
            loss_type=loss_type,
            preprocessor_type=preprocessor_type,
            visualizer=visualizer,
            evaluator=evaluator,
            result=result,
            datasplitter_type=datasplitter_type,
            config=config,
            custom_split=custom_splits,
        )

    @config_method(
        valid_params={
            "config",
            "batch_size",
            "epochs",
            "learning_rate",
            "n_workers",
            "device",
            "n_gpus",
            "gpu_strategy",
            "weight_decay",
            "reproducible",
            "global_seed",
            "reconstruction_loss",
            "checkpoint_interval",
        }
    )
    def sample_latent_space(
        self, config, split: str = "test", epoch: int = -1
    ) -> torch.Tensor:
        """Samples new latent space points from the learned distribution.

        Args:
            split: The split to sample from (train, valid, test), default is test
            epoch: The epoch to sample from, default is the last epoch (-1)
        Returns:
            z: torch.Tensor - The sampled latent space points
        Raises:
            ValueError: if model has not been trained.
            TypeError: if mu and or logvar are no numpy arrays
        """

        if not hasattr(self, "_trainer") or self._trainer is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        if self.result.mus is None or self.result.sigmas is None:
            raise ValueError("Model has not learned the latent space distribution yet.")
        mu = self.result.mus.get(split=split, epoch=epoch)
        logvar = self.result.sigmas.get(split=split, epoch=epoch)

        if not isinstance(mu, np.ndarray):
            raise TypeError(
                f"Expected value to be of type numpy.ndarray, got {type(mu)}."
            )
        if not isinstance(logvar, np.ndarray):
            raise TypeError(
                f"Expected value to be of type numpy.ndarray, got {type(logvar)}."
            )
        mu_t = torch.from_numpy(mu)
        logvar_t = torch.from_numpy(logvar)

        # Move to same device and dtype as model
        mu_t = mu_t.to(
            device=self._trainer._model.device, dtype=self._trainer._model.dtype
        )
        logvar_t = logvar_t.to(
            device=self._trainer._model.device, dtype=self._trainer._model.dtype
        )

        with self._trainer._fabric.autocast(), torch.no_grad():
            z = self._trainer._model.reparameterize(mu_t, logvar_t)
            return z
