from typing import Dict, Optional, Type, Union, List

import numpy as np
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_loss import BaseLoss
from autoencodix.data.datapackage import DataPackage
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_evaluator import BaseEvaluator
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.utils._losses import VanillixLoss
from autoencodix.visualize._general_visualizer import GeneralVisualizer


class Vanillix(BasePipeline):
    """Vanillix specific version of the BasePipeline class.

    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.
    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to VanillixConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[BaseAutoencoder] = VanillixArchitecture,
        loss_type: Type[BaseLoss] = VanillixLoss,
        preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor,
        visualizer: Type[BaseVisualizer] = GeneralVisualizer,
        evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
        ontologies: Optional[Union[List, Dict]] = None,
    ) -> None:
        """Initialize Vanillix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        See implementation of parent class for list of full Args.
        """
        self._default_config = VanillixConfig()
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
            ontologies=ontologies,
        )

    def sample_latent_space(
        self,
        n_samples: int,
        split: str = "test",
        epoch: int = -1,
    ) -> torch.Tensor:
        """Samples latent space points from the empirical latent distribution.

        This method draws new latent points by fitting a diagonal Gaussian
        distribution to the latent codes of the specified split and epoch, and
        sampling from it. This enables approximate generative sampling for
        autoencoders that do not model uncertainty explicitly.

        Args:
            n_samples: The number of latent points to sample. Must be a positive
                integer.
            split: The split to sample from (train, valid, test), default is test.
            epoch: The epoch to sample from, default is the last epoch (-1).

        Returns:
            z: torch.Tensor - The sampled latent space points.

        Raises:
            ValueError: If the model has not been trained, latent codes have not
                been computed, or n_samples is not a positive integer.
            TypeError: If the stored latent codes are not numpy arrays.
        """

        if not hasattr(self, "_trainer") or self._trainer is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        if self.result.latentspaces is None:
            raise ValueError("Model has no stored latent codes for sampling.")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        Z = self.result.latentspaces.get(split=split, epoch=epoch)

        if not isinstance(Z, np.ndarray):
            raise TypeError(
                f"Expected latent codes to be of type numpy.ndarray, got {type(Z)}."
            )

        Z_t = torch.from_numpy(Z).to(
            device=self._trainer._model.device,
            dtype=self._trainer._model.dtype,
        )

        with torch.no_grad():
            # Fit empirical diagonal Gaussian
            global_mu = Z_t.mean(dim=0)
            global_std = Z_t.std(dim=0)

            eps = torch.randn(
                n_samples,
                Z_t.shape[1],
                device=Z_t.device,
                dtype=Z_t.dtype,
            )

            z = global_mu + eps * global_std
            return z
