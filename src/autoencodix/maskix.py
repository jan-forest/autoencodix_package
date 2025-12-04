from typing import Dict, Optional, Type, Union, Callable, Any
import numpy as np
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_evaluator import BaseEvaluator
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.evaluate._general_evaluator import GeneralEvaluator

from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.maskix_config import MaskixConfig
from autoencodix.utils._losses import MaskixLoss
from autoencodix.modeling._maskix_architecture import MaskixArchitectureVanilla
from autoencodix.trainers._maskix_trainer import MaskixTrainer
from autoencodix.visualize._general_visualizer import GeneralVisualizer
from autoencodix.utils._model_output import ModelOutput


class Maskix(BasePipeline):
    """Maskix specific version of the BasePipeline class.

    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to OntixConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = MaskixTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[BaseAutoencoder] = MaskixArchitectureVanilla,
        loss_type: Type[BaseLoss] = MaskixLoss,
        preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor,
        visualizer: Type[BaseVisualizer] = GeneralVisualizer,
        evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
        masking_fn: Optional[Callable] = None,
        masking_fn_kwargs: Dict[str, Any] = {},
        **kwargs: dict,
    ) -> None:
        """Initialize Maskix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        See parent class for full list of Arguments.

        Raises:
            TypeError: if ontologies are not a Tuple or List.

        """
        self._default_config = MaskixConfig()

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
            masking_fn=masking_fn,
            masking_fn_kwargs=masking_fn_kwargs,
            **kwargs,
        )

    def impute(self, corrupted_tensor: torch.Tensor) -> ModelOutput:
        """Impute missing values in the corrupted tensor using the trained Maskix model.

        Args:
            corrupted_tensor: A tensor with missing values to be imputed.
        Returns:
            model_output: A tensor with imputed values.
        """
        if self._trainer is None:
            raise ValueError(
                "Model needs to be trained before imputation., Run fit() first."
            )
        self._trainer._model.eval()
        with torch.no_grad(), self._trainer._fabric.autocast():
            corrupted_tensor = corrupted_tensor.to(self._trainer._model.device)
            model_output = self._trainer._model(corrupted_tensor)
        return model_output
