from typing import Dict, Optional, Type, Union
from typing import Dict, Optional, Type, Union
import torch
import numpy as np

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._utils import config_method
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._image_dataset import ImageDataset
from autoencodix.data._image_processor import ImagePreprocessor
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.visualize._imagix_visualizer import ImagixVisualizer


class Imagix(BasePipeline):
    """Imagix specific version of the BasePipeline class.

    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to DefaultConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = ImageDataset,
        model_type: Type[BaseAutoencoder] = ImageVAEArchitecture,
        loss_type: Type[BaseLoss] = VarixLoss,
        preprocessor_type: Type[BasePreprocessor] = ImagePreprocessor,
        visualizer: Optional[Type[BaseVisualizer]] = ImagixVisualizer,
        evaluator: Optional[Type[GeneralEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """Initialize Imagix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        See Parentclass for full list of Args.

        """
        self._default_config = DefaultConfig()
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