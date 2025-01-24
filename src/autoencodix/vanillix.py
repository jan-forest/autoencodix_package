from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.preprocessor import Preprocessor
from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.trainers._vanillix_trainer import VanillixTrainer
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.visualize.visualize import Visualizer


class Vanillix(BasePipeline):
    """
    Vanillix specific version of the BasePipeline class.
    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

    Attributes
    ----------
    data : Union[np.ndarray, AnnData, pd.DataFrame]
        Input data from the user
    config : Optional[Union[None, DefaultConfig]]
        Configuration object containing customizations for the pipeline
    _preprocessor : Preprocessor
        Preprocessor object to preprocess the input data (custom for Vanillix)
    _visualizer : Visualizer
        Visualizer object to visualize the model output (custom for Vanillix)
    _trainer : VanillixTrainer
        Trainer object that trains the model (custom for Vanillix)
    _evaluator : Evaluator
        Evaluator object that evaluates the model performance or downstream tasks (custom for Vanillix)
    result : Result
        Result object to store the pipeline results
    _datasets : Optional[DatasetContainer]
        Container for train, validation, and test datasets (preprocessed)
    data_splitter : DataSplitter
        DataSplitter object to split the data into train, validation, and test sets
    """

    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        trainer_type: Type[BaseTrainer] = VanillixTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[BaseAutoencoder] = VanillixArchitecture,
        preprocessor: Optional[Preprocessor] = None,
        visualizer: Optional[BaseVisualizer] = None,
        evaluator: Optional[Evaluator] = None,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """Initialize Vanillix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        Parameters
        ----------
        data : Union[np.ndarray, AnnData, pd.DataFrame]
            Input data to be processed
        trainer_type : Type[BaseTrainer], optional
            Type of trainer to be instantiated during fit step, default is VanillixTrainer
        dataset_type : Type[BaseDataset], optional
            Type of dataset to be instantiated post-preprocessing, default is NumericDataset
        preprocessor : Optional[Preprocessor]
            For data preprocessing, default creates new Preprocessor
        visualizer : Optional[Visualizer]
            For result visualization, default creates new Visualizer
        evaluator : Optional[Evaluator]
            For model evaluation, default creates new Evaluator
        result : Optional[Result]
            Container for pipeline results, default creates new Result
        datasplitter_type : Type[DataSplitter], optional
            Type of splitter to be instantiated during preprocessing, default is DataSplitter
        custom_splits : Optional[Dict[str, np.ndarray]]
            Custom train/valid/test split indices
        config : Optional[DefaultConfig]
            Configuration for all pipeline components
        """
        super().__init__(
            data=data,
            dataset_type=dataset_type,
            trainer_type=trainer_type,
            model_type=model_type,
            preprocessor=preprocessor or Preprocessor(),
            visualizer=visualizer or Visualizer(),
            evaluator=evaluator or Evaluator(),
            result=result or Result(),
            datasplitter_type=datasplitter_type,
            config=config or DefaultConfig(),
            custom_split=custom_splits,
        )
