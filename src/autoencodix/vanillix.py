from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.data._datasetcontainer import DataSetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.preprocessor import Preprocessor
from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.trainers._vanillix_trainer import VanillixTrainer
from autoencodix.trainers.predictor import Predictor
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
    _predictor : Predictor
        Predictor object that uses trained model on new data (custom for Vanillix)
    _trainer : VanillixTrainer
        Trainer object that trains the model (custom for Vanillix)
    _evaluator : Evaluator
        Evaluator object that evaluates the model performance or downstream tasks (custom for Vanillix)
    result : Result
        Result object to store the pipeline results
    _datasets : Optional[DataSetContainer]
        Container for train, validation, and test datasets (preprocessed)
    _is_fitted : bool
        Flag to check if the model has been trained
    _id : str
        Identifier for the pipeline (Vanillix)
    data_splitter : DataSplitter
        DataSplitter object to split the data into train, validation, and test sets

    """

    _dataset_class: Type[BaseDataset] = NumericDataset
    _trainer_class: Type[BaseTrainer] = VanillixTrainer

    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
    ) -> None:
        super().__init__(data, config)
        self._id = "Vanillix"
        self.data = data
        self._preprocessor = Preprocessor()
        self._visualizer = Visualizer()
        self._predictor = Predictor()
        self._trainer: self._trainer_class
        self._evaluator = Evaluator()
        self.result = Result()

        self._datasets = None
        self._is_fitted = False
        self.data_splitter = DataSplitter(
            config=self.config, custom_splits=custom_splits
        )

    def _build_datasets(self) -> None:
        """
        Fills the datasets attribute with train, validation, and test datasets.
        After callind the dataset_splitter, the data is split into train, validation, and test sets.

        Returns:
            None

        """
        split_indices = self.data_splitter.split(self._features)
        train_ids, valid_ids, test_ids = (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        )

        self._datasets = DataSetContainer(
            train=self._dataset_class(data=self._features[train_ids], float_precision=self.config.float_precision),
            valid=self._dataset_class(data=self._features[valid_ids], float_precision=self.config.float_precision),
            test=self._dataset_class(data=self._features[test_ids], float_precision=self.config.float_precision),
        )
        self.result.datasets = self._datasets
