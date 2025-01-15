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

    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        trainer_type: Type[BaseTrainer] = VanillixTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        preprocessor: Optional[Preprocessor] = None,
        visualizer: Optional[Visualizer] = None,
        predictor: Optional[Predictor] = None,
        evaluator: Optional[Evaluator] = None,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        super().__init__(
            data=data,
            dataset_type=dataset_type,
            trainer_type=trainer_type,
            preprocessor=preprocessor or Preprocessor(),
            visualizer=visualizer or Visualizer(),
            predictor=predictor or Predictor(),
            evaluator=evaluator or Evaluator(),
            result=result or Result(),
            datasplitter_type=datasplitter_type,
            config=config or DefaultConfig(),
            custom_split=custom_splits,
        )

        self._id = "Vanillix"
        self.data = data

        self._datasets = None
        self._is_fitted = False

    def _build_datasets(self) -> None:
        """
        Fills the datasets attribute with train, validation, and test datasets.
        After callind the dataset_splitter, the data is split into train, validation, and test sets.

        Returns:
            None

        """
        split_indices = self._data_splitter.split(self._features)
        train_ids, valid_ids, test_ids = (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        )

        self._datasets = DataSetContainer(
            train=self._dataset_type(
                data=self._features[train_ids],
                float_precision=self.config.float_precision,
            ),
            valid=self._dataset_type(
                data=self._features[valid_ids],
                float_precision=self.config.float_precision,
            ),
            test=self._dataset_type(
                data=self._features[test_ids],
                float_precision=self.config.float_precision,
            ),
        )
        self.result.datasets = self._datasets
