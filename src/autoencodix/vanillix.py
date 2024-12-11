from typing import Optional, Union, Type

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore

from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.data._datasetcontainer import DataSetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.preprocessor import Preprocessor
from autoencodix.trainers.trainer import Trainer
from autoencodix.trainers.predictor import Predictor
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.visualize.visualize import Visualizer
from autoencodix.evaluate.evaluate import Evaluator


class Vanillix(BasePipeline):
    _dataset_class: Type[BaseDataset] = NumericDataset

    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        config: Optional[Union[None, DefaultConfig]] = None,
        data_splitter: Optional[DataSplitter] = None,
    ):
        super().__init__(data, config)
        self.data = data
        self._preprocessor = Preprocessor()
        self._trainer = Trainer()
        self._visualizer = Visualizer()
        self._predictor = Predictor()
        self._evaluator = Evaluator()
        self.result = Result()

        self._datasets = None
        self._is_fitted = False
        if data_splitter is None:
            self.data_splitter = DataSplitter()

    def _build_datasets(self):
        """
        Takes the self.x attribute and creates train, valid, and test datasets.
        If no Datasplitter is provided, it will split the data according to the default configuration.


        """
        split_indices = self.data_splitter.split(self._features)
        train_ids, valid_ids, test_ids = (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        )

        self._datasets = DataSetContainer(
            train=self._dataset_class(self._features[train_ids]),
            valid=self._dataset_class(self._features[valid_ids]),
            test=self._dataset_class(self._features[test_ids]),
        )
        self.result.datasets = self._datasets
