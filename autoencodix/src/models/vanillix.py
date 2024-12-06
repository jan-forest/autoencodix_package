from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore

from autoencodix.src.core._base_pipeline import BasePipeline
from autoencodix.src.core._vanillix_architecture import VanillixArchitecture
from autoencodix.src.data._datasetcontainer import DataSetContainer
from autoencodix.src.data._datasplitter import DataSplitter
from autoencodix.src.data._numeric_dataset import NumericDataset
from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.trainers.trainer import Trainer
from autoencodix.src.trainers.predictor import Predictor
from autoencodix.src.utils._result import Result
from autoencodix.src.utils.default_config import DefaultConfig, config_method
from autoencodix.src.visualize.visualize import Visualizer
from autoencodix.src.evaluate.evaluate import Evaluator


class Vanillix(BasePipeline):
    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        config: Optional[Union[None, DefaultConfig]] = None,
        data_splitter: Optional[DataSplitter] = None,
    ):
        super().__init__(data, config)
        self.data = data
        self._model = VanillixArchitecture(config=self.config)
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
            train=NumericDataset(self._features[train_ids]),
            valid=NumericDataset(self._features[valid_ids]),
            test=NumericDataset(self._features[test_ids]),
        )
