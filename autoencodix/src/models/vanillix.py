from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData #type: ignore
from typing import Dict, Any

from autoencodix.src.core._base_pipeline import BasePipeline
from autoencodix.src.core._default_config import DefaultConfig
from autoencodix.src.core._result import Result
from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.core._vanillix_architecture import VanillixArchitecture
from autoencodix.src.data._datasplitter import DataSplitter
from autoencodix.src.data._datasetcontainer import DataSetContainer
from autoencodix.src.data._numeric_dataset import NumericDataset
from autoencodix.src.trainers._trainer import Trainer





class Predictor:
    """
    Handles prediction process
    """

    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Predictor

        Parameters:
        -----------
        model : torch.nn.Module
            Trained model for prediction
        config : Optional[Dict[str, Any]]
            Configuration dictionary for prediction
        """
        self.model = model
        self.config = config or {"batch_size": 32}

    def predict(self):
        pass


class Vanillix(BasePipeline):
    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        config: Optional[DefaultConfig] = None,
        data_splitter: Optional[DataSplitter] = None,
    ):
        super().__init__(data, config)
        self.data = data
        self.x: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.model = VanillixArchitecture(config=self.config)
        self.preprocessor = Preprocessor()
        self.trainer = Trainer()
        self.config = config
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
        split_indices = self.data_splitter.split(self.x)
        train_ids, valid_ids, test_ids = (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        )

        self._datasets = DataSetContainer(
            train=NumericDataset(self.x[train_ids]),
            valid=NumericDataset(self.x[valid_ids]),
            test=NumericDataset(self.x[test_ids]),
        )

    def preprocess(self) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self.x attribute with the preprocessed data as a numpy array.

        """
        self.x = self.preprocessor.preprocess(self.data)
        self.result.preprocessed_data = self.x

    def fit(
        self,
    ) -> None:
        """
        Trains the model using the training data and updates the model attribute.
        """
        if self.x is None:
            print(
                "Warning: data not preprocessed. Preprocessing data now."
            )  # TODO replace with logger
            self.preprocess()

        self._build_datasets()
        self.model = self.trainer.train(
            model=self.model, train=self._datasets.train, valid=self._datasets.valid
        )

        self._is_fitted = True

    def predict(
        self, data: Optional[Union[np.ndarray, pd.DataFrame, AnnData]] = None
    ) -> None:
        """
        Prediction method with flexibility
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # If no data is provided, use test dataset
        if data is None:
            # Explicitly check for test dataset
            if "test" not in self._datasets:
                raise ValueError(
                    "No test dataset available. "
                    "Ensure a test split was created during fitting, or pass new data for prediction."
                )
            # should return mu
            prediction = self.predictor.predict(self._datasets.test)

        # test if data has correct type
        elif not isinstance(data, (np.ndarray, pd.DataFrame, AnnData)):
            raise TypeError(
                f"Expected data type to be one of np.ndarray, AnnData, or pd.DataFrame, got {type(data)}."
            )
        else:
            # Preprocess and create dataset for new data
            processed_data = self.preprocessor.preprocess(data)
