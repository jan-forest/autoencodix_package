import abc
from typing import List, Optional, Type, Union, Tuple

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore

from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.utils.default_config import DefaultConfig


class BasePreprocessor(abc.ABC):
    def __init__(self):

        pass

    def preprocess(
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray, List[np.ndarray]],
        data_splitter: DataSplitter,
        config: Optional[DefaultConfig],
        dataset_type: Type,
        split: bool = True,
    ) -> Tuple[DatasetContainer, torch.Tensor]:
        """
        Main method of the class. Handles the following steps:
        1. align the data (in case we have multiple datasets with different sample ids)
        2. split the data into training and validation sets
        3. deals with missing values in each split and dataset
        4. normalizes or standardizes the data in each split and dataset
        5. applies filtering (like variance thresholding) in each split and dataset

        Parameters:
            data : Union[pd.DataFrame, AnnData, np.ndarray, List[np.ndarray]]
                Input data from the user
            data_splitter : Optional[DataSplitter]
                DataSplitter object to split the data into train, validation, and test sets
            config : Optional[DefaultConfig]
                Configuration object containing customizations for the pipeline

        Returns:
            DatasetContainer: Container for train, validation, and test datasets (preprocessed)

        """
        # not implemented yet for pandas, AnnData, or List[np.ndarray]
        if isinstance(data, pd.DataFrame):
            raise NotImplementedError(
                "Preprocessing for pandas DataFrames is not implemented yet, only numpy arrays are supported."
            )
        if isinstance(data, AnnData):
            raise NotImplementedError(
                "Preprocessing for AnnData objects is not implemented yet, only numpy arrays are supported."
            )
        if isinstance(data, List):
            raise NotImplementedError(
                "Preprocessing for List of numpy arrays is not implemented yet, only single numpy arrays are supported."
            )
        self._data_splitter = data_splitter
        self.config = config
        self._extract_metadata_numpy(data=data)
        self._features = self._preprocess_numpy(data=data)
        self._dataset_type = dataset_type
        if split:
            self._build_datasets()  # populates self._datasets
        return self._datasets, self._features

    def _preprocess_numpy(self, data: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(data)
        return t

    def _align_numpy_data(self, data: List[np.ndarray]) -> np.ndarray:
        # TODO
        return np.array(data)

    def align_ann_data(self, data: AnnData) -> np.ndarray:
        # TODO
        return np.array(data)

    def _extract_metadata_pandas(
        self, data: Union[pd.DataFrame, AnnData]
    ) -> np.ndarray:
        # TODO
        return np.array(data)

    def _extract_metadata_anndata(self, data: AnnData) -> np.ndarray:
        # TODO
        return np.array(data)

    def _extract_metadata_numpy(self, data: np.ndarray) -> np.ndarray:
        self._ids = None  # TODO
        # TODO
        return np.array(data)

    def _build_datasets(self) -> None:
        """
        Build datasets for training, validation, and testing.

        Raises
        ------
        NotImplementedError
            If the data splitter is not initialized.
        ValueError
            If self._features is None.
        """

        if self._features is None:
            raise ValueError("No data available for splitting")

        split_indices = self._data_splitter.split(self._features)
        if self._ids is None:
            train_ids, valid_ids, test_ids = None, None, None
        else:
            train_ids = (
                None
                if len(split_indices["train"]) == 0
                else self._ids[split_indices["train"]]
            )
            valid_ids = (
                None
                if len(split_indices["valid"]) == 0 or self._ids is None
                else self._ids[split_indices["valid"]]
            )
            test_ids = (
                None
                if len(split_indices["test"]) == 0 or self._ids is None
                else self._ids[split_indices["test"]]
            )

        train_data = (
            None
            if len(split_indices["train"]) == 0
            else self._features[split_indices["train"]]
        )
        valid_data = (
            None
            if len(split_indices["valid"]) == 0
            else self._features[split_indices["valid"]]
        )

        test_data = (
            None
            if len(split_indices["test"]) == 0
            else self._features[split_indices["test"]]
        )
        self._datasets = DatasetContainer(
            train=self._dataset_type(
                data=train_data, config=self.config, ids=train_ids
            ),
            valid=self._dataset_type(
                data=valid_data, config=self.config, ids=valid_ids
            ),
            test=self._dataset_type(data=test_data, config=self.config, ids=test_ids),
        )
