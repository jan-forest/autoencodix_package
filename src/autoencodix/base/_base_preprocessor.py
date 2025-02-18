import abc
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore
import scanpy as sc

from autoencodix.data._datapackage import DataPackage
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader
from autoencodix.utils._screader import SingleCellDataReader
from autoencodix.utils.default_config import DataCase, DefaultConfig


class BasePreprocessor(abc.ABC):
    def __init__(self, config: Optional[DefaultConfig] = None):
        self.config = config

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
        self._features = self._preprocess_numpy(data=data)
        self._data_package = self._fill_dataclass(config=self.config)
        self._dataset_type = dataset_type
        if split:
            self._build_datasets()  # populates self._datasets

        return self._datasets, self._features

    def _filter_sc_data(adata: AnnData) -> AnnData:
        sc.pp.filter_genes(adata, min_cells=int(adata.shape[0] * 0.01))

    def _fill_dataclass(self) -> DataPackage:
        """
        Fills a DataPackage object based on the provided configuration.

        Args:
            config (DefaultConfig): Configuration object containing the data case and other settings.

        Returns:
            DataPackage: An object containing the loaded data based on the specified data case.

        Raises:
            ValueError: If the data case is not supported or if unpaired translation is requested.

        Supported Data Cases:
            - DataCase.MULTI_SINGLE_CELL
            - DataCase.SINGLE_CELL_TO_SINGLE_CELL
            - DataCase.MULTI_BULK
            - DataCase.BULK_TO_BULK
            - DataCase.IMG_TO_BULK
            - DataCase.SINGLE_CELL_TO_IMG
        """
        result = DataPackage()
        bulkreader = BulkDataReader()
        screader = SingleCellDataReader()
        imgreader = ImageDataReader()
        datacase = self.config.data_case
        print(f"datacase: {datacase}")
        if not self.config.paired_translation:
            raise ValueError("Unpaired translation is not supported as of now")

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        if (
            datacase == DataCase.MULTI_SINGLE_CELL
            or datacase == DataCase.SINGLE_CELL_TO_SINGLE_CELL
        ):
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            return result

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        elif datacase == DataCase.MULTI_BULK or DataCase.BULK_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data(config=self.config)
            result.multi_bulk = bulk_dfs
            result.annotation = annotation
            return result

        # TRANSLATION CASES
        elif datacase == DataCase.IMG_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data(config=self.config)
            images = imgreader.read_data(config=self.config)
            result.multi_bulk = bulk_dfs
            result.annotation = annotation
            result.img = images
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_IMG:
            adata = screader.read_data(config=self.config)
            images = imgreader.read_data(config=self.config)
            annotation = adata.obs
            result.multi_sc = adata
            result.img = images
            result.annotation = annotation
            return result
        else:
            raise ValueError("Non valid data case")


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
