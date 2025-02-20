import abc
from typing import List, Optional, Tuple, Type, Union, Dict

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore
from scipy.sparse import issparse

from autoencodix.data._datapackage import DataPackage
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader
from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.utils._screader import SingleCellDataReader
from autoencodix.utils.default_config import DataCase, DefaultConfig
from autoencodix.data._filter import DataFilter
from autoencodix.data._nanremover import NaNRemover


class BasePreprocessor(abc.ABC):
    def __init__(self, config: Optional[DefaultConfig] = None):
        self.config = config

    def _validata_data(self) -> None:
        pass

    def _unspecific_preprocess(self) -> None:
        #
        self._data_package = self._fill_dataclass()
        self._data_package = self._nanremover.remove_nans(data=self._data_package)

    def preprocess(
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray, List[np.ndarray]],
        data_splitter: DataSplitter,
        config: Optional[DefaultConfig],
        dataset_type: Type,
        split: bool = True,
    ) -> Tuple[DatasetContainer, torch.Tensor]:
        self._data_splitter = data_splitter
        self._dataset_type = dataset_type
        self.config = config
        self._data_package = self._fill_dataclass()
        self._nanremover = NaNRemover(
            relevant_cols=self.config.data_config.annotation_columns
        )
        self._data_package = self._nanremover.remove_nans(data=self._data_package)
        if not self.config.paired_translation:
            raise ValueError("Unpaired translation is not supported as of now")
        n_samples = self._data_package.get_n_samples(
            is_paired=self.config.paired_translation
        )

        self._split_indicies = self._data_splitter.split(
            n_samples=n_samples["paired_count"]
        )
        if self.config.paired_translation:
            self._package_splitter = DataPackageSplitter(
                data_package=self._data_package, indicies=self._split_indicies
            )
        else:
            raise NotImplementedError(
                "Unpaired translation is not supported as of now"
            )  # TODO

        self._train_packge = self._package_splitter._split_data_package(
            indices=self._split_indicies["train"]
        )
        self._test_package = self._package_splitter._split_data_package(
            indices=self._split_indicies["test"]
        )
        self._valid_package = self._package_splitter._split_data_package(
            indices=self._split_indicies["valid"]
        )
        self._filter_and_scale()  # applies filtering and scaling to each split
        if self.config.data_case == DataCase.MULTI_BULK:
            self._build_numeric_datasets()
            return self._datasets

        return self._datasets

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

        from_key, to_key = None, None
        for k, v in self.config.data_config.data_info.items():
            if v.translate_direction is None:
                continue
            if v.translate_direction == "from":
                from_key = k
            if v.translate_direction == "to":
                to_key = k
        if not self.config.paired_translation:
            raise ValueError("Unpaired translation is not supported as of now")

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        if datacase == DataCase.MULTI_SINGLE_CELL:
            # UNPAIRED case done via required_common_cells = False in config
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            return result

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        elif datacase == DataCase.MULTI_BULK:
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
            if from_key in bulk_dfs.keys():
                result.from_modality = bulk_dfs[from_key]
                result.to_modality = result.img
            elif to_key in bulk_dfs.keys():
                result.to_modality = bulk_dfs[to_key]
                result.from_modality = result.img
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_IMG:
            adata = screader.read_data(config=self.config)
            images = imgreader.read_data(config=self.config)
            annotation = adata.obs
            result.multi_sc = adata
            result.img = images
            result.annotation = annotation
            if from_key in adata.mod.keys():
                result.from_modality = adata.mod[from_key]
                result.to_modality = result.img
            elif to_key in adata.mod.keys():
                result.to_modality = adata.mod[to_key]
                result.from_modality = result.img
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_SINGLE_CELL:
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            result.to_modality = adata.mod[to_key]
            result.from_modality = adata.mod[from_key]
            return result
        elif datacase == DataCase.BULK_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data(config=self.config)
            result.multi_bulk = bulk_dfs
            result.annotation = annotation
            result.from_modality = bulk_dfs[from_key]
            result.to_modality = bulk_dfs[to_key]
            return result
        else:
            raise ValueError("Non valid data case")

    def _filter_and_scale(self):
        if (
            self.config.data_case == DataCase.MULTI_BULK
            or self.config.data_case == DataCase.BULK_TO_BULK
            or self.config.data_case == DataCase.IMG_TO_BULK
        ):
            if not self._train_packge.is_empty():
                self._train_packge = self.scale_filter_multi_bulk(
                    package=self._train_packge
                )
            if not self._valid_package.is_empty():
                self._valid_package = self.scale_filter_multi_bulk(
                    package=self._valid_package
                )
            if not self._test_package.is_empty():
                self._test_package = self.scale_filter_multi_bulk(
                    package=self._test_package
                )

    def scale_filter_multi_bulk(self, package: DataPackage):
        filtered_multi_bulk = {}
        for k, v in self.config.data_config.data_info.items():
            print(k)
            if v.data_type == "ANNOTATION" or v.data_type == "IMG":
                continue
            elif v.data_type == "NUMERIC":
                df = package.multi_bulk[k]
                # v.k_filter = 4
                data_filter = DataFilter(df=df, data_info=v)
                filtered_df = data_filter.filter()
                print(f"filtered_df k{k}.shape: {filtered_df.shape}")
                scaled_df = data_filter.scale(filtered_df)
                filtered_multi_bulk[k] = scaled_df
                # filtered_multi_bulk[k] = filtered_df
        package.multi_bulk = filtered_multi_bulk
        return package

    def _build_numeric_datasets(self) -> None:
        packages = {
            "train": self._train_packge,
            "valid": self._valid_package,
            "test": self._test_package,
        }
        helper = {}
        for k, v in packages.items():
            if v.is_empty():
                helper[k] = None
                continue
            dfs = [v.multi_bulk[key].values for key in v.multi_bulk.keys()]
            # concatenate all dfs
            features = np.concatenate(dfs, axis=1)
            labels = v.annotation.index.to_list()
            indices = self._split_indicies[k]
            # features to torch tensor
            features = torch.tensor(features, dtype=torch.float32)
            dataset = NumericDataset(
                data=features,
                config=self.config,
                ids=labels,
                split_ids=indices,
                data_package=v,
            )
            helper[k] = dataset
        self._datasets = DatasetContainer(
            train=helper["train"], valid=helper["valid"], test=helper["test"]
        )

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
