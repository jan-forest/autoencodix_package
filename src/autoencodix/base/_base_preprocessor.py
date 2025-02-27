import abc
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore
from scipy.sparse import issparse

from autoencodix.data._datapackage import DataPackage
from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._filter import DataFilter
from autoencodix.data._nanremover import NaNRemover
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader
from autoencodix.utils._screader import SingleCellDataReader
from autoencodix.utils.default_config import DataCase, DefaultConfig


class BasePreprocessor(abc.ABC):
    def __init__(self, config: DefaultConfig):
        self.config = config

    def _validata_data(self) -> None:
        pass

    def _unspecific_preprocess(self) -> None:
        #
        self._data_package = self._fill_dataclass()
        self._data_package = self._nanremover.remove_nans(data=self._data_package)

    def preprocess(
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray, List[np.ndarray]] = None,
        data_splitter: DataSplitter = None,
        dataset_type: Type = NumericDataset,
        split: bool = True,
    ) -> Tuple[DatasetContainer, torch.Tensor]:
        self._data_splitter = DataSplitter(config=self.config) if data_splitter is None else data_splitter
        self._dataset_type = dataset_type
        self._data_package = self._fill_dataclass()
        self._nanremover = NaNRemover(
            relevant_cols=self.config.data_config.annotation_columns
        )
        self._data_package = self._nanremover.remove_nan(data=self._data_package)
        n_samples = self._data_package.get_n_samples(
            is_paired=self.config.paired_translation
        )

        if self.config.paired_translation:
            self._split_indicies = self._data_splitter.split(
                n_samples=n_samples["paired_count"]
            )
            self._package_splitter = DataPackageSplitter(
                data_package=self._data_package,
                indices=self._split_indicies,
                config=self.config,
            )
            self._split_packages = self._package_splitter.split()
        else:
            self._from_indices = self._data_splitter.split(
                n_samples=n_samples["from"], is_paired=False
            )
            self._to_indices = self._data_splitter.split(
                n_samples=n_samples["to"], is_paired=False
            )
            self._package_splitter = DataPackageSplitter(
                data_package=self._data_package,
                config=self.config,
                from_indices=self._from_indices,
                to_indices=self._to_indices,
            )
            self._split_packages = self._package_splitter.split()

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
        bulkreader = BulkDataReader(config=self.config)
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

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        if datacase == DataCase.MULTI_SINGLE_CELL:
            # UNPAIRED case done via required_common_cells = False in config
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            return result

        # even if reading is the same for these two cases the validation is different, thats why we have them separated
        elif datacase == DataCase.MULTI_BULK:
            bulk_dfs, annotation = bulkreader.read_data()
            result.multi_bulk = bulk_dfs
            result.annotation = annotation
            return result

        # TRANSLATION CASES
        elif datacase == DataCase.IMG_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data()

            images = imgreader.read_data(config=self.config)
            result.multi_bulk = bulk_dfs
            result.img = images
            if from_key in bulk_dfs.keys():
                # direction BULK -> IMG
                result.from_modality = bulk_dfs[from_key]
                result.to_modality = result.img
                # I know that I have only one bulk data so I can use the first key
                from_annotation = next(iter(annotation.keys()))
                result.annotation = {"from": annotation[from_annotation], "to": None}

            elif to_key in bulk_dfs.keys():
                # direction IMG -> BULK
                result.to_modality = bulk_dfs[to_key]
                result.from_modality = result.img
                # I know that I have only one bulk data so I can use the first key
                to_annotation = next(iter(annotation.keys()))
                result.annotation = {"from": None, "to": annotation[to_annotation]}
                print(f"annotation keys: {annotation.keys()}")

            # to keep unambigous remove on translation relevant and duplicate data
            result.multi_bulk = None
            result.img = None
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_IMG:
            adata = screader.read_data(config=self.config)
            images = imgreader.read_data(config=self.config)
            result.multi_sc = adata
            result.img = images
            if from_key in adata.mod.keys():
                result.from_modality = adata.mod[from_key]
                result.to_modality = result.img
            elif to_key in adata.mod.keys():
                result.to_modality = adata.mod[to_key]
                result.from_modality = result.img

            # to keep unambigous remove on translation relevant and duplicate data
            result.annotation = None
            result.img = None
            result.multi_sc = None
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_SINGLE_CELL:
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            result.to_modality = adata.mod[to_key]
            result.from_modality = adata.mod[from_key]
            result.mulit_sc = None
            return result
        elif datacase == DataCase.BULK_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data()
            result.multi_bulk = bulk_dfs
            to_annotation = annotation[to_key]
            from_annotation = annotation[from_key]
            result.annotation = {"to": to_annotation, "from": from_annotation}
            result.from_modality = bulk_dfs[from_key]
            result.to_modality = bulk_dfs[to_key]
            result.multi_bulk = None
            return result
        elif datacase == DataCase.IMG_TO_IMG:
            images = imgreader.read_data(config=self.config)
            result.img = images
            result.from_modality = images[from_key]
            result.to_modality = images[to_key]
            result.img = None
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
