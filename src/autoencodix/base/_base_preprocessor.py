import abc
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import mudata as md
from mudata import MuData
import pandas as pd
import torch
from anndata import AnnData  # type: ignore

from autoencodix.data._imgdataclass import ImgData
from autoencodix.data._datapackage import DataPackage
from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._sc_filter import SingleCellFilter
from autoencodix.data._filter import DataFilter
from autoencodix.data._nanremover import NaNRemover
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader, ImageNormalizer
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
        self._data_splitter = (
            DataSplitter(config=self.config) if data_splitter is None else data_splitter
        )
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

        for k, v in self._split_packages.items():
            data = v["data"]
            processed_data = self._filter_and_scale(data=data)
            self._split_packages[k]["data"] = processed_data
        return self._split_packages

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
                result.from_modality = {from_key: bulk_dfs[from_key]}
                result.to_modality = {to_key: result.img}
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
                result.from_modality = {from_key: adata.mod[from_key]}
                result.to_modality = {to_key: result.img}
            elif to_key in adata.mod.keys():
                result.to_modality = {to_key: adata.mod[to_key]}
                result.from_modality = {from_key: result.img}

            # to keep unambigous remove on translation relevant and duplicate data
            result.annotation = None
            result.img = None
            result.multi_sc = None
            return result
        elif datacase == DataCase.SINGLE_CELL_TO_SINGLE_CELL:
            adata = screader.read_data(config=self.config)
            result.multi_sc = adata
            result.to_modality = {to_key: adata.mod[to_key]}
            result.from_modality = {from_key: adata.mod[from_key]}
            result.mulit_sc = None
            return result
        elif datacase == DataCase.BULK_TO_BULK:
            bulk_dfs, annotation = bulkreader.read_data()
            result.multi_bulk = bulk_dfs
            to_annotation = annotation[to_key]
            from_annotation = annotation[from_key]
            result.annotation = {"to": to_annotation, "from": from_annotation}
            result.from_modality = {from_key: bulk_dfs[from_key]}
            result.to_modality = {to_key: bulk_dfs[to_key]}
            result.multi_bulk = None
            return result
        elif datacase == DataCase.IMG_TO_IMG:
            images = imgreader.read_data(config=self.config)
            result.img = images
            result.from_modality = {"from_key": images[from_key]}
            result.to_modality = {"to_key": images[to_key]}
            result.img = None
            return result
        else:
            raise ValueError("Non valid data case")

    def _filter_and_scale_dataframe(self, df, info_key):
        """Process a dataframe with filtering and scaling."""
        data_info = self.config.data_config.data_info[info_key]
        filter_obj = DataFilter(df=df, data_info=data_info)
        filtered = filter_obj.filter()
        return filter_obj.scale(filtered)

    def _normalize_image_data(self, images, info_key):
        """Process images with normalization."""
        scaling_method = self.config.data_config.data_info[info_key].scaling
        for img in images:
            img.img = ImageNormalizer.normalize_image(
                image=img.img, method=scaling_method
            )
        return images

    def _process_single_cell(self, mudata: MuData, key: str) -> MuData:
        """Process MuData with SingleCellFilter."""
        data_info = self.config.data_config.data_info[key]
        sc_preprocessor = SingleCellFilter(mudata=mudata, data_info=data_info)
        return sc_preprocessor.preprocess()

    def _filter_and_scale(self, data: DataPackage) -> DataPackage:
        """Filter and scale data based on data case type."""

        # Process MULTI_BULK case
        if self.config.data_case == DataCase.MULTI_BULK:
            if data.multi_bulk:
                processed = {}
                for key, df in data.multi_bulk.items():
                    processed[key] = self._filter_and_scale_dataframe(df, key)
                data.multi_bulk = processed

        # Process MULTI_SINGLE_CELL case
        elif self.config.data_case == DataCase.MULTI_SINGLE_CELL:
            if data.multi_sc:
                # Get data_info for single-cell data
                sc_key = list(self.config.data_config.data_info.keys())[0]
                data_info = self.config.data_config.data_info[sc_key]

                # Process with SingleCellFilter that includes DataFilter processing
                sc_filter = SingleCellFilter(mudata=data.multi_sc, data_info=data_info)
                data.multi_sc = sc_filter.preprocess()

        # Process BULK_TO_BULK case
        elif self.config.data_case == DataCase.BULK_TO_BULK:
            # Process both modalities
            data.process_modality("from", self._filter_and_scale_dataframe)
            data.process_modality("to", self._filter_and_scale_dataframe)

        # Process SINGLE_CELL_TO_SINGLE_CELL case
        elif self.config.data_case == DataCase.SINGLE_CELL_TO_SINGLE_CELL:
            # Process both single-cell modalities
            for direction in ["from", "to"]:
                modality_data = data.get_modality_data(direction)
                if modality_data is not None and isinstance(modality_data, MuData):
                    modality_key = data.get_modality_key(direction)
                    data_info = self.config.data_config.data_info[modality_key]

                    # Process with SingleCellFilter
                    sc_filter = SingleCellFilter(
                        mudata=modality_data, data_info=data_info
                    )
                    processed_data = sc_filter.preprocess()

                    # Update the modality with processed data
                    data.set_modality_data(direction, processed_data)

        # Process IMG_TO_BULK case
        elif self.config.data_case == DataCase.IMG_TO_BULK:
            # Process each modality with appropriate processor
            for direction in ["from", "to"]:
                modality_data = data.get_modality_data(direction)
                if modality_data is None:
                    continue

                modality_key = data.get_modality_key(direction)
                data_info = self.config.data_config.data_info[modality_key]

                if isinstance(modality_data, pd.DataFrame):
                    processed_data = self._filter_and_scale_dataframe(
                        modality_data, modality_key
                    )
                    data.set_modality_data(direction, processed_data)
                elif isinstance(modality_data, MuData):
                    sc_filter = SingleCellFilter(
                        mudata=modality_data, data_info=data_info
                    )
                    processed_data = sc_filter.preprocess()
                    data.set_modality_data(direction, processed_data)
                elif (
                    isinstance(modality_data, list)
                    and modality_data
                    and hasattr(modality_data[0], "img")
                ):
                    img_filter = ImageNormalizer(
                        images=modality_data, data_info=data_info
                    )
                    processed_data = img_filter.scale()
                    data.set_modality_data(direction, processed_data)

        # Process IMG_TO_IMG case
        elif self.config.data_case == DataCase.IMG_TO_IMG:
            # Process both image modalities
            for direction in ["from", "to"]:
                modality_data = data.get_modality_data(direction)
                if (
                    modality_data is None
                    or not isinstance(modality_data, list)
                    or not modality_data
                ):
                    continue

                modality_key = data.get_modality_key(direction)
                data_info = self.config.data_config.data_info[modality_key]

                if hasattr(modality_data[0], "img"):
                    img_filter = ImageNormalizer(
                        images=modality_data, data_info=data_info
                    )
                    processed_data = img_filter.scale()
                    data.set_modality_data(direction, processed_data)

        # Process SINGLE_CELL_TO_IMG case
        elif self.config.data_case == DataCase.SINGLE_CELL_TO_IMG:
            # Process each modality with appropriate processor
            for direction in ["from", "to"]:
                modality_data = data.get_modality_data(direction)
                if modality_data is None:
                    continue

                modality_key = data.get_modality_key(direction)
                data_info = self.config.data_config.data_info[modality_key]

                if isinstance(modality_data, MuData):
                    sc_filter = SingleCellFilter(
                        mudata=modality_data, data_info=data_info
                    )
                    processed_data = sc_filter.preprocess()
                    data.set_modality_data(direction, processed_data)
                elif (
                    isinstance(modality_data, list)
                    and modality_data
                    and hasattr(modality_data[0], "img")
                ):
                    img_filter = ImageNormalizer(
                        images=modality_data, data_info=data_info
                    )
                    processed_data = img_filter.scale()
                    data.set_modality_data(direction, processed_data)

        return data

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
