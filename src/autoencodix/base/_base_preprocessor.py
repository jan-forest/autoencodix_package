import abc
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore

from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._filter import DataFilter
from autoencodix.data._imgdataclass import ImgData
from autoencodix.data._nanremover import NaNRemover
from autoencodix.data._sc_filter import SingleCellFilter
from autoencodix.data.datapackage import DataPackage
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader, ImageNormalizer
from autoencodix.utils._screader import SingleCellDataReader
from autoencodix.utils.default_config import DataCase, DefaultConfig
from autoencodix.utils._result import Result

if TYPE_CHECKING:
    import mudata as md  # type: ignore

    MuData = md.MuData.MuData
else:
    MuData = Any


class BasePreprocessor(abc.ABC):
    """Contains logic for data preprocessing in the Autoencodix framework.

    This class defines the general preprocessing workflow and provides
    methods for handling different data modalities and data cases.
    Subclasses should implement the `preprocess` method to perform
    specific preprocessing steps.

    Attributes:
        config: A DefaultConfig object containing preprocessing configurations.
        processed_data: A dictionary to store processed DataPackage objects for each data split.
        bulk_genes_to_keep: Optional list of genes to keep for bulk data.
        bulk_scalers: Optional dictionary of scalers for bulk data.
        sc_genes_to_keep: Optional dictionary mapping modality keys to lists of genes to keep for single-cell data.
        sc_scalers: Optional dictionary mapping modality keys to scalers for single-cell data.
        sc_general_genes_to_keep: Optional dictionary mapping modality keys to lists of genes to keep filtered by non-SC specific methods.
        data_readers: A dictionary mapping DataCase enum values to data reader instances for different modalities.
        _dataset_container: Optional DatasetContainer to hold the processed datasets.
    """

    def __init__(
        self,
        config: DefaultConfig,
        ontologies: Optional[Union[Tuple[Any, Any], Dict[Any, Any]]] = None,
    ):
        """Initializes the BasePreprocessor with a configuration object.

        Args :
            config: A DefaultConfig object containing preprocessing configurations.
        """
        self.config = config
        self._dataset_container: Optional[DatasetContainer] = None
        self.processed_data = Dict[str, Dict[str, Union[Any, DataPackage]]]
        self.bulk_genes_to_keep: Optional[Dict[str, List[str]]] = None
        self.bulk_scalers: Optional[Dict[str, Any]] = None
        self.sc_genes_to_keep: Optional[Dict[str, List[str]]] = None
        self.sc_scalers: Optional[Dict[str, Dict[str, Any]]] = None
        self.sc_general_genes_to_keep: Optional[Dict[str, List]] = None
        self._ontologies: Optional[Union[Tuple[Any, Any], Dict[Any, Any]]] = ontologies
        self.data_readers: Dict[Enum, Any] = {
            DataCase.MULTI_SINGLE_CELL: SingleCellDataReader(),
            DataCase.MULTI_BULK: BulkDataReader(config=self.config),
            DataCase.BULK_TO_BULK: BulkDataReader(config=self.config),
            DataCase.SINGLE_CELL_TO_SINGLE_CELL: SingleCellDataReader(),
            DataCase.IMG_TO_BULK: {
                "bulk": BulkDataReader(config=self.config),
                "img": ImageDataReader(),
            },
            DataCase.SINGLE_CELL_TO_IMG: {
                "sc": SingleCellDataReader(),
                "img": ImageDataReader(),
            },
            DataCase.IMG_TO_IMG: ImageDataReader(),
        }

    @abc.abstractmethod
    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        """To be implemented by subclasses for specific preprocessing steps.
        Args:
            raw_user_data: Users can provide raw data. This is an alternative way of
                providing data via filepaths in the config. If this param is passed, we skip the data reading step.
            predict_new_data: Indicates whether the user wants to predict with unseen data.
                If this is the case, we don't split the data and only prerpocess.
        """
        pass

    def _general_preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Orchestrates the preprocessing steps.

        This method determines the data case from the configuration and calls
        the appropriate processing function for that data case.

        Args:
            raw_user_data: Optional DataPackage containing user-provided data.
                If provided, the data reading step is skipped.
            predict_new_data: Boolean indicating whether to preprocess new unseen data
                without splitting it into train/validation/test sets.

        Returns:
            A dictionary containing processed DataPackage objects for each data split
            (e.g., 'train', 'validation', 'test').

        Raises:
            ValueError: If an unsupported data case is encountered.
        """
        self.predict_new_data = predict_new_data
        datacase = self.config.data_case
        if datacase is None:
            raise TypeError(
                "datacase can't be None. Please ensure the configuration specifies a valid DataCase."
            )
        if raw_user_data is None:
            self.from_key, self.to_key = self._get_translation_keys()
        else:
            self.from_key, self.to_key = self._get_user_translation_keys(
                raw_user_data=raw_user_data
            )
        process_function = self._get_process_function(datacase=datacase)
        if process_function:
            return process_function(raw_user_data=raw_user_data)
        else:
            raise ValueError(f"Unsupported data case: {datacase}")

    def _get_process_function(self, datacase: DataCase) -> Any:
        """Returns the appropriate processing function based on the data case.

        Args:
            datacase: The DataCase enum value representing the current data case.

        Returns:
            A callable function that performs the preprocessing for the given data case,
            or None if the data case is not supported.
        """
        process_map = {
            DataCase.MULTI_SINGLE_CELL: self._process_multi_single_cell,
            DataCase.MULTI_BULK: self._process_multi_bulk_case,
            DataCase.BULK_TO_BULK: self._process_bulk_to_bulk_case,
            DataCase.SINGLE_CELL_TO_SINGLE_CELL: self._process_sc_to_sc_case,
            DataCase.IMG_TO_BULK: self._process_img_to_bulk_case,
            DataCase.SINGLE_CELL_TO_IMG: self._process_sc_to_img_case,
            DataCase.IMG_TO_IMG: self._process_img_to_img_case,
        }
        return process_map.get(datacase)

    def _process_data_case(
        self, data_package: DataPackage, modality_processors: Dict[Any, Any]
    ) -> Union[Dict[str, Dict[str, Union[Any, DataPackage]]], Dict[str, Any]]:
        """Processes the data package based on the provided modality processors.

        This method handles the common preprocessing steps for different data cases,
        including splitting the data package, removing NaNs, and applying
        odality-specific processors.

        Args::
            data_package: The DataPackage object to be processed.
            modality_processors: A dictionary mapping modality keys (e.g., 'multi_sc', 'from_modality')
                to callable processor functions that will be applied to the corresponding modality data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        """
        if self.predict_new_data:
            # use train, because processing logic expects train split
            mock_split: Dict[str, Dict[str, Union[Any, DataPackage]]] = {
                "test": {
                    "data": data_package,
                    "indices": {"paired": np.array([])},
                },
                "valid": {"data": None, "indices": {"paired": np.array([])}},
                "train": {"data": None, "indices": {"paired": np.array([])}},
            }
            if self.config.skip_preprocessing:
                return mock_split

            clean_package = self._remove_nans(data_package=data_package)
            mock_split["test"]["data"] = clean_package
            for modality_key, (
                presplit_processor,
                postsplit_processor,
            ) in modality_processors.items():
                modality_data = clean_package[modality_key]
                if modality_data:
                    processed_modality_data = presplit_processor(modality_data)
                    # mock the split
                    clean_package[modality_key] = processed_modality_data
                    mock_split["test"]["data"] = clean_package
                    mock_split = postsplit_processor(mock_split)
            return mock_split
        # normal case without new data -----------------------------------
        if self.config.skip_preprocessing:
            split_packages, _ = self._split_data_package(data_package=data_package)
            return split_packages
        clean_package = self._remove_nans(data_package=data_package)
        for modality_key, (presplit_processor, _) in modality_processors.items():
            modality_data = clean_package[modality_key]
            if modality_data:
                processed_modality_data = presplit_processor(modality_data)
                clean_package[modality_key] = processed_modality_data
        split_packages, indices = self._split_data_package(data_package=clean_package)
        processed_splits = {}
        for modality_key, (_, postsplit_processor) in modality_processors.items():
            split_packages = postsplit_processor(split_packages)
        for split_name, split_package in split_packages.items():
            split_indices = {
                name: {
                    split: idx
                    for split, idx in indices[name].items()
                    if split == split_name
                }
                for name in indices.keys()
            }
            processed_splits[split_name] = {
                "data": split_package["data"],
                "indices": split_indices,
            }
        return processed_splits

    def _process_multi_single_cell(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Process MULTI_SINGLE_CELL case

        Reads multi-single-cell data, performs data splitting, NaN removal,
        and applies single-cell specific filtering.
        Args:
            raw_user_data: Optional DataPackage containing user-provided data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        """
        if raw_user_data is None:
            screader = self.data_readers[DataCase.MULTI_SINGLE_CELL]  # type: ignore

            mudata = screader.read_data(config=self.config)
            data_package = DataPackage()
            data_package.multi_sc = mudata
        else:
            data_package = raw_user_data
        if self.config.requires_paired:
            print(
                f"datapackge in process_multi_single_cell {data_package} and multi_sc: {data_package.multi_sc}"
            )
            common_ids = data_package.get_common_ids()
            data_package.multi_sc = {
                "multi_sc": data_package.multi_sc["multi_sc"][common_ids]
            }

        def presplit_processor(modality_data: Any) -> Any:
            if modality_data is None:
                return modality_data
            sc_filter = SingleCellFilter(
                data_info=self.config.data_config.data_info, config=self.config
            )
            return sc_filter.presplit_processing(multi_sc=modality_data)

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
            return self._postsplit_multi_single_cell(
                split_data=split_data, datapackage_key="multi_sc"
            )

        return self._process_data_case(
            data_package,
            modality_processors={"multi_sc": (presplit_processor, postsplit_processor)},
        )

    def _process_multi_bulk_case(
        self,
        raw_user_data: Optional[DataPackage] = None,
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process MULTI_BULK case.

        Reads multi-bulk data, performs data splitting, NaN removal,
        and applies filtering and scaling to bulk dataframes.
        Args:
            raw_user_data: Optional DataPackage containing user-provided data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        """
        if raw_user_data is None:
            bulkreader = self.data_readers[DataCase.MULTI_BULK]
            bulk_dfs, annotation = bulkreader.read_data()

            data_package = DataPackage()
            data_package.multi_bulk = bulk_dfs
            data_package.annotation = annotation
        else:
            data_package = raw_user_data
        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()
            unpaired_data = data_package.multi_bulk
            unpaired_anno = data_package.annotation
            data_package.multi_bulk = {
                k: v.loc[common_ids] for k, v in unpaired_data.items()
            }
            data_package.annotation = {
                k: v.loc[common_ids] for k, v in unpaired_anno.items()
            }

        def presplit_processor(
            modality_data: Dict[str, Union[pd.DataFrame, None]],
        ) -> Dict[str, Union[pd.DataFrame, None]]:
            """For the multi_bulk modality we perform all operations after splitting at the moment."""
            return modality_data

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
            return self._postsplit_multi_bulk(split_data=split_data)

        return self._process_data_case(
            data_package,
            modality_processors={
                "multi_bulk": (presplit_processor, postsplit_processor)
            },
        )

    def _calc_k_filter(
        self, i: int, remainder: int, base_features: int
    ) -> Optional[int]:
        if self.config.k_filter is None:
            return None
        extra = 1 if i < remainder else 0
        return base_features + extra

    def _postsplit_multi_single_cell(
        self,
        split_data: Dict[str, Dict[str, Any]],
        datapackage_key: str = "multi_sc",
        modality_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Post-split processing for multi-single-cell data.
        This method applies filtering and scaling to the single-cell data after it has been split.
        Now supports multiple MuData objects in the input dictionary.

        Args:
            split_data: A dictionary containing the split data for each data split.
            datapackage_key: The key in the DataPackage that contains the multi-single-cell data.
            modality_key: Optional specific modality key for backward compatibility.
                        If provided, only processes that specific modality.
                        If None, processes all modalities in the dictionary.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.

        Raises:
            ValueError: If the train split data is None.
        """
        processed_splits: Dict[str, Dict[str, Any]] = {}
        train_split: Optional[Dict[str, Any]] = split_data.get("train")

        if train_split is None:
            raise ValueError(
                "Train split data is None. Ensure that the data package contains valid train data."
            )

        train_data: Optional[Any] = train_split.get("data")
        if train_data is None:
            raise ValueError(
                "Train split data is None. Ensure that the data package contains valid train data."
            )

        # Get all modality keys from the train data
        mudata_dict = train_data[datapackage_key]

        if modality_key is not None:
            if modality_key not in mudata_dict:
                raise ValueError(
                    f"Specified modality_key '{modality_key}' not found in {list(mudata_dict.keys())}"
                )
            modality_keys = [modality_key]
            print(
                f"Processing single modality (backward compatibility): {modality_key}"
            )
        else:
            modality_keys = list(mudata_dict.keys())
            print(f"Processing {len(modality_keys)} MuData objects: {modality_keys}")

        # Initialize storage for scalers and gene filters for each modality
        if (
            self.sc_scalers is None
            and self.sc_genes_to_keep is None
            and self.sc_general_genes_to_keep is None
        ) or ("modality" in datapackage_key):
            # Process each MuData object in the train split
            processed_mudata_dict = {}
            all_scalers = {}
            all_sc_genes_to_keep = {}
            all_general_genes_to_keep = {}

            for current_modality_key in modality_keys:
                print(f"Processing train modality: {current_modality_key}")

                sc_filter = SingleCellFilter(
                    data_info=self.config.data_config.data_info, config=self.config
                )

                # Single-cell specific filtering
                filtered_train, sc_genes_to_keep = sc_filter.sc_postsplit_processing(
                    mudata=mudata_dict[current_modality_key]
                )

                # General post-processing
                processed_train, general_genes_to_keep, scalers = (
                    sc_filter.general_postsplit_processing(
                        mudata=filtered_train, scaler_map=None, gene_map=None
                    )
                )

                # Store processed data and filters for this modality
                processed_mudata_dict[current_modality_key] = processed_train
                all_scalers[current_modality_key] = scalers
                all_sc_genes_to_keep[current_modality_key] = sc_genes_to_keep
                all_general_genes_to_keep[current_modality_key] = general_genes_to_keep

            # Store all scalers and gene filters
            self.sc_scalers = all_scalers
            self.sc_genes_to_keep = all_sc_genes_to_keep
            self.sc_general_genes_to_keep = all_general_genes_to_keep

            # Update train data with processed MuData objects
            train_data[datapackage_key] = processed_mudata_dict

        else:
            # Use existing scalers and gene filters
            all_scalers = self.sc_scalers  # type: ignore
            all_sc_genes_to_keep = self.sc_genes_to_keep  # type: ignore
            all_general_genes_to_keep = self.sc_general_genes_to_keep  # type: ignore

        # Store processed train split
        processed_splits["train"] = {
            "data": train_data,
            "indices": split_data["train"]["indices"],
        }

        # Process other splits (val, test, etc.)
        for split, split_package in split_data.items():
            if split == "train":
                continue

            data_package = split_package["data"]
            if data_package is None:
                processed_splits[split] = split_package
                continue

            print(f"Processing {split} split")
            processed_mudata_dict = {}

            # Process each MuData object in this split
            for current_modality_key in modality_keys:
                print(f"Processing {split} modality: {current_modality_key}")

                sc_filter = SingleCellFilter(
                    data_info=self.config.data_config.data_info, config=self.config
                )

                # Apply single-cell filtering using train-derived gene map
                filtered_sc_data, _ = sc_filter.sc_postsplit_processing(
                    mudata=data_package[datapackage_key][current_modality_key],
                    gene_map=all_sc_genes_to_keep[current_modality_key],
                )

                # Apply general processing using train-derived scalers and gene map
                processed_general_data, _, _ = sc_filter.general_postsplit_processing(
                    mudata=filtered_sc_data,
                    gene_map=all_general_genes_to_keep[current_modality_key],
                    scaler_map=all_scalers[current_modality_key],
                )

                processed_mudata_dict[current_modality_key] = processed_general_data

            # Update data package with all processed MuData objects
            data_package[datapackage_key] = processed_mudata_dict

            processed_splits[split] = {
                "data": data_package,
                "indices": split_package["indices"],
            }

        return processed_splits

    def _postsplit_multi_bulk(
        self,
        split_data: Dict[str, Dict[str, Any]],
        datapackage_key: str = "multi_bulk",
    ) -> Dict[str, Dict[str, Any]]:
        """Post-split processing for multi-bulk data.

        This method applies filtering and scaling to the bulk dataframes after they have been split.

        Args:
            split_data: A dictionary containing the split data for each data split.
            datapackage_key: The key in the DataPackage that contains the multi-bulk data.
        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            ValueError: If the train split data is None.
        """

        train_split: Optional[Dict[str, Any]] = split_data.get("train")
        if train_split is None:
            raise ValueError(
                "Train split data is None. Ensure that the data package contains valid train data."
            )
        train_data: Optional[Any] = train_split.get("data")
        genes_to_keep_map: Dict[str, List[str]] = {}
        scalers: Dict[str, Any] = {}
        processed_splits: Dict[str, Dict[str, Any]] = {}

        if (self.bulk_scalers is None and self.bulk_genes_to_keep is None) or (
            "modality" in datapackage_key
        ):
            if train_data is None:
                raise ValueError(
                    "Train split data is None. Ensure that the data package contains valid train data."
                )
            n_modalities: int = len(train_data[datapackage_key].keys())
            remainder: int = 0
            base_features = 0
            if self.config.k_filter is not None:
                base_features = self.config.k_filter // n_modalities
                remainder = self.config.k_filter % n_modalities

            # Get valid modality keys (those that are not None)
            modality_keys = [
                k for k, v in train_data[datapackage_key].items() if v is not None
            ]

            for i, k in enumerate(modality_keys):
                v = train_data[datapackage_key][k]
                cur_k_filter = self._calc_k_filter(
                    i=i, base_features=base_features, remainder=remainder
                )
                self.config.data_config.data_info[k].k_filter = cur_k_filter

                data_processor = DataFilter(
                    data_info=self.config.data_config.data_info[k],
                    ontologies=self._ontologies,
                )
                filtered_df, genes_to_keep = data_processor.filter(df=v)
                scaler = data_processor.fit_scaler(df=filtered_df)
                genes_to_keep_map[k] = genes_to_keep
                scalers[k] = scaler
                scaled_df = data_processor.scale(df=filtered_df, scaler=scaler)
                train_data[datapackage_key][k] = scaled_df
                # Check if indices stayed the same after filtering
                if not filtered_df.index.equals(v.index):
                    mismatched_indices = filtered_df.index.symmetric_difference(v.index)
                    raise ValueError(
                        f"Indices mismatch after filtering for modality {k}. "
                        f"Mismatched indices: {mismatched_indices}. "
                        "Ensure filtering does not alter the indices."
                    )

            self.bulk_scalers = scalers
            self.bulk_genes_to_keep = genes_to_keep_map  # type: ignore
        else:
            scalers, genes_to_keep_map = self.bulk_scalers, self.bulk_genes_to_keep  # type: ignore

        processed_splits["train"] = {
            "data": train_data,
            "indices": split_data["train"]["indices"],
        }

        for split_name, split_package in split_data.items():
            if split_name == "train":
                continue
            if split_package["data"] is None:
                processed_splits[split_name] = split_data[split_name]
                continue

            processed_package = split_package["data"]
            for k, v in processed_package[datapackage_key].items():
                if v is None:
                    continue
                data_processor = DataFilter(
                    data_info=self.config.data_config.data_info[k],
                    ontologies=self._ontologies,
                )
                filtered_df, _ = data_processor.filter(
                    df=v, genes_to_keep=genes_to_keep_map[k]
                )
                scaled_df = data_processor.scale(df=filtered_df, scaler=scalers[k])
                processed_package[datapackage_key][k] = scaled_df
                if not filtered_df.index.equals(v.index):
                    raise ValueError(
                        f"Indices mismatch after filtering for modality {k}. "
                        "Ensure filtering does not alter the indices."
                    )

            processed_splits[split_name] = {
                "data": processed_package,
                "indices": split_package["indices"],
            }

        return processed_splits

    def _process_bulk_to_bulk_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Union[dict[str, dict[str, Union[Any, DataPackage]]]]:
        """Process BULK_TO_BULK case.

        Reads bulk data, prepares from/to modalities, performs data splitting,
        NaN removal, and applies filtering and scaling to from and to bulk dataframes.

        Args:
            raw_user_data: Optional DataPackage containing user-provided data.
                If provided, the data reading step is skipped.
        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            TypeError: If from_key or to_key is None, indicating that translation keys must be specified.

        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("from_key and to_key cannot be None for Translation")
        if raw_user_data is None:
            bulkreader = self.data_readers[DataCase.BULK_TO_BULK]
            bulk_dfs, annotation = bulkreader.read_data()

            data_package = DataPackage()
            data_package.from_modality = {self.from_key: bulk_dfs[self.from_key]}
            data_package.to_modality = {self.to_key: bulk_dfs[self.to_key]}
            data_package.multi_bulk = None
            data_package.annotation = {
                "from": annotation[self.from_key],
                "to": annotation[self.to_key],
            }
        else:
            data_package = raw_user_data

        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()

            data_package.from_modality = {
                k: v.loc[common_ids]  # type: ignore
                for k, v in data_package.from_modality.items()  # type: ignore
            }  # type: ignore
            data_package.to_modality = {
                k: v.loc[common_ids]  # type: ignore
                for k, v in data_package.to_modality.items()  # type: ignore
            }  # type: ignore

            data_package.annotation = {
                k: v.loc[common_ids] for k, v in data_package.annotation.items()
            }  # type: ignore

        def presplit_processor(
            modality_data: Dict[str, pd.DataFrame],
        ) -> Dict[str, pd.DataFrame]:
            return modality_data  # no presplit processing needed for multi bulk

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]], datapackage_key: str
        ) -> Dict[str, Dict[str, Any]]:
            return self._postsplit_multi_bulk(
                split_data=split_data, datapackage_key=datapackage_key
            )

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": (
                    presplit_processor,
                    lambda data: postsplit_processor(data, "from_modality"),
                ),
                "to_modality": (
                    presplit_processor,
                    lambda data: postsplit_processor(data, "to_modality"),
                ),
            },
        )

    def _process_sc_to_sc_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Process SINGLE_CELL_TO_SINGLE_CELL case.

        Reads single-cell data, prepares from/to modalities, performs data splitting,
        NaN removal, and applies single-cell specific filtering to from and to mudata objects.

        Args:
            raw_user_data: Optional DataPackage containing user-provided data.
                If provided, the data reading step is skipped.
        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            TypeError: If from_key or to_key is None, indicating that translation keys must be specified.
        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("from_key and to_key cannot be None for Translation")

        if raw_user_data is None:
            screader = self.data_readers[DataCase.SINGLE_CELL_TO_SINGLE_CELL]
            mudata = screader.read_data(config=self.config)
            mod_dict = mudata["multi_sc"]

            data_package = DataPackage()
            data_package.from_modality = {self.from_key: mod_dict[self.from_key]}
            data_package.to_modality = {self.to_key: mod_dict[self.to_key]}
        else:
            data_package = raw_user_data
        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()

            # Subset both from_modality and to_modality
            data_package.from_modality = {
                k: v[common_ids]
                for k, v in data_package.from_modality.items()  # type: ignore
            }
            data_package.to_modality = {
                k: v[common_ids]
                for k, v in data_package.to_modality.items()  # type: ignore
            }

        def presplit_processor(
            modality_data: Dict[str, Union[AnnData, Any]],
            modality_key: str,
        ) -> Dict[str, Any]:
            from mudata import MuData

            if modality_data is None:
                return modality_data
            sc_filter = SingleCellFilter(
                data_info=self.config.data_config.data_info, config=self.config
            )
            modality_data = modality_data[modality_key]
            # Check if it is AnnData and convert to MuData if needed
            if isinstance(modality_data, AnnData):
                mudata = MuData({modality_key: modality_data})
                return {modality_key: sc_filter.presplit_processing(multi_sc=mudata)}

            elif isinstance(modality_data, MuData):
                return {
                    modality_key: sc_filter.presplit_processing(multi_sc=modality_data)
                }
            else:
                raise TypeError(
                    f"Expected AnnData or MuData, got {type(modality_data)}"
                )

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]],
            datapackage_key: str,
            modality_key: str,
        ) -> Dict[str, Dict[str, Any]]:
            return self._postsplit_multi_single_cell(
                split_data=split_data,
                datapackage_key=datapackage_key,
                modality_key=modality_key,
            )

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": (
                    lambda data: presplit_processor(data, self.from_key),
                    lambda data: postsplit_processor(
                        data, "from_modality", self.from_key
                    ),
                ),
                "to_modality": (
                    lambda data: presplit_processor(data, self.to_key),
                    lambda data: postsplit_processor(data, "to_modality", self.to_key),
                ),
            },
        )

    def _process_img_to_bulk_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Process IMG_TO_BULK case

        Reads image and bulk data, prepares from/to modalities (IMG->BULK or BULK->IMG),
        performs data splitting, NaN removal, and applies normalization to image data
        and filtering/scaling to bulk dataframes.
        Args:
            raw_user_data: Optional DataPackage containing user-provided data.
                If provided, the data reading step is skipped.
        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            TypeError: If from_key or to_key is None, indicating that translation keys must be specified.
        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("from_key and to_key cannot be None for Translation")

        if raw_user_data is None:
            bulkreader = self.data_readers[DataCase.IMG_TO_BULK]["bulk"]
            imgreader = self.data_readers[DataCase.IMG_TO_BULK]["img"]

            bulk_dfs, annotation = bulkreader.read_data()
            images = imgreader.read_data(config=self.config)

            data_package = DataPackage()

            if self.to_key in bulk_dfs.keys():
                # IMG -> BULK direction
                data_package.from_modality = {self.from_key: images[self.from_key]}
                data_package.to_modality = {self.to_key: bulk_dfs[self.to_key]}
                to_annotation = next(iter(annotation.keys()))
                data_package.annotation = {
                    "from": None,
                    "to": annotation[to_annotation],
                }
            else:
                # BULK -> IMG direction
                data_package.from_modality = {self.from_key: bulk_dfs[self.from_key]}
                data_package.to_modality = {self.to_key: images[self.to_key]}
                from_annotation = next(iter(annotation.keys()))
                data_package.annotation = {
                    "from": annotation[from_annotation],
                    "to": None,
                }
        else:
            data_package = raw_user_data

        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()
            if data_package.from_modality is not None:
                for k, v in data_package.from_modality.items():
                    if self._is_image_data(v):
                        filtered_list = [
                            img for img in v if img.sample_id in common_ids
                        ]
                        data_package.from_modality[k] = filtered_list
                    else:
                        data_package.from_modality[k] = v.loc[common_ids]

            if data_package.to_modality is not None:
                for k, v in data_package.to_modality.items():
                    if self._is_image_data(v):
                        filtered_list = [
                            img for img in v if img.sample_id in common_ids
                        ]
                        data_package.to_modality[k] = filtered_list
                    else:
                        data_package.to_modality[k] = v.loc[common_ids]

            if data_package.annotation is not None:
                for side in ["from", "to"]:
                    ann = data_package.annotation.get(side)
                    if ann is not None:
                        data_package.annotation[side] = ann.loc[common_ids]

        def presplit_processor(
            modality_data: Dict[str, Union[pd.DataFrame, List[ImgData]]],
            modality_key: str,
        ) -> Dict[str, Union[pd.DataFrame, List[ImgData]]]:
            if modality_key not in modality_data:
                raise ValueError(f"Modality key '{modality_key}' not found in data.")
            data = modality_data[modality_key]
            if self._is_image_data(data=data):
                modality_data[modality_key] = self._normalize_image_data(
                    images=modality_data[modality_key],
                    info_key=modality_key,  # type: ignore
                )
            # we don't need to filter bulk data here
            # because we do it in the postsplit step
            return modality_data

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]], datapackage_key: str
        ) -> Dict[str, Dict[str, Any]]:
            non_empty_split = None
            for _, split_content in split_data.items():
                if split_content["data"][
                    datapackage_key
                ]:  # Check if the split has data
                    non_empty_split = split_content["data"][datapackage_key]
                    break

            if non_empty_split is None:
                return split_data

            for _, modality_data in non_empty_split.items():
                if isinstance(modality_data, pd.DataFrame):  # Bulk data
                    return self._postsplit_multi_bulk(
                        split_data=split_data, datapackage_key=datapackage_key
                    )
                elif isinstance(modality_data, list):  # Image data
                    continue
            return split_data  # for img data we don't need to do anything

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": (
                    lambda data: presplit_processor(data, self.from_key),
                    lambda data: postsplit_processor(data, "from_modality"),
                ),
                "to_modality": (
                    lambda data: presplit_processor(data, self.to_key),
                    lambda data: postsplit_processor(data, "to_modality"),
                ),
            },
        )

    def _process_sc_to_img_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Process SC_TO_IMG case.

        Reads single-cell and image data, prepares from/to modalities (SC->IMG or IMG->SC),
        performs data splitting, NaN removal, and applies single-cell specific filtering
        to single-cell data and normalization to image data.

        Args:
            raw_user_data: Optional DataPackage containing user-provided data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            TypeError: If from_key or to_key is None, indicating that translation keys must be specified.
        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("to and from keys cant be None for SC-to-Img")
        if raw_user_data is None:
            screader = self.data_readers[DataCase.SINGLE_CELL_TO_IMG]["sc"]
            imgreader = self.data_readers[DataCase.SINGLE_CELL_TO_IMG]["img"]

            # only one mudata type in this case we know this
            mudata = next(iter(screader.read_data(config=self.config).values()))
            images = imgreader.read_data(config=self.config)

            data_package = DataPackage()
        else:
            data_package = raw_user_data

        if self.from_key in images.keys():
            data_package.from_modality = {self.from_key: images[self.from_key]}
            data_package.to_modality = {self.to_key: mudata}
        else:
            data_package.from_modality = {self.from_key: mudata}
            data_package.to_modality = {self.to_key: images[self.to_key]}

        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()
            if data_package.from_modality is not None:
                for k, v in data_package.from_modality.items():
                    if self._is_image_data(v):
                        filtered_list = [
                            img for img in v if img.sample_id in common_ids
                        ]
                        data_package.from_modality[k] = filtered_list
                    else:
                        data_package.from_modality[k] = v.loc[common_ids]

            if data_package.to_modality is not None:
                for k, v in data_package.to_modality.items():
                    if self._is_image_data(v):
                        filtered_list = [
                            img for img in v if img.sample_id in common_ids
                        ]
                        data_package.to_modality[k] = filtered_list
                    else:
                        data_package.to_modality[k] = v.loc[common_ids]

        def presplit_processor(
            modality_data: Dict[str, Union[Any, List[ImgData]]], modality_key: str
        ) -> Dict[str, Union[Any, List[ImgData]]]:
            if modality_key not in modality_data:
                raise ValueError(f"Modality key '{modality_key}' not found in data.")

            if isinstance(modality_data[modality_key], list):  # Image data
                modality_data[modality_key] = self._normalize_image_data(
                    images=modality_data[modality_key], info_key=modality_key
                )
            return modality_data

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]], datapackage_key: str
        ) -> Dict[str, Dict[str, Any]]:
            non_empty_split = None
            for _, split_content in split_data.items():
                if split_content["data"][datapackage_key]:
                    non_empty_split = split_content["data"][datapackage_key]
                    break

            if non_empty_split is None:
                return split_data

            for _, modality_data in non_empty_split.items():
                if isinstance(modality_data, MuData):  # Single-cell data
                    return self._postsplit_multi_single_cell(
                        split_data=split_data, datapackage_key=datapackage_key
                    )
                elif isinstance(modality_data, list):  # Image data
                    # No postsplit processing needed for image data
                    continue
            return split_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": (
                    lambda data: presplit_processor(data, self.from_key),
                    lambda data: postsplit_processor(data, "from_modality"),
                ),
                "to_modality": (
                    lambda data: presplit_processor(data, self.to_key),
                    lambda data: postsplit_processor(data, "to_modality"),
                ),
            },
        )

    def _process_img_to_img_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, DataPackage]:
        """Process IMG_TO_IMG case.

        Reads image data for from/to modalities, performs data splitting,
        NaN removal, and applies normalization to both from and to image data.

        Args:
            raw_user_data: Optional DataPackage containing user-provided data.
                If provided, the data reading step is skipped.
        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        Raises:
            TypeError: If from_key or to_key is None, indicating that translation keys must be specified.
        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("from_key and to_key cannot be None for Translation")
        if raw_user_data is None:
            imgreader = self.data_readers[DataCase.IMG_TO_IMG]
            images = imgreader.read_data(config=self.config)

            data_package = DataPackage()
            data_package.from_modality = {self.from_key: images[self.from_key]}
            data_package.to_modality = {self.to_key: images[self.to_key]}
        else:
            data_package = raw_user_data

        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()

            def filter_imgdata_list(img_list, ids):
                filtered = []
                for imgdata in img_list:
                    if imgdata.sample_id in ids:
                        filtered.append(imgdata)
                return filtered

            data_package.from_modality[self.from_key] = filter_imgdata_list(
                data_package.from_modality[self.from_key], common_ids
            )

            data_package.to_modality[self.to_key] = filter_imgdata_list(
                data_package.to_modality[self.to_key], common_ids
            )

        def presplit_processor(
            modality_data: Dict[str, List], modality_key: str
        ) -> Dict[str, List]:
            """Processes img-to-img modality data with normalization for images."""
            if modality_key is None:
                raise TypeError("modality_key cannot be None for translation")
            if modality_data and modality_key in modality_data:
                data = modality_data[modality_key]
                if self._is_image_data(data):
                    return {
                        modality_key: self._normalize_image_data(data, modality_key)
                    }
            return modality_data

        def postsplit_processor(
            split_data: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
            """No postsplit processing needed for image data."""
            return split_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": (
                    lambda data: presplit_processor(
                        data,
                        self.from_key,  # type: ignore
                    ),
                    postsplit_processor,
                ),
                "to_modality": (
                    lambda data: presplit_processor(
                        data,
                        self.to_key,  # type: ignore
                    ),
                    postsplit_processor,
                ),
            },
        )

    def _split_data_package(
        self, data_package: DataPackage
    ) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, Any]]:
        """Splits data package into train/validation/test sets.

        Uses DataSplitter and DataPackageSplitter to divide the DataPackage
        into training, validation, and test sets based on the configuration.

        Args:
            data_package: The DataPackage to be split.

        Returns:
            A dictionary containing the split DataPackages, keyed by split names
            ('train', 'validation', 'test').

            split_indiced_config - (dict): the actual indicies used for splitting
        """
        data_splitter = DataSplitter(config=self.config)
        n_samples = data_package.get_n_samples()
        split_indices_config = n_samples.copy()

        print(f" n_samples: {n_samples}")
        for k, v in n_samples.items():
            for subkey, n in v.items():
                if n == 0 or n is None:
                    split_indices_config[k][subkey] = None
                    continue
                split_indices_config[k][subkey] = data_splitter.split(n_samples=n)
        data_splitter_instance = DataPackageSplitter(
            data_package=data_package,
            config=self.config,
            indices=split_indices_config,
        )
        return data_splitter_instance.split(), split_indices_config

    def _is_image_data(self, data: Any) -> bool:
        """Check if data is image data.

        Determines if the provided data is a list of objects that are considered
        image data based on having an 'img' attribute.

        Args:
            data: The data to check.

        Returns:
            True if the data is image data, False otherwise.
        """
        if data is None:
            return False
        if isinstance(data, list) and hasattr(data[0], "img"):
            return True
        return False

    def _remove_nans(self, data_package: DataPackage) -> DataPackage:
        """Remove NaN values from the data package.

        Utilizes NaNRemover to identify and remove rows containing NaN values
        in relevant annotation columns within the DataPackage.

        Args:
            data_package: The DataPackage from which to remove NaNs.

        Returns:
            The DataPackage with NaN values removed.
        """
        nanremover = NaNRemover(
            config=self.config,
        )
        return nanremover.remove_nan(data=data_package)

    def _normalize_image_data(self, images: List, info_key: str) -> List:
        """Process images with normalization.

        Normalizes a list of image data objects using ImageNormalizer based on
        the scaling method specified in the configuration for the given info_key.

        Args:
            images: A list of image data objects (each having an 'img' attribute).
            info_key: The key referencing data information in the configuration to get the scaling method.

        Returns:
            A list of processed image data objects with normalized image data.
        """
        scaling_method = self.config.data_config.data_info[info_key].scaling
        processed_images = []
        normalizer = ImageNormalizer()  # Instance created once here

        for img in images:
            img.img = normalizer.normalize_image(  # Modify directly
                image=img.img, method=scaling_method
            )
            processed_images.append(img)

        return processed_images

    def _get_translation_keys(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract from and to keys from config.

        Retrieves the 'from' and 'to' modality keys from the data configuration
        based on the 'translate_direction' setting.

        Returns:
            A tuple containing the from_key and to_key as strings, or None if not found.

        Raises:
            ValueError: If neither 'from' nor 'to' keys are found in the data configuration.
            TypeError: If the translate_direction is not set for the data_info.
        """
        from_key, to_key = None, None
        for k, v in self.config.data_config.data_info.items():
            if v.translate_direction is None:
                continue
            if v.translate_direction == "from":
                from_key = k
            if v.translate_direction == "to":
                to_key = k
        return from_key, to_key

    def _get_user_translation_keys(self, raw_user_data: DataPackage):
        if len(raw_user_data.from_modality) == 0:  # type: ignore
            return None, None
        elif len(raw_user_data.to_modality) == 0:  # type: ignore
            return None, None
        else:
            if raw_user_data.from_modality is None or raw_user_data.to_modality is None:
                raise TypeError(
                    "from_modality and to_modality cannot be None for Translation"
                )
            try:
                return next(iter(raw_user_data.from_modality.keys())), next(
                    iter(raw_user_data.to_modality.keys())
                )
            except Exception as e:
                print("error getting from or to keys")
                print(e)
                print("returning None")
                return None, None

    @abstractmethod
    def format_reconstruction(
        self, reconstruction: torch.Tensor, result: Optional[Result] = None
    ) -> DataPackage:
        pass
