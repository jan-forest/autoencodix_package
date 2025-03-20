import abc
import copy
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from autoencodix.data.datapackage import DataPackage
from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._filter import DataFilter
from autoencodix.data._nanremover import NaNRemover
from autoencodix.data._sc_filter import SingleCellFilter
from autoencodix.utils._bulkreader import BulkDataReader
from autoencodix.utils._imgreader import ImageDataReader, ImageNormalizer
from autoencodix.utils._screader import SingleCellDataReader
from autoencodix.utils.default_config import DataCase, DefaultConfig

if TYPE_CHECKING:
    import mudata as md  # type: ignore

    MuData = md.MuData.MuData
else:
    MuData = Any


class BasePreprocessor(abc.ABC):
    """
    Abstract base class for data preprocessors.

    This class defines the general preprocessing workflow and provides
    methods for handling different data modalities and data cases.
    Subclasses should implement the `preprocess` method to perform
    specific preprocessing steps.
    """

    def __init__(self, config: DefaultConfig):
        """
        Initializes the BasePreprocessor with a configuration object.

        Parameters:
            config: A DefaultConfig object containing preprocessing configurations.
        """
        self.config = config
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
        """
        Abstract method to be implemented by subclasses for specific preprocessing steps.
        Parameters:
            raw_user_data - (Optional[DataPackage]): Users can provide raw data. This is an alternative way of
                provding to providing data via filepaths in the config. If this param is passed, we the data reading step.
            is_predict - (bool): Indicates whether the user wants to predict with unseen data.
                If this is the case, we don't split the data and only prerpocess.

        """
        pass

    def _general_preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Main preprocessing method that orchestrates the process flow.

        This method determines the data case from the configuration and calls
        the appropriate processing function for that data case.

        Returns:
            A dictionary containing processed DataPackage objects for each data split
            (e.g., 'train', 'validation', 'test').

        Raises:
            ValueError: If an unsupported data case is encountered.
        """
        self.predict_new_data = predict_new_data
        datacase = self.config.data_case
        if datacase is None:
            raise TypeError("datacase can't be None")
        if raw_user_data is None:
            self.from_key, self.to_key = self._get_translation_keys()
        else:
            self.from_key, self.to_key = self._get_user_translation_keys(
                raw_user_data=raw_user_data
            )
        process_function = self._get_process_function(datacase=datacase)
        if process_function:
            return process_function(
                raw_user_data=raw_user_data
            )  # No need to pass from_key, to_key anymore
        else:
            raise ValueError(f"Unsupported data case: {datacase}")

    def _get_process_function(self, datacase: DataCase) -> Any:
        """
        Returns the appropriate processing function based on the data case.

        Parameters:
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
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Generic method to process a data case.

        This method handles the common preprocessing steps for different data cases,
        including splitting the data package, removing NaNs, and applying
        modality-specific processors.

        Parameters:
            data_package: The DataPackage object to be processed.
            modality_processors: A dictionary mapping modality keys (e.g., 'multi_sc', 'from_modality')
                to callable processor functions that will be applied to the corresponding modality data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        """
        if self.predict_new_data:
            clean_package = self._remove_nans(data_package=data_package)
            for modality_key, processor in modality_processors.items():
                modality_data = getattr(clean_package, modality_key, None)
                if modality_data:
                    processed_modality_data = processor(modality_data)
                    setattr(clean_package, modality_key, processed_modality_data)
            return {
                "test": {"data": clean_package, "indices": []},
                "train": {"data": None, "indices": None},  # Updated to return None
                "valid": {"data": None, "indices": None},  # Updated to return None
            }

        split_packages, indices = self._split_data_package(data_package=data_package)
        processed_splits = {}
        for split_name, split_package in split_packages.items():
            if split_package["data"] is None:  # Handle missing splits
                processed_splits[split_name] = {
                    "data": None,
                    "indices": None,
                }
                continue

            clean_package = self._remove_nans(data_package=split_package["data"])
            for modality_key, processor in modality_processors.items():
                modality_data = getattr(clean_package, modality_key, None)
                if modality_data:
                    processed_modality_data = processor(modality_data)
                    setattr(clean_package, modality_key, processed_modality_data)

            split_indices = {
                name: {
                    split: idx
                    for split, idx in indices[name].items()
                    if split == split_name
                }
                for name in indices.keys()
            }
            processed_splits[split_name] = {
                "data": clean_package,
                "indices": split_indices,
            }
        return processed_splits

    def _process_multi_single_cell(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process MULTI_SINGLE_CELL case end-to-end.

        Reads multi-single-cell data, performs data splitting, NaN removal,
        and applies single-cell specific filtering.

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

        def process_sc_modality(modality_data: MuData) -> MuData:
            """Processes single-cell modality data with filtering."""
            if modality_data is not None:
                sc_filter = SingleCellFilter(
                    mudata=modality_data, data_info=self.config.data_config.data_info
                )
                return sc_filter.preprocess()
            return modality_data

        return self._process_data_case(
            data_package, modality_processors={"multi_sc": process_sc_modality}
        )

    def _process_multi_bulk_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process MULTI_BULK case end-to-end.

        Reads multi-bulk data, performs data splitting, NaN removal,
        and applies filtering and scaling to bulk dataframes.

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

        def process_bulk_modality(
            modality_data: Dict[str, pd.DataFrame | None],
        ) -> Dict[str, pd.DataFrame | None]:
            """Processes multi-bulk modality data with filtering and scaling."""
            if modality_data:
                processed_bulk = {}
                for key, df in modality_data.items():
                    processed_bulk[key] = self._filter_and_scale_dataframe(
                        df=df,  # type: ignore
                        info_key=key,  # type: ignore
                    )
                return processed_bulk
            return modality_data

        return self._process_data_case(
            data_package, modality_processors={"multi_bulk": process_bulk_modality}
        )

    def _process_bulk_to_bulk_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, DataPackage]:
        """
        Process BULK_TO_BULK case end-to-end.

        Reads bulk data, prepares from/to modalities, performs data splitting,
        NaN removal, and applies filtering and scaling to from and to bulk dataframes.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
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

        def process_bulk_to_bulk_modality(
            modality_data: Dict[str, pd.DataFrame], modality_key: str
        ) -> Dict[str, pd.DataFrame]:
            """Processes bulk-to-bulk modality data with filtering and scaling."""
            if not isinstance(modality_key, str):
                raise TypeError(
                    f"Modality key as to be string got {type(modality_key)}"
                )

            if modality_data and modality_key in modality_data:
                return {
                    modality_key: self._filter_and_scale_dataframe(
                        modality_data[modality_key], modality_key
                    )
                }
            return modality_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": lambda data: process_bulk_to_bulk_modality(
                    data,
                    self.from_key,  # type: ignore
                ),
                "to_modality": lambda data: process_bulk_to_bulk_modality(
                    data,
                    self.to_key,  # type: ignore
                ),
            },
        )

    def _process_sc_to_sc_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process SC_TO_SC case end-to-end.

        Reads single-cell data, prepares from/to modalities, performs data splitting,
        NaN removal, and applies single-cell specific filtering to from and to mudata objects.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
        """
        if self.from_key is None or self.to_key is None:
            raise TypeError("from_key and to_key cannot be None for Translation")

        if raw_user_data is None:
            screader = self.data_readers[DataCase.SINGLE_CELL_TO_SINGLE_CELL]
            mudata = screader.read_data(config=self.config)

            data_package = DataPackage()
            data_package.from_modality = {self.from_key: mudata[self.from_key]}
            data_package.to_modality = {self.to_key: mudata[self.to_key]}
        else:
            data_package = raw_user_data

        def process_sc_to_sc_modality(
            modality_data: Dict[str, MuData], modality_key: str
        ) -> Dict[str, MuData]:
            """Processes sc-to-sc modality data with single-cell filtering."""
            if modality_data and modality_key in modality_data:
                data_info = self.config.data_config.data_info[modality_key]
                sc_filter = SingleCellFilter(
                    mudata=modality_data[modality_key], data_info=data_info
                )
                return {modality_key: sc_filter.preprocess()}
            return modality_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": lambda data: process_sc_to_sc_modality(
                    data, self.from_key
                ),
                "to_modality": lambda data: process_sc_to_sc_modality(
                    data, self.to_key
                ),
            },
        )

    def _process_img_to_bulk_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process IMG_TO_BULK case end-to-end.

        Reads image and bulk data, prepares from/to modalities (IMG->BULK or BULK->IMG),
        performs data splitting, NaN removal, and applies normalization to image data
        and filtering/scaling to bulk dataframes.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
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
                # IMG -> BULK direction (Corrected condition) # Use self.to_key
                data_package.from_modality = {self.from_key: images[self.from_key]}  # type: ignore
                data_package.to_modality = {self.to_key: bulk_dfs[self.to_key]}
                to_annotation = next(iter(annotation.keys()))
                data_package.annotation = {
                    "from": None,
                    "to": annotation[to_annotation],
                }
            else:  # BULK -> IMG direction (Corrected condition order)
                data_package.from_modality = {self.from_key: bulk_dfs[self.from_key]}
                data_package.to_modality = {self.to_key: images[self.to_key]}  # type: ignore
                from_annotation = next(iter(annotation.keys()))
            data_package.annotation = {"from": annotation[from_annotation], "to": None}
        else:
            data_package = raw_user_data

        def process_img_to_bulk_modality(
            modality_data: Dict[str, Union[pd.DataFrame, List]], modality_key: str
        ) -> Dict[str, Union[pd.DataFrame, List]]:
            """Processes img-to-bulk modality data with normalization for images and filtering/scaling for dataframes."""
            if modality_key is None:
                raise TypeError("modality_key cannot be None")
            if modality_data and modality_key in modality_data:
                data = modality_data[modality_key]
                if isinstance(data, pd.DataFrame):
                    return {
                        modality_key: self._filter_and_scale_dataframe(  # type: ignore
                            df=data, info_key=modality_key
                        )
                    }
                elif self._is_image_data(data):
                    return {
                        modality_key: self._normalize_image_data(
                            images=data, info_key=modality_key
                        )
                    }
            return modality_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": lambda data: process_img_to_bulk_modality(
                    data, self.from_key
                ),
                "to_modality": lambda data: process_img_to_bulk_modality(
                    data, self.to_key
                ),
            },
        )

    def _process_sc_to_img_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """
        Process SC_TO_IMG case end-to-end.

        Reads single-cell and image data, prepares from/to modalities (SC->IMG or IMG->SC),
        performs data splitting, NaN removal, and applies single-cell specific filtering
        to single-cell data and normalization to image data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
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
            data_package.to_modality = {self.to_key: mudata}  # Use self.to_key
        else:  # SC -> IMG case (Corrected condition order)
            data_package.from_modality = {self.from_key: mudata}  # Use self.from_key
            data_package.to_modality = {self.to_key: images[self.to_key]}

        def process_sc_to_img_modality(
            modality_data: Dict[str, Union[MuData, List]], modality_key: str
        ) -> Dict[str, Union[MuData, List]]:
            """Processes sc-to-img modality data with single-cell filtering for mudata and normalization for images."""
            if not isinstance(modality_key, str):
                raise TypeError(
                    f"Modality key as to be string, got {type(modality_key)}"
                )
            if modality_data and modality_key in modality_data:
                data = modality_data[modality_key]
                if isinstance(data, MuData):
                    data_info = self.config.data_config.data_info[modality_key]
                    sc_filter = SingleCellFilter(mudata=data, data_info=data_info)
                    return {modality_key: sc_filter.preprocess()}
                elif self._is_image_data(data):
                    return {
                        modality_key: self._normalize_image_data(data, modality_key)
                    }
            return modality_data

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": lambda data: process_sc_to_img_modality(
                    data, self.from_key
                ),
                "to_modality": lambda data: process_sc_to_img_modality(
                    data, self.to_key
                ),
            },
        )

    def _process_img_to_img_case(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> Dict[str, DataPackage]:
        """
        Process IMG_TO_IMG case end-to-end.

        Reads image data for from/to modalities, performs data splitting,
        NaN removal, and applies normalization to both from and to image data.

        Returns:
            A dictionary containing processed DataPackage objects for each data split.
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

        def process_img_to_img_modality(
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

        return self._process_data_case(
            data_package,
            modality_processors={
                "from_modality": lambda data: process_img_to_img_modality(
                    data,
                    self.from_key,  # type: ignore
                ),
                "to_modality": lambda data: process_img_to_img_modality(
                    data,
                    self.to_key,  # type: ignore
                ),
            },
        )

    def _split_data_package(
        self, data_package: DataPackage
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, np.ndarray]]]:
        """
        Split data package into train/validation/test sets.

        Uses DataSplitter and DataPackageSplitter to divide the DataPackage
        into training, validation, and test sets based on the configuration.

        Parameters:
            data_package: The DataPackage to be split.

        Returns:
            A dictionary containing the split DataPackages, keyed by split names
            ('train', 'validation', 'test').

            split_indiced_config - (dict): the actual indicies used for splitting
        """
        data_splitter = DataSplitter(config=self.config)
        n_samples = data_package.get_n_samples(is_paired=self.config.paired_translation)

        split_indices_config: dict = {}
        if self.config.paired_translation or self.config.paired_translation is None:
            split_indices_config["paired"] = data_splitter.split(
                n_samples=n_samples["paired_count"]
            )
        else:
            split_indices_config["from_indices"] = data_splitter.split(
                n_samples=n_samples["from"]
            )
            split_indices_config["to_indices"] = data_splitter.split(
                n_samples=n_samples["to"]
            )
        data_splitter_instance = DataPackageSplitter(
            data_package=data_package,
            config=self.config,
            indices=split_indices_config.get("paired"),
            from_indices=split_indices_config.get("from_indices"),
            to_indices=split_indices_config.get("to_indices"),
        )

        return data_splitter_instance.split(), split_indices_config

    def _is_image_data(self, data: Any) -> bool:
        """
        Check if data is image data.

        Determines if the provided data is a list of objects that are considered
        image data based on having an 'img' attribute.

        Parameters:
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
        """
        Remove NaN values from the data package.

        Utilizes NaNRemover to identify and remove rows containing NaN values
        in relevant annotation columns within the DataPackage.

        Parameters:
            data_package: The DataPackage from which to remove NaNs.

        Returns:
            The DataPackage with NaN values removed.
        """
        nanremover = NaNRemover(
            relevant_cols=self.config.data_config.annotation_columns
        )
        return nanremover.remove_nan(data=data_package)

    def _filter_and_scale_dataframe(
        self, df: pd.DataFrame, info_key: str
    ) -> Optional[pd.DataFrame]:
        """
        Process a dataframe with filtering and scaling.

        Applies filtering based on DataFilter and scaling to a pandas DataFrame
        according to the data information specified by info_key.

        Parameters:
            df: The DataFrame to be processed.
            info_key: The key referencing data information in the configuration for filtering and scaling.

        Returns:
            The processed DataFrame after filtering and scaling, or None if the input DataFrame is None.
        """
        if df is None:
            return None

        data_info = self.config.data_config.data_info[info_key]
        if data_info.data_type == "ANNOTATION":
            return df

        filter_obj = DataFilter(df=df, data_info=data_info)
        filtered_df = filter_obj.filter()

        if filtered_df.empty or filtered_df.shape[1] == 0:
            filtered_df = df  # Return original if filter makes it empty

        scaled_df = filter_obj.scale(filtered_df)
        return scaled_df

    def _normalize_image_data(self, images: List, info_key: str) -> List:
        """
        Process images with normalization.

        Normalizes a list of image data objects using ImageNormalizer based on
        the scaling method specified in the configuration for the given info_key.

        Parameters:
            images: A list of image data objects (each having an 'img' attribute).
            info_key: The key referencing data information in the configuration to get the scaling method.

        Returns:
            A list of processed image data objects with normalized image data.
        """
        scaling_method = self.config.data_config.data_info[info_key].scaling
        processed_images = []
        normalizer = ImageNormalizer()  # Instance created once here

        for img in images:
            img_copy = copy.deepcopy(img)
            img_copy.img = normalizer.normalize_image(  # Use instance here
                image=img_copy.img, method=scaling_method
            )
            processed_images.append(img_copy)

        return processed_images

    def _get_translation_keys(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract from and to keys from config.

        Retrieves the 'from' and 'to' modality keys from the data configuration
        based on the 'translate_direction' setting.

        Returns:
            A tuple containing the from_key and to_key as strings, or None if not found.
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
        if len(raw_user_data.from_modality) == 0:
            return None, None
        elif len(raw_user_data.to_modality) == 0:
            return None, None
        else:
            try:
                return next(iter(raw_user_data.from_modality.keys())), next(
                    iter(raw_user_data.to_modality.keys())
                )
            except Exception as e:
                print("error getting from or to keys")
                print(e)
                print("returning None")
                return None, None
