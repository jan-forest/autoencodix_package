# from typing import Dict, List, Union, Literal

# import numpy as np
# import pandas as pd
# from anndata import AnnData
# from mudata import MuData

# from autoencodix.data._datapackage import DataPackage
# from autoencodix.data._imgdataclass import ImgData
# from autoencodix.utils.default_config import DefaultConfig


# class DataPackageSplitter:
#     def __init__(
#         self,
#         data_package: DataPackage,
#         indicies: Dict[str, np.ndarray],
#         config: DefaultConfig,
#         to_indicies: Dict[str, np.ndarray] = None,  # for unpaired translation
#         from_indicies: Dict[str, np.ndarray] = None,
#     ) -> None:
#         self._data_package = data_package
#         self.indicies = indicies
#         self.config = config
#         self.to_indicies = to_indicies
#         self.from_indicies = from_indicies

#     def _split_data_package(self, indices: np.ndarray) -> DataPackage:
#         """
#         Split the data package according to the provided indices.

#         Parameters
#         ----------
#         indices : np.ndarray
#             Indices to use for splitting the data.

#         Returns
#         -------
#         DataPackage
#             A new DataPackage containing the split data.
#         """
#         split = {}
#         for key, value in self._data_package.__dict__.items():
#             split[key] = self._package_splitter(value, indices)
#         return DataPackage(**split)

#     def _package_splitter(
#         self,
#         dataobj: Union[MuData | pd.DataFrame | List[ImgData] | AnnData],
#         indices: np.ndarray,
#     ) -> Union[MuData | pd.DataFrame | List[ImgData] | AnnData]:
#         if len(indices) == 0:
#             return DataPackage()
#         if dataobj is not None:
#             if isinstance(dataobj, pd.DataFrame):
#                 return dataobj.iloc[indices]
#             elif isinstance(dataobj, dict):
#                 if isinstance(next(iter(dataobj.values())), list):
#                     return {
#                         key: [value[i] for i in indices]
#                         for key, value in dataobj.items()
#                     }
#                 if isinstance(next(iter(dataobj.values())), pd.DataFrame):
#                     return {key: value.iloc[indices] for key, value in dataobj.items()}
#             elif isinstance(dataobj, list):
#                 return [value for i, value in enumerate(dataobj) if i in indices]
#             elif isinstance(dataobj, AnnData):
#                 return dataobj[indices]
#             elif isinstance(dataobj, MuData):
#                 return dataobj[indices]
#             else:
#                 return dataobj

#     def _split_translation(
#         self,
#         indices: np.ndarray,
#         data,
#         direction: Literal["to", "from"],
#         datapackage: DataPackage,
#     ) -> DataPackage:
#         if direction == "to":
#             dataobj = data.to_modality
#             annotation = data.annotation["to"]
#             split_data = self._package_splitter(dataobj=dataobj, indices=indices)
#             split_anno = self._package_splitter(dataobj=annotation, indices=indices)
#             datapackage.to_modality = split_data
#             datapackage.annotation["to"] = split_anno

#         elif direction == "from":
#             dataobj = data.from_modality
#             annotation = data.annotation["from"]
#             split_data = self._package_splitter(dataobj=dataobj, indices=indices)
#             split_anno = self._package_splitter(dataobj=annotation, indices=indices)
#             datapackage.from_modality = split_data
#             datapackage.annotation["from"] = split_anno
#         else:
#             raise ValueError(f"Invalid direction: {direction}, use 'to' or 'from'")
#         return datapackage

#     def split(self) -> None:
#         """
#         Build datasets for training, validation, and testing from a DataPackage.
#         Uses pre-aligned IDs to split the data.

#         Raises
#         ------
#         ValueError
#             If no data is available for splitting.
#         """
#         if self._data_package is None:
#             raise ValueError("No data package available for splitting")

#         # Get split indices based on aligned IDs
#         if not self.config.is_paired:
#             train_package = self._split_translation(
#                 indices=self.indicies["train"],
#                 data=self._data_package,
#                 direction="to",
#                 datapackage=self._data_package,
#             )
#             train_package = self._split_translation(
#                 indices=self.indicies["train"],
#                 data=train_package,
#                 direction="from",
#                 datapackage=train_package,
#             )
#             valid_package = self._split_translation(
#                 indices=self.indicies["valid"],
#                 data=self._data_package,
#                 direction="to",
#                 datapackage=self._data_package,
#             )
#             valid_package = self._split_translation(
#                 indices=self.indicies["valid"],
#                 data=valid_package,
#                 direction="from",
#                 datapackage=valid_package,
#             )
#             test_package = self._split_translation(
#                 indices=self.indicies["test"],
#                 data=self._data_package,
#                 direction="to",
#                 datapackage=self._data_package,
#             )
#             test_package = self._split_translation(
#                 indices=self.indicies["test"],
#                 data=test_package,
#                 direction="from",
#                 datapackage=test_package,
#             )
#             return {
#                 "train": {"data": train_package, "indices": self.indicies["train"]},
#                 "valid": {
#                     "data": valid_package,
#                     "indices": self.indicies["valid"],
#                 },
#                 "test": {
#                     "data": test_package,
#                     "indices": self.indicies["test"],
#                 },
#             }
#         train_data = self._split_data_package(self.indicies["train"])
#         valid_data = self._split_data_package(self.indicies["valid"])
#         test_data = self._split_data_package(self.indicies["test"])
#         return {
#             "train": {"data": train_data, "indices": self.indicies["train"]},
#             "valid": {"data": valid_data, "indices": self.indicies["valid"]},
#             "test": {"data": test_data, "indices": self.indicies["test"]},
#         }

from typing import Dict, List, Union, Literal, Optional, Any, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from autoencodix.data._datapackage import DataPackage
from autoencodix.data._imgdataclass import ImgData
from autoencodix.utils.default_config import DefaultConfig

# Define type variable for better typing
T = TypeVar("T", MuData, pd.DataFrame, List[ImgData], AnnData, dict, list)


class DataPackageSplitter:
    """
    A class for splitting DataPackage objects into training, validation, and testing sets.
    Supports both paired and unpaired (translation) data splitting.
    """

    def __init__(
        self,
        data_package: DataPackage,
        config: DefaultConfig,
        indices: Dict[str, np.ndarray] = None,
        to_indices: Optional[Dict[str, np.ndarray]] = None,  # for unpaired translation
        from_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Initialize the DataPackageSplitter.

        Parameters
        ----------
        data_package : DataPackage
            The data package to split.
        indices : Dict[str, np.ndarray]
            Dictionary containing indices for train, valid, and test splits.
        config : DefaultConfig
            Configuration object.
        to_indices : Optional[Dict[str, np.ndarray]]
            Indices for "to" modality in unpaired translation.
        from_indices : Optional[Dict[str, np.ndarray]]
            Indices for "from" modality in unpaired translation.
        """
        self._data_package = data_package
        self.indices = indices
        self.config = config
        self.to_indices = to_indices
        self.from_indices = from_indices

    def _split_data_package(self, indices: np.ndarray) -> DataPackage:
        """
        Split the data package according to the provided indices.

        Parameters
        ----------
        indices : np.ndarray
            Indices to use for splitting the data.

        Returns
        -------
        DataPackage
            A new DataPackage containing the split data.
        """
        if len(indices) == 0:
            return DataPackage()

        # Create a new package with split data
        split_data = {}
        for key, value in self._data_package.__dict__.items():
            split_result = self._package_splitter(value, indices)
            if split_result is not None:  # Only include non-None results
                split_data[key] = split_result

        return DataPackage(**split_data)

    def _package_splitter(
        self,
        dataobj: Optional[T],
        indices: np.ndarray,
    ) -> Optional[T]:
        """
        Split different types of data objects based on indices.

        Parameters
        ----------
        dataobj : Optional[Union[MuData, pd.DataFrame, List[ImgData], AnnData, dict, list]]
            The data object to split.
        indices : np.ndarray
            Indices to use for splitting.

        Returns
        -------
        Optional[Union[MuData, pd.DataFrame, List[ImgData], AnnData, dict, list]]
            The split data object, or None if input is None.
        """
        if dataobj is None:
            return None

        # Handle DataFrame
        if isinstance(dataobj, pd.DataFrame):
            return dataobj.iloc[indices]

        # Handle dictionary of lists or DataFrames
        elif isinstance(dataobj, dict):
            # Get the first value to determine the type
            if not dataobj:  # Empty dictionary
                return {}

            first_value = next(iter(dataobj.values()))

            if isinstance(first_value, list):
                return {
                    key: [value[i] for i in indices if i < len(value)]
                    for key, value in dataobj.items()
                }
            elif isinstance(first_value, pd.DataFrame):
                return {key: value.iloc[indices] for key, value in dataobj.items()}
            else:
                return dataobj  # Return unchanged if values are not list or DataFrame

        # Handle list - use numpy's take for efficiency
        elif isinstance(dataobj, list):
            # Filter indices that are within range
            valid_indices = [i for i in indices if i < len(dataobj)]
            if not valid_indices:
                return []
            return [dataobj[i] for i in valid_indices]

        # Handle AnnData and MuData
        elif isinstance(dataobj, (AnnData, MuData)):
            return dataobj[indices]

        # Return unchanged for other types
        return dataobj

    def _split_modality_and_annotation(
        self,
        data_package: DataPackage,
        indices: np.ndarray,
        modality_type: Literal["to", "from"],
    ) -> DataPackage:
        """
        Split a specific modality and its annotations in the data package.

        Parameters
        ----------
        data_package : DataPackage
            The data package to split.
        indices : np.ndarray
            Indices to use for splitting.
        modality_type : Literal["to", "from"]
            Which modality to split ("to" or "from").

        Returns
        -------
        DataPackage
            A new DataPackage with the specified modality and annotations split.
        """
        # Create a new DataPackage to avoid modifying the original
        result = DataPackage()

        # Copy attributes from the original package
        for key, value in data_package.__dict__.items():
            if key != f"{modality_type}_modality" and (
                key != "annotation" or modality_type not in data_package.annotation
            ):
                setattr(result, key, value)

        # Split the modality data
        modality_attr = f"{modality_type}_modality"
        original_modality = getattr(data_package, modality_attr)
        if original_modality is not None:
            split_modality = self._package_splitter(original_modality, indices)
            setattr(result, modality_attr, split_modality)

        # Split the annotation if it exists
        if (
            hasattr(data_package, "annotation")
            and data_package.annotation
            and modality_type in data_package.annotation
        ):
            result.annotation = result.annotation.copy() if result.annotation else {}
            result.annotation[modality_type] = self._package_splitter(
                data_package.annotation[modality_type], indices
            )

        return result

    def split(self) -> Dict[str, Dict[str, Any]]:
        """
        Build datasets for training, validation, and testing from a DataPackage.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing data and indices for train, valid, and test splits.

        Raises
        ------
        ValueError
            If no data is available for splitting.
        """
        if self._data_package is None:
            raise ValueError("No data package available for splitting")

        result = {}

        # Handle paired vs unpaired data
        if not self.config.paired_translation:
            # For unpaired data, split "to" and "from" modalities separately
            for split_name in ["train", "valid", "test"]:
                # Create an empty DataPackage
                split_package = DataPackage()

                # Split "to" modality
                split_package = self._split_modality_and_annotation(
                    self._data_package, self.to_indices[split_name], "to"
                )

                # Split "from" modality
                split_package = self._split_modality_and_annotation(
                    split_package, self.from_indices[split_name], "from"
                )

                result[split_name] = {
                    "data": split_package,
                    "indices": {"from": self.from_indices[split_name], "to": self.to_indices[split_name]},
                }
        else:
            # For paired data, split the entire package at once
            for split_name in ["train", "valid", "test"]:
                split_data = self._split_data_package(self.indices[split_name])
                result[split_name] = {
                    "data": split_data,
                    "indices": {"paired":self.indices[split_name]},
                }

        return result
