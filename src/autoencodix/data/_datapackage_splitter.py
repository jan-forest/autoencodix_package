from typing import Any, Dict, List, Literal, Optional, TypeVar

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
            if key == "annotation":
                print("processing ANNOTATION")
                print(self._data_package)

            split_result = self._package_splitter(value, indices)
            if split_result is not None:  # Only include non-None results
                split_data[key] = split_result

        return DataPackage(**split_data)

    def _package_splitter(
        self,
        dataobj: Dict,
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
        Optional[Dict]
            The split data object, or None if input is None.
        """
        if dataobj is None:
            return None

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
            elif isinstance(first_value, (AnnData, MuData)):
                return  {key: value[indices] for key, value in dataobj.items()}
            else:
                raise TypeError(
                    f"Expected dataobj to be pd.DataFrame, list or MuData, got {type(dataobj)}"
                )
        else:
            raise TypeError(f" dataobj should be dict or None, got {type(dataobj)}")

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
            print("splitting unpaired annotations")

        return result

    def split(self) -> Dict[str, Dict[str, Any]]:
        """
        Build datasets for training, validation, and testing from a DataPackage.

        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]
            Dictionary containing data and indices for train, valid, and test splits.
            Exmaple: {"train": {"data": DataPackage(), "indicies": {"paired": np.ndarray}}, "valid" ...}

        Raises
        ------
        ValueError
            If no data is available for splitting.
        """
        if self._data_package is None:
            raise ValueError("No data package available for splitting")

        result = {}

        if self.config.paired_translation is None or self.config.paired_translation:
            for split_name in ["train", "valid", "test"]:
                split_data = self._split_data_package(self.indices[split_name])
                result[split_name] = {
                    "data": split_data,
                    "indices": {"paired": self.indices[split_name]},
                }

            print(f" splitting result train : {result['train']['data']}")
            return result
        # UNPAIRED CASE
        for split_name in ["train", "valid", "test"]:
            split_package = DataPackage()
            split_package = self._split_modality_and_annotation(
                self._data_package, self.to_indices[split_name], "to"
            )

            split_package = self._split_modality_and_annotation(
                split_package, self.from_indices[split_name], "from"
            )
            result[split_name] = {
                "data": split_package,
                "indices": {
                    "from": self.from_indices[split_name],
                    "to": self.to_indices[split_name],
                },
            }
        print(f" splitting result train : {result['train']['data']}")
        return result
