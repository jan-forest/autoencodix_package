import copy
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore
from mudata import MuData  # type: ignore

from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.default_config import DefaultConfig


class DataPackageSplitter:
    """Splits DataPackage objects into training, validation, and testing sets.

    Supports paired and unpaired (translation) splitting.

    Attributes:
        data_package: The original DataPackage to split.
        config: The configuration settings for the splitting process.
        indices: The indices for each split (train/val/test).
    """

    def __init__(
        self,
        data_package: DataPackage,
        config: DefaultConfig,
        indices: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ) -> None:
        """Store the DataPackage, config and precomputed split indices.

        Args:
            data_package: The DataPackage to split.
            config: Pipeline configuration.
            indices: Sample indices per data attribute, modality and split, i.e.
                ``indices[attribute][modality][split]``.

        Raises:
            TypeError: If ``data_package`` is not a DataPackage.

        (LLM-generated docstring, pending human review.)
        """
        self._data_package = data_package
        self.indices = indices
        self.config = config

        if not isinstance(self._data_package, DataPackage):
            raise TypeError(
                f"Expected data_package to be of type DataPackage, got {type(self._data_package)}"
            )

    def _shallow_copy(self, value: Any) -> Any:
        """Return a shallow copy of ``value``, or ``value`` itself if it cannot be copied.

        (LLM-generated docstring, pending human review.)
        """
        try:
            return copy.copy(value)
        except AttributeError:
            return value

    def _indexing(self, obj: Any, indices: np.ndarray) -> Any:
        """Indexes pd.DataFrame, list, AnnData, or MuData objects using the provided indices.

        Args:
            obj: The object to index (can be pd.DataFrame, list, AnnData, MuData, or None).
            indices: The indices to use for indexing.
        Returns:
            The indexed object, or None if the input object is None.
        Raises:
            TypeError: If an unsupported type is encountered.
        """

        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[indices]
        elif isinstance(obj, list):
            return [obj[i] for i in indices]
        elif isinstance(obj, (AnnData, MuData)):
            # print(f"shape of obj: {obj.shape}")
            # print(f"obj: {obj}")
            # print(f"len(ind): {len(indices)}")
            # print(f"max of index{np.max(indices)}")
            # print(f"ind: {indices}")
            return obj[indices]
        else:
            raise TypeError(
                f"Unsupported type for indexing: {type(obj)}. "
                "Supported types are pd.DataFrame, list, AnnData, and MuData."
            )

    def _split_data_package(self, split: str) -> Optional[DataPackage]:
        """Create a DataPackage that contains only the samples of one split.

        Args:
            split: Name of the split ("train", "valid" or "test").

        Returns:
            A new DataPackage with every modality indexed by the split's indices,
            or None if no indices are available.
        """
        if len(self.indices) == 0:
            return None

        split_data = {}
        for key, value in self._data_package.__dict__.items():
            if value is None:
                continue
            split_data[key] = {
                modality: self._indexing(data, self.indices[key][modality][split])
                for modality, data in value.items()
            }
        return DataPackage(**split_data)

    def _split_mudata(
        self,
        mudata: MuData,  # ty: ignore[invalid-type-form]
        indices_map: Dict[str, Dict[str, np.ndarray]],
        split: str,
    ) -> MuData:  # ty: ignore[invalid-type-form]
        """Splits a MuData object based on the provided indices map.

        Args:
            mudata: The MuData object to split.
            indices_map: A dictionary mapping modalities to their respective indices.
            split: The split type ("train", "valid", or "test").
        Returns:
            A new MuData object with the specified splits applied.
        """
        for modality, data in mudata.mod.items():
            indices = indices_map[modality][split]
            mudata.mod[modality] = self._indexing(data, indices)
        return mudata

    def _requires_paired(self) -> bool:
        """Return True unless the config explicitly allows unpaired data.

        (LLM-generated docstring, pending human review.)
        """
        return self.config.requires_paired is None or self.config.requires_paired

    def split(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Splits the underlying DataPackage into train, valid, and test subsets.
        Returns:
            A dictionary containing the split data packages for "train", "valid", and "test".
            Each entry contains a "data" key with the DataPackage and an "indices" key with
            the corresponding indices.
        Raises:
            ValueError: If no data package is available for splitting.
            TypeError: If indices are not provided for unpaired translation case.
        """
        if self._data_package is None:
            raise ValueError("No data package available for splitting")

        splits = ["train", "valid", "test"]
        result: Dict[str, Optional[Dict[str, Any]]] = {
            "train": {},
            "valid": {},
            "test": {},
        }

        for split in splits:
            if self.indices is None:  # or split not in self.indices:
                result[split] = None
                continue
            result[split] = {
                "data": self._split_data_package(split=split),
                "indices": self.indices,
            }

        return result
