import copy
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore
from mudata import MuData  # type: ignore

from autoencodix.data.datapackage import DataPackage
from autoencodix.utils.default_config import DefaultConfig


class DataPackageSplitter:
    """Splits DataPackage objects into training, validation, and testing sets.

    Supports paired and unpaired (translation) splitting.

    Attributes:
        data_package: The original DataPackage to split.
        config: The configuration settings for the splitting process.
        indices: The indices for each split (train/val/test).
        to_indices: The indices for the "to" modality (if applicable).
        from_indices: The indices for the "from" modality (if applicable).
    """

    def __init__(
        self,
        data_package: DataPackage,
        config: DefaultConfig,
        indices: Union[None, Dict[str, np.ndarray]] = None,
        to_indices: Union[Dict[str, np.ndarray], None] = None,
        from_indices: Union[Dict[str, np.ndarray], None] = None,
    ) -> None:
        self._data_package = data_package
        self.indices = indices
        self.config = config
        self.to_indices = to_indices
        self.from_indices = from_indices

        if not isinstance(self._data_package, DataPackage):
            raise TypeError(
                f"Expected data_package to be of type DataPackage, got {type(self._data_package)}"
            )

    def _shallow_copy(self, value: Any) -> Any:
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
        """

        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[indices]
        if isinstance(obj, list):
            return [obj[i] for i in indices]
        if isinstance(obj, (AnnData, MuData)):
            return obj[indices]
        return obj

    def _split_data_package(self, indices: np.ndarray) -> Optional[DataPackage]:
        """Creates a new DataPackage where each attribute is indexed (if applicable)
        by the given indices. Returns None if indices are empty.

        Args:
            indices: The indices to use for splitting the DataPackage.
        Returns:
            A new DataPackage with attributes indexed by the provided indices,
            or None if indices are empty.
        Raises:
            TypeError: If an unsupported type is encountered in the DataPackage attributes.
        """
        if len(indices) == 0:
            return None

        split_data = {}
        for key, value in self._data_package.__dict__.items():
            if value is None:
                continue

            if key == "multi_sc":
                if isinstance(value, dict):
                    split_data[key] = {
                        k: self._indexing(v, indices) for k, v in value.items()
                    }
                elif isinstance(value, (AnnData, MuData)):
                    split_data[key] = self._indexing(value, indices)
                continue
            elif isinstance(value, dict):
                first_val = next(iter(value.values()), None)
                if isinstance(first_val, (pd.DataFrame, list, AnnData, MuData)):
                    split_data[key] = {
                        k: self._indexing(v, indices) for k, v in value.items()
                    }
                else:
                    warnings.warn(
                        f"Skipping indexing for key '{key}' with unsupported inner type {type(first_val)}."
                    )
                    split_data[key] = dict(value)
            else:
                raise TypeError(
                    f"Unsupported type {type(value)} for attribute '{key}'. Has it been implemented in the DataPackage class?"
                )

        return DataPackage(**split_data)

    def _create_modality_specific_package(
        self,
        original_package: DataPackage,
        indices: np.ndarray,
        modality_type: Literal["to", "from"],
    ) -> Optional[DataPackage]:
        """Returns a new DataPackage with the specified modality (either "to" or "from")
        split using the given indices. Returns None if indices are empty.

        Args:
            original_package: The original DataPackage to split.
            indices: The indices to use for splitting the modality.
            modality_type: The type of modality to split ("to" or "from").
        Returns:
            A new DataPackage with the specified modality split by the provided indices,
            or None if indices are empty.
        """
        if len(indices) == 0:
            return None

        result = DataPackage()

        # Process multi_sc (for single cell data)
        multi_sc = getattr(original_package, "multi_sc", None)
        if multi_sc is not None:
            if isinstance(multi_sc, dict):
                result.multi_sc = {
                    k: self._indexing(v, indices) for k, v in multi_sc.items()
                }
            elif isinstance(multi_sc, (AnnData, MuData)):
                result.multi_sc = self._indexing(multi_sc, indices)

        # Process modalities
        to_mod = getattr(original_package, "to_modality", None)
        from_mod = getattr(original_package, "from_modality", None)
        if to_mod is not None:
            if modality_type == "to":
                result.to_modality = {
                    k: self._indexing(v, indices) for k, v in to_mod.items()
                }
            else:
                result.to_modality = dict(to_mod)
        if from_mod is not None:
            if modality_type == "from":
                result.from_modality = {
                    k: self._indexing(v, indices) for k, v in from_mod.items()
                }
            else:
                result.from_modality = dict(from_mod)

        # Process annotations if present
        if hasattr(original_package, "annotation") and original_package.annotation:
            if isinstance(original_package.annotation, dict):
                result.annotation = dict(original_package.annotation)
            else:
                result.annotation = {}
            ann = original_package.annotation.get(modality_type)
            if ann is not None:
                result.annotation[modality_type] = self._indexing(ann, indices)
        return result

    def _is_paired_translation_enabled(self) -> bool:
        return self.config.paired_translation is None or self.config.paired_translation

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
        result: Dict[str, Optional[Dict[str, Any]]] = {}

        if self._is_paired_translation_enabled():
            if self.indices is None:
                raise ValueError("In paired/normal case we need split indices")
            result = {
                split: {
                    "data": self._split_data_package(indices=self.indices[split]),
                    "indices": {"paired": self.indices[split]}
                    if len(self.indices[split]) > 0
                    else {"paired": np.array([])},
                }
                for split in splits
            }
        else:
            if self.to_indices is None:
                raise TypeError("For Unpaired Case we need to_indices")
            if self.from_indices is None:
                raise TypeError("For Unpaired Case we need from_indices")

            for split in splits:
                package_split = self._create_modality_specific_package(
                    original_package=self._data_package,
                    indices=self.to_indices[split],
                    modality_type="to",
                )
                package_split = (
                    self._create_modality_specific_package(
                        original_package=package_split,
                        indices=self.from_indices[split],
                        modality_type="from",
                    )
                    if package_split is not None
                    else None
                )
                result[split] = {
                    "data": package_split,
                    "indices": {
                        "to": self.to_indices[split]
                        if len(self.to_indices[split]) > 0
                        else np.array([]),
                        "from": self.from_indices[split]
                        if len(self.from_indices[split]) > 0
                        else np.array([]),
                    },
                }
        return result
