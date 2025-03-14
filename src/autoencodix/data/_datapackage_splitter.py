from typing import Any, Dict, List, Literal, Optional, Union
import copy
import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore
from mudata import MuData  # type: ignore

from autoencodix.data._datapackage import DataPackage
from autoencodix.utils.default_config import DefaultConfig


class DataPackageSplitter:
    """
    Splits DataPackage objects into training, validation, and testing sets.
    Supports paired and unpaired (translation) splitting.
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

    def _shallow_copy(self, value: Any) -> Any:
        try:
            return copy.copy(value)
        except Exception:
            return value

    def _indexing(self, obj: Any, indices: np.ndarray) -> Any:
        """
        Indexes objects based on valid indices. For recognized types (DataFrame, list, AnnData/MuData),
        returns a subset based on the indices. For other types, returns the object unchanged.
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

    def _split_data_package(self, indices: np.ndarray) -> DataPackage:
        """
        Creates a new DataPackage where each attribute is indexed (if applicable)
        by the given indices.
        """
        if len(indices) == 0:
            return DataPackage()

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
            elif isinstance(value, dict):
                first_val = next(iter(value.values()), None)
                if isinstance(first_val, (pd.DataFrame, list, AnnData, MuData)):
                    split_data[key] = {
                        k: self._indexing(v, indices) for k, v in value.items()
                    }
                else:
                    split_data[key] = dict(value)
            else:
                split_data[key] = self._shallow_copy(value)

        return DataPackage(**split_data)

    def _create_modality_specific_package(
        self,
        original_package: DataPackage,
        indices: np.ndarray,
        modality_type: Literal["to", "from"],
    ) -> DataPackage:
        """
        Returns a new DataPackage with the specified modality (either "to" or "from")
        split using the given indices.
        """
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
            result.annotation = dict(original_package.annotation)
            ann = original_package.annotation.get(modality_type)
            if ann is not None:
                result.annotation[modality_type] = self._indexing(ann, indices)
        return result

    def _merge_packages(self, packages: List[DataPackage]) -> DataPackage:
        """
        Merges multiple DataPackage objects into a single package.
        For most attributes, the first non-None value is taken (via deepcopy),
        while annotations are merged.
        """
        if not packages:
            return DataPackage()

        result = DataPackage()
        for pkg in packages:
            for key, value in pkg.__dict__.items():
                if value is None:
                    continue
                if key == "annotation":
                    result.annotation = result.annotation or {}
                    for k, v in value.items():
                        result.annotation[k] = copy.deepcopy(v)
                elif key.endswith("_modality") and not getattr(result, key, None):
                    setattr(result, key, copy.deepcopy(value))
                elif not getattr(result, key, None):
                    setattr(result, key, copy.deepcopy(value))
        return result

    def split(self) -> Dict[str, Dict[str, Any]]:
        """
        Splits the underlying DataPackage into train, valid, and test subsets.
        Returns a dictionary where each key ("train", "valid", "test") maps to a
        dictionary with the split DataPackage and the corresponding indices.
        """
        if self._data_package is None:
            raise ValueError("No data package available for splitting")

        splits = ["train", "valid", "test"]
        result = {}

        if self.config.paired_translation is None or self.config.paired_translation:
            if self.indices is None:
                raise ValueError("In paired/normal case we need split indices")
            result = {
                split: {
                    "data": self._split_data_package(self.indices[split]),
                    "indices": {"paired": self.indices[split]},
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
                    self._data_package, self.to_indices[split], "to"
                )
                package_split = self._create_modality_specific_package(
                    package_split, self.from_indices[split], "from"
                )
                result[split] = {
                    "data": package_split,
                    "indices": {
                        "to": self.to_indices[split],
                        "from": self.from_indices[split],
                    },
                }
        return result
