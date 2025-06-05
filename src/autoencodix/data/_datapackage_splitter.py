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
        indices: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ) -> None:
        self._data_package = data_package
        self.indices = indices
        self.config = config

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
        elif isinstance(obj, list):
            return [obj[i] for i in indices]
        elif isinstance(obj, (AnnData)):
            return obj[indices]
        else:
            raise TypeError(
                f"Unsupported type for indexing: {type(obj)}. "
                "Supported types are pd.DataFrame, list, AnnData, and MuData."
            )

    def _split_data_package(self, split: str) -> Optional[DataPackage]:
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
        if len(self.indices) == 0:
            return None

        split_data = {}
        for key, value in self._data_package.__dict__.items():
            if value is None:
                continue
            if key == "multi_sc":
                mudata = value["multi_sc"]
                split_data[key] = self._split_mudata(mudata, self.indices[key], split)
            elif key in ["to_modality", "from_modality"]:
                if isinstance(value[key], MuData):
                    mudata = value[key]
                    split_data[key] = self._split_mudata(
                        mudata, self.indices[key], split
                    )
            else:
                split_data[key] = {
                    modality: self._indexing(data, self.indices[key][modality][split])
                    for modality, data in value.items()
                }
        return DataPackage(**split_data)

    def _split_mudata(
        self, mudata, indices_map: Dict[str, Dict[str, np.ndarray]], split: str
    ) -> MuData:
        """Splits a MuData object based on the provided indices map.

        Args:
            mudata: The MuData object to split.
            indices_map: A dictionary mapping modalities to their respective indices.
        Returns:
            A new MuData object with the specified splits applied.
        """
        for modality, data in mudata.mod.items():
            indices = indices_map[modality][split]
            mudata.mod[modality] = self._indexing(data, indices)
        return mudata

    def _requires_paired(self) -> bool:
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
        result: Dict[str, Optional[Dict[str, Any]]] = {}

        # if self._requires_paired():
            # check_lens = []
            # for _, sub_dict in self.indices.items():
            #     for _, cur_splits in sub_dict.items():
            #         for split in splits:
            #             ids = cur_splits[split]
            #             if len(ids) == 0:
            #                 continue
            #             check_lens.append(len(ids))
            # if not len(set(check_lens)) == 1:
            #     raise ValueError(
            #         "All splits must have the same number of indices in the paired case."
            #     )

        for split in splits:
            if self.indices is None or split not in self.indices:
                result[split] = None
                continue
            result = {
                split: {
                    "data": self._split_data_package(split=split),
                    "indices": self.indices
                }
            }
        return result
