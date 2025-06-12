from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, TypeVar

import pandas as pd
from anndata import AnnData  # type: ignore
from mudata import MuData  # type: ignore

from autoencodix.data._imgdataclass import ImgData


T = TypeVar("T")  # For generic type hints


@dataclass
class DataPackage:
    """
    A class to represent a data package containing multiple types of data.
    """

    multi_sc: Optional[Dict[str, MuData]] = None
    multi_bulk: Optional[Dict[str, pd.DataFrame]] = None
    annotation: Optional[Dict[str, Union[pd.DataFrame, None]]] = None
    img: Optional[Dict[str, List[ImgData]]] = None

    from_modality: Optional[
        Dict[str, Union[pd.DataFrame, List[ImgData], MuData, AnnData]]
    ] = field(default_factory=dict, repr=False)
    to_modality: Optional[
        Dict[str, Union[pd.DataFrame, List[ImgData], MuData, AnnData]]
    ] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to top-level attributes.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in DataPackage.")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-like item assignment to top-level attributes.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} not found in DataPackage.")

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Make DataPackage iterable, yielding (key, value) pairs.
        For dictionary attributes, yields nested items as (parent_key.child_key, value).
        """
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)

            if attr_value is None:
                continue
            if isinstance(attr_value, dict):
                for sub_key, sub_value in attr_value.items():
                    yield f"{attr_name}.{sub_key}", sub_value
            else:
                yield attr_name, attr_value

    def format_shapes(self) -> str:
        """Format the shape dictionary in a clean, readable way."""
        shapes = self.shape()
        lines = []

        for data_type, data_info in shapes.items():
            # Skip empty entries
            if not data_info:
                continue

            sub_items = []
            for subtype, shape in data_info.items():
                if shape is not None:
                    if isinstance(shape, tuple):
                        sub_items.append(
                            f"{subtype}: {shape[0]} samples × {shape[1]} features"
                        )
                    else:
                        sub_items.append(f"{subtype}: {shape} items")

            if sub_items:
                lines.append(f"{data_type}:")
                lines.extend(f"  {item}" for item in sub_items)

        if not lines:
            return "Empty DataPackage"

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_shapes()

    def __repr__(self) -> str:
        return self.__str__()

    def is_empty(self) -> bool:
        """Check if the data package is empty."""
        return all(
            [
                self.multi_sc is None,
                self.multi_bulk is None or len(self.multi_bulk) == 0,
                self.annotation is None,
                self.img is None,
                not self.from_modality,
                not self.to_modality,
            ]
        )

    def get_n_samples(self) -> Dict[str, Dict[str, int]]:
        """Get the number of samples for each data type in nested dictionary format.

        Returns:
            Dictionary with nested structure: {modality_type: {sub_key: count}}
        """
        n_samples: Dict[str, Dict[str, int]] = {}

        # Process each main attribute
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)

            if isinstance(attr_value, dict):
                # Handle dictionary attributes (multi_sc, multi_bulk, etc.)
                sub_counts = {}
                for sub_key, sub_value in attr_value.items():
                    if sub_value is None or len(sub_value) == 0:
                        continue
                    sub_counts[sub_key] = self._get_n_samples(sub_value)
                n_samples[attr_name] = sub_counts if sub_counts else {}
            else:
                # Handle non-dictionary attributes
                count = self._get_n_samples(attr_value)
                n_samples[attr_name] = {attr_name: count}

        paired_count = self._calculate_paired_count()
        n_samples["paired_count"] = {"paired_count": paired_count}

        return n_samples

    def _calculate_paired_count(self) -> int:
        """
        Calculate the number of samples that are common across modalities that have data.

        Returns:
            Number of common samples across modalities with data
        """
        all_counts = []

        # Collect all sample counts from modalities that have data
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                continue

            if isinstance(attr_value, dict):
                if attr_value:  # Non-empty dictionary
                    for sub_value in attr_value.values():
                        if sub_value is not None:
                            count = self._get_n_samples(sub_value)
                            if count > 0:
                                all_counts.append(count)
            else:
                count = self._get_n_samples(attr_value)
                if count > 0:
                    all_counts.append(count)

        # Return minimum count (intersection) or 0 if no data
        return min(all_counts) if all_counts else 0

    def get_common_ids(self) -> List[str]:
        """
        Get the common sample IDs across modalities that have data.

        Returns:
            List of sample IDs that are present in all modalities with data
        """
        all_ids = []

        # Collect sample IDs from each modality that has data
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                continue

            if isinstance(attr_value, dict):
                if attr_value:  # Non-empty dictionary
                    for sub_value in attr_value.values():
                        if sub_value is not None:
                            ids = self._get_sample_ids(sub_value)
                            if ids:
                                all_ids.append(set(ids))
            else:
                ids = self._get_sample_ids(attr_value)
                if ids:
                    all_ids.append(set(ids))

        # Find intersection of all ID sets
        if not all_ids:
            return []

        common = all_ids[0]
        for id_set in all_ids[1:]:
            common = common.intersection(id_set)

        return sorted(list(common))

    def _get_sample_ids(
        self, dataobj: Union[MuData, pd.DataFrame, List[ImgData], AnnData]
    ) -> List[str]:
        """
        Extract sample IDs from a data object.

        Parameters:
            dataobj: Data object to extract IDs from

        Returns:
            List of sample IDs
        """
        if dataobj is None:
            return []

        if isinstance(dataobj, pd.DataFrame):
            return dataobj.index.astype(str).tolist()
        elif isinstance(dataobj, list):
            # For lists of ImgData, extract sample_id from each object
            return [
                img_data.sample_id
                for img_data in dataobj
                if hasattr(img_data, "sample_id")
            ]
        elif isinstance(dataobj, AnnData):
            return dataobj.obs.index.astype(str).tolist()
        elif isinstance(dataobj, MuData):
            # For MuData, we can use the obs.index directly
            return dataobj.obs.index.astype(str).tolist()
        else:
            return []

    def _get_n_samples(
        self, dataobj: Union[MuData, pd.DataFrame, List[ImgData], AnnData, Dict]
    ) -> int:
        """Get the number of samples for a specific attribute."""
        if dataobj is None:
            return 0

        if isinstance(dataobj, pd.DataFrame):
            return dataobj.shape[0]
        elif isinstance(dataobj, dict):
            if not dataobj:  # Empty dict
                return 0
            first_value = next(iter(dataobj.values()))
            return self._get_n_samples(first_value)
        elif isinstance(dataobj, list):
            return len(dataobj)
        elif isinstance(dataobj, AnnData):
            return dataobj.obs.shape[0]
        elif isinstance(dataobj, MuData):
            if not dataobj.mod:  # Empty MuData
                return 0
            return dataobj.n_obs
        else:
            raise ValueError(
                f"Unknown data type {type(dataobj)} for dataobj. Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
            )

    def shape(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the shape of the data for each data type in nested dictionary format.

        Returns:
            Dictionary with nested structure: {modality_type: {sub_key: shape}}
        """
        shapes: Dict[str, Dict[str, Any]] = {}

        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)

            if isinstance(attr_value, dict):
                # Handle dictionary attributes
                if attr_value is None or len(attr_value) == 0:
                    # Empty or None dictionary
                    shapes[attr_name] = {}
                else:
                    sub_dict = self._get_shape_from_dict(attr_value)
                    shapes[attr_name] = sub_dict
            else:
                # Handle non-dictionary attributes
                shape = self._get_single_shape(attr_value)
                shapes[attr_name] = {attr_name: shape}

        return shapes

    def _get_single_shape(self, dataobj: Any) -> Optional[Union[Tuple, int]]:
        """Get shape for a single data object."""
        if dataobj is None:
            return None
        elif isinstance(dataobj, list):
            return len(dataobj)
        elif isinstance(dataobj, pd.DataFrame):
            return dataobj.shape
        elif isinstance(dataobj, AnnData):
            return dataobj.shape
        elif isinstance(dataobj, MuData):
            return (dataobj.n_obs, dataobj.n_vars)
        else:
            return None

    def _get_shape_from_dict(self, data_dict: Dict) -> Dict[str, Any]:
        """
        Recursively process dictionary to extract shapes of contained data objects.

        Parameters:
            data_dict: Dictionary containing data objects

        Returns:
            Dictionary with shapes information
        """
        result: Dict[str, Any] = {}
        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame):
                result[key] = value.shape
            elif isinstance(value, list):
                # For lists of objects, just store the length
                result[key] = len(value)
            elif isinstance(value, AnnData):
                result[key] = value.shape
            elif isinstance(value, MuData):
                result[key] = (value.n_obs, value.n_vars)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_result = self._get_shape_from_dict(value)
                result[key] = nested_result
            elif value is None:
                result[key] = None
            else:
                # For unknown types, store a descriptive string instead of raising an error
                # This is more robust as it won't crash the entire method
                result[key] = f"<{type(value).__name__}>"

        return result

    def get_modality_key(self, direction: str) -> Optional[str]:
        """
        Get the first key for a specific direction's modality.

        Parameters:
            direction: Either 'from' or 'to'

        Returns:
            First key of the modality dictionary or None if empty
        """
        if direction not in ["from", "to"]:
            raise ValueError(f"Direction must be 'from' or 'to', got {direction}")

        modality_dict = self.from_modality if direction == "from" else self.to_modality
        if not modality_dict:
            return None

        return next(iter(modality_dict.keys()), None)
