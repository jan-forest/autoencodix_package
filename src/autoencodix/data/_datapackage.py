from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, Callable, TypeVar

import pandas as pd
from anndata import AnnData
from mudata import MuData

from autoencodix.data._imgdataclass import ImgData


T = TypeVar("T")  # For generic type hints


@dataclass
class DataPackage:
    """
    A class to represent a data package containing multiple types of data.
    """

    multi_sc: Optional[MuData] = None
    multi_bulk: Optional[Dict[str, pd.DataFrame]] = None
    annotation: Optional[Dict[str, pd.DataFrame]] = None
    img: Optional[Dict[str, List[ImgData]]] = None

    from_modality: Optional[
        Dict[str, Union[pd.DataFrame, List[ImgData], MuData, AnnData]]
    ] = field(default_factory=dict, repr=False)
    to_modality: Optional[
        Dict[str, Union[pd.DataFrame, List[ImgData], MuData, AnnData]]
    ] = field(default_factory=dict, repr=False)

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
            if data_info.get("shape") is None and not any(
                v for k, v in data_info.items() if k != "shape" and v is not None
            ):
                continue

            if "shape" in data_info and data_info["shape"] is not None:
                # Handle the case where we have a direct shape
                if isinstance(data_info["shape"], tuple):
                    samples, features = data_info["shape"]
                    lines.append(
                        f"{data_type}: {samples} samples × {features} features"
                    )
                else:
                    lines.append(f"{data_type}: {data_info['shape']} items")
            else:
                # Handle nested dictionaries (like multi_bulk, annotation, etc.)
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

    def get_n_samples(self, is_paired: bool) -> Dict[str, Union[int, Dict[str, int]]]:
        """Get the number of samples for each data type."""
        if not is_paired:
            return {
                "from": self._get_n_samples(self.from_modality),
                "to": self._get_n_samples(self.to_modality),
            }

        n_samples = {}
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                continue
            n_samples[attr_name] = self._get_n_samples(attr_value)
        n_samples["paired_count"] = max(n_samples.values())
        return n_samples

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
            first_key = next(iter(dataobj.mod.keys()))
            first_anno = dataobj.mod[first_key].obs
            return first_anno.shape[0]
        else:
            raise ValueError(
                f"Unknown data type {type(dataobj)} for dataobj. Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
            )

    def shape(self) -> Dict[str, Dict[str, Any]]:
        """Get the shape of the data for each data type."""
        shapes = {k: {"shape": None} for k in self.__annotations__.keys()}
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, dict):
                    # Use a recursive helper function for dictionaries
                    sub_dict = self._get_shape_from_dict(attr_value)
                    shapes[attr_name] = sub_dict
                elif isinstance(attr_value, list):
                    shapes[attr_name]["shape"] = len(attr_value)
                elif isinstance(attr_value, pd.DataFrame):
                    shapes[attr_name]["shape"] = attr_value.shape
                elif isinstance(attr_value, AnnData):
                    shapes[attr_name]["shape"] = attr_value.shape
                elif isinstance(attr_value, MuData):
                    shapes[attr_name]["shape"] = (attr_value.n_obs, attr_value.n_vars)
                else:
                    raise ValueError(
                        f"Unknown data type {type(attr_value)} for dataobj. Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
                    )
        return shapes

    def _get_shape_from_dict(self, data_dict: Dict) -> Dict[str, Any]:
        """
        Recursively process dictionary to extract shapes of contained data objects.

        Args:
            data_dict: Dictionary containing data objects

        Returns:
            Dictionary with shapes information
        """
        result = {}
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
                result[key] = self._get_shape_from_dict(value)
            else:
                # For unknown types, store a descriptive string instead of raising an error
                # This is more robust as it won't crash the entire method
                result[key] = f"<{type(value).__name__}>"

        return result

    def get_modality_key(self, direction: str) -> Optional[str]:
        """
        Get the first key for a specific direction's modality.

        Args:
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

    def get_modality_data(self, direction: str) -> Any:
        """
        Get the data for a specific direction's modality.

        Args:
            direction: Either 'from' or 'to'

        Returns:
            Data object for the specified modality or None if not available
        """
        key = self.get_modality_key(direction)
        if not key:
            return None

        modality_dict = self.from_modality if direction == "from" else self.to_modality
        return modality_dict.get(key)

    def set_modality_data(self, direction: str, data: Any) -> None:
        """
        Set the data for a specific direction's modality.

        Args:
            direction: Either 'from' or 'to'
            data: The data to set
        """
        key = self.get_modality_key(direction)
        if not key:
            return

        modality_dict = self.from_modality if direction == "from" else self.to_modality
        modality_dict[key] = data

    def process_modality(
        self, direction: str, processor_func: Callable[[Any, str], Any]
    ) -> None:
        """
        Process a specific modality with the given processor function.

        Args:
            direction: Either 'from' or 'to'
            processor_func: Function that takes (data, key) and returns processed data
        """
        key = self.get_modality_key(direction)
        if not key:
            return

        data = self.get_modality_data(direction)
        if data is not None:
            processed_data = processor_func(data, key)
            self.set_modality_data(direction, processed_data)

    def process_multi_bulk(
        self, processor_func: Callable[[pd.DataFrame, str], pd.DataFrame]
    ) -> None:
        """
        Process all DataFrames in multi_bulk with the given processor function.

        Args:
            processor_func: Function that takes (dataframe, key) and returns processed dataframe
        """
        if not self.multi_bulk:
            return

        processed = {}
        for key, df in self.multi_bulk.items():
            processed[key] = processor_func(df, key)
        self.multi_bulk = processed

    def process_multi_sc(self, processor_func: Callable[[MuData], MuData]) -> None:
        """
        Process single-cell data with the given processor function.

        Args:
            processor_func: Function that takes MuData and returns processed MuData
        """
        if not self.multi_sc:
            return

        self.multi_sc = processor_func(self.multi_sc)

    def process_images(
        self, processor_func: Callable[[List[ImgData], str], List[ImgData]]
    ) -> None:
        """
        Process all image data with the given processor function.

        Args:
            processor_func: Function that takes (image_list, key) and returns processed image_list
        """
        if not self.img:
            return

        processed = {}
        for key, img_list in self.img.items():
            processed[key] = processor_func(img_list, key)
        self.img = processed

    def detect_modality_type(self, direction: str) -> Optional[str]:
        """
        Detect the type of a modality ('dataframe', 'image', 'anndata', 'mudata').

        Args:
            direction: Either 'from' or 'to'

        Returns:
            String indicating the data type or None if not detected
        """
        data = self.get_modality_data(direction)
        if data is None:
            return None

        if isinstance(data, pd.DataFrame):
            return "dataframe"
        elif isinstance(data, list) and data and hasattr(data[0], "img"):
            return "image"
        elif isinstance(data, AnnData):
            return "anndata"
        elif isinstance(data, MuData):
            return "mudata"
        return None
