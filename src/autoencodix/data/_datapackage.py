from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd
from anndata import AnnData
from mudata import MuData

from autoencodix.data._imgdataclass import ImgData


@dataclass
class DataPackage:
    """
    A class to represent a data package containing multiple types of data.
    """

    multi_sc: Optional[MuData] = None
    multi_bulk: Optional[Dict[str, pd.DataFrame]] = None
    annotation: Optional[Dict[str, pd.DataFrame]] = None
    img: Optional[Dict[str, List[ImgData]]] = None

    from_modality: Optional[Union[List[ImgData] | MuData | pd.DataFrame]] = field(
        default=None, repr=False
    )
    to_modality: Optional[Union[List[ImgData] | MuData | pd.DataFrame]] = field(
        default=None, repr=False
    )

    # to keep unambigous remove on translation relevant and duplicate data
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
                self.multi_bulk is None,
                self.annotation is None,
                self.img is None,
            ]
        )

    def get_n_samples(self, is_paired: bool) -> Dict[str, int | Dict[str, int]]:
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
        self, dataobj: Union[MuData | pd.DataFrame | List[ImgData] | AnnData]
    ) -> int:
        """Get the number of samples for a specific attribute."""
        if dataobj is not None:
            if isinstance(dataobj, pd.DataFrame):
                return dataobj.shape[0]
            elif isinstance(dataobj, dict):
                first_value = next(iter(dataobj.values()))
                return self._get_n_samples(first_value)

            elif isinstance(dataobj, list):
                return len(dataobj)
            elif isinstance(dataobj, AnnData):
                return dataobj.obs.shape[0]
            elif isinstance(dataobj, MuData):
                first_key = next(iter(dataobj.mod.keys()))
                first_anno = dataobj.mod[first_key].obs
                return first_anno.shape[0]

            else:
                raise ValueError(
                    f"Unknown data type {type(dataobj)} for dataobj, Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
                )

    def shape(self) -> Dict[str, Dict[str, int]]:
        """Get the shape of the data for each data type."""

        shapes = {k: {"shape": None} for k in self.__annotations__.keys()}
        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, dict):
                    sub_dict = {}
                    for key, value in attr_value.items():
                        if isinstance(value, pd.DataFrame):
                            sub_dict[key] = value.shape
                        elif isinstance(value, list):
                            sub_dict[key] = len(value)
                        elif isinstance(value, AnnData):
                            sub_dict[key] = value.shape
                        elif isinstance(value, MuData):
                            print(f"key: {key} is MuData")
                        else:
                            raise ValueError(
                                f"Unknown data type {type(value)} for dataobj, Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
                            )
                    shapes[attr_name] = sub_dict
                elif isinstance(attr_value, list):
                    shapes[attr_name]["shape"] = len(attr_value)
                elif isinstance(attr_value, pd.DataFrame):
                    shapes[attr_name]["shape"] = attr_value.shape
                elif isinstance(attr_value, AnnData):
                    shapes[attr_name]["shape"] = attr_value.shape
                elif isinstance(attr_value, MuData):
                    shapes[attr_name]["shape"] = attr_value.shape
                else:
                    raise ValueError(
                        f"Unknown data type {type(attr_value)} for dataobj, Probably you've implemented a new attribute in the DataPackage class or changed the data type of an existing attribute."
                    )
        return shapes
