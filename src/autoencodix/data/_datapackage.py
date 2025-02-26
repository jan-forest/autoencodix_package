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
