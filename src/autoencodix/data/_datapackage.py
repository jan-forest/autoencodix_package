from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from anndata import AnnData

from autoencodix.data._imgdataclass import ImgData


@dataclass
class DataPackage:
    """
    A class to represent a data package containing multiple types of data.
    """

    multi_sc: Optional[AnnData] = None
    multi_bulk: Optional[Dict[str, pd.DataFrame]] = None
    annotation: Optional[pd.DataFrame] = None
    img: Optional[List[ImgData]] = None

    def get_n_samples(
        self, is_paired: bool
    ) -> Dict[str, int | Dict[str, int]]:
        """Get the number of samples for each data type."""

        n_samples = {}

        for attr_name in self.__annotations__.keys():
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, pd.DataFrame):
                    n_samples[attr_name] = attr_value.shape[0]
                elif isinstance(attr_value, AnnData):
                    n_samples[attr_name] = attr_value.obs.shape[0]
                elif isinstance(attr_value, list):
                    n_samples[attr_name] = len(attr_value)
                elif isinstance(attr_value, dict):
                    n_samples[attr_name] = {
                        key: df.shape[0] for key, df in attr_value.items()
                    }

        if is_paired:
            # For paired data, return the first available count or raise an error
            for key in self.__annotations__.keys():
                if key in n_samples:
                    if isinstance(n_samples[key], dict):  # Handle multi_bulk case
                        return {"paired_count": next(iter(n_samples[key].values()))}
                    else:
                        return {"paired_count": n_samples[key]}
            raise ValueError("No data found for paired data package.")
        else:
            return n_samples  # Always return a dictionary
