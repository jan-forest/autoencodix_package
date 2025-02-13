import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from autoencodix.utils.default_config import DefaultConfig


@dataclass
class ImgData:
    img: np.ndarray
    sample_id: str
    annotation: pd.DataFrame


@dataclass
class MyData:
    multi_sc: AnnData | None = None
    multi_bulk: Dict[str, pd.DataFrame] = None
    annotation: pd.DataFrame | None = None
    img: List[ImgData] | None = None


class BulkDataReader:
    @staticmethod
    def read_data(
        config: DefaultConfig,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Read all data according to config and return a MyData object in a single pass.
        """
        common_samples: Set[str] | None = None
        bulk_dfs: Dict[str, pd.DataFrame] = {}
        annotation_df = pd.DataFrame()
        has_annotation = False

        for key, info in config.data_config.data_info.items():
            print(info.data_type)
            if info.data_type == "IMG":
                continue

            file_path = os.path.join(info.file_path)

            try:
                if file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith((".csv", ".txt", ".tsv")):
                    df = pd.read_csv(file_path, sep=info.sep, index_col=0)
                else:
                    print(f"Unsupported file type for {file_path}")
                    continue

                if info.data_type == "NUMERIC" and not info.is_single_cell:
                    current_samples = set(df.index)
                    if common_samples is None:
                        common_samples = current_samples
                    else:
                        common_samples &= current_samples

                    bulk_dfs[key] = df

                elif info.data_type == "ANNOTATION":
                    has_annotation = True
                    annotation_df = df

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if common_samples:
            print("common samples")
            for key, df in bulk_dfs.items():
                bulk_dfs[key] = df.reindex(list(common_samples))
            if has_annotation:
                annotation = annotation_df.reindex(list(common_samples))
            else:
                annotation_df.index = list(common_samples)
                annotation = annotation_df
        else:
            annotation = annotation_df
        return bulk_dfs, annotation
