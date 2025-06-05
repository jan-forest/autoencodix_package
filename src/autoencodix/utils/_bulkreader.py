import os
from typing import Dict, Set, Tuple, Optional, Union

import pandas as pd

from autoencodix.utils.default_config import DefaultConfig


class BulkDataReader:
    """
    Class for reading bulk data from files based on configuration.
    Supports both paired and unpaired data reading strategies.
    """

    def __init__(self, config: DefaultConfig):
        """
        Initialize the BulkDataReader with a configuration.

        Parameters
        ----------
        config : DefaultConfig
            Configuration object containing data paths and specifications.
        """
        self.config = config

    def read_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Read all data according to the configuration.

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            A tuple containing (bulk_dataframes, annotation_dataframes)
        """
        if self.config.requires_paired or self.config.requires_paired is None:
            return self.read_paired_data()
        else:
            return self.read_unpaired_data()

    def read_paired_data(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Read data where samples are paired across modalities.
        Finds common samples across all data sources.

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            A tuple containing (bulk_dataframes, annotation_dataframes)
        """
        common_samples: Optional[Set[str]] = None
        bulk_dfs: Dict[str, pd.DataFrame] = {}
        annotation_df = pd.DataFrame()
        has_annotation = False

        # First pass: read all data files and track common samples
        for key, info in self.config.data_config.data_info.items():
            if info.data_type == "IMG":
                continue  # Skip image data in this reader

            file_path = os.path.join(info.file_path)
            df = self._read_tabular_data(file_path, info.sep or "\t")

            if df is None:
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

        # Second pass: filter to common samples
        if common_samples:
            common_samples_list = list(common_samples)

            # Reindex bulk dataframes to common samples
            for key in bulk_dfs:
                bulk_dfs[key] = bulk_dfs[key].reindex(common_samples_list)

            # Handle annotation dataframe
            if has_annotation:
                annotation = annotation_df.reindex(common_samples_list)
            else:
                # Create empty annotation with common sample indices
                annotation_df = pd.DataFrame(index=common_samples_list)
                annotation = annotation_df
        else:
            print("Warning: No common samples found across datasets")
            annotation = annotation_df

        return bulk_dfs, {"paired": annotation}

    def read_unpaired_data(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Read data without enforcing sample alignment across modalities.

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            A tuple containing (bulk_dataframes, annotation_dataframes)
        """
        bulk_dfs: Dict[str, pd.DataFrame] = {}
        annotations: Dict[str, pd.DataFrame] = {}

        for key, info in self.config.data_config.data_info.items():
            if info.data_type == "IMG" or info.is_single_cell:
                continue  # Skip image and single-cell data

            # Read main data file
            file_path = os.path.join(info.file_path)
            df = self._read_tabular_data(file_path=file_path, sep=info.sep)

            if df is None:
                continue

            if info.data_type == "NUMERIC":
                bulk_dfs[key] = df

                # Handle extra annotation file if specified
                if hasattr(info, "extra_anno_file") and info.extra_anno_file:
                    extra_anno_file = os.path.join(info.extra_anno_file)
                    extra_anno_df = self._read_tabular_data(
                        file_path=extra_anno_file, sep=info.sep
                    )
                    if extra_anno_df is not None:
                        annotations[key] = extra_anno_df

            elif info.data_type == "ANNOTATION":
                annotations[key] = df

        return bulk_dfs, annotations

    def _read_tabular_data(
        self, file_path: str, sep: Union[str, None] = None
    ) -> pd.DataFrame:
        """
        Read tabular data from a file with error handling.

        Parameters
        ----------
        file_path : str
            Path to the data file.
        sep : str
            Separator character for CSV/TSV files.

        Returns
        -------
        Optional[pd.DataFrame]
            The loaded DataFrame or None if loading failed.
        """
        try:
            if file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif file_path.endswith((".csv", ".txt", ".tsv")):
                return pd.read_csv(file_path, sep=sep, index_col=0)
            else:
                raise ValueError(
                    f"Unsupported file type for {file_path}. Supported formats: .parquet, .csv, .txt, .tsv"
                )
        except Exception as e:
            raise e
