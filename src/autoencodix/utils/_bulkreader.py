import os
import warnings
from typing import Dict, Set, Tuple, Optional, Union

import pandas as pd

from autoencodix.configs.default_config import DefaultConfig


class BulkDataReader:
    """Reads bulk data from files based on configuration.

    Supports both paired and unpaired data reading strategies.

    Attributes:
        config: Configuration object
    """

    def __init__(self, config: DefaultConfig):
        """Initialize the BulkDataReader with a configuration.

        Args:
            config: Configuration object containing data paths and specifications.
        """
        self.config = config

    def read_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Read all data according to the configuration.

        Returns:
            A tuple containing (bulk_dataframes, annotation_dataframes)
        """
        if self.config.requires_paired or self.config.requires_paired is None:
            return self.read_paired_data()
        else:
            return self.read_unpaired_data()

    def read_paired_data(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Reads numeric paired data

        Returns:
            Tuple containing two Dicts:
                 1. with name of the data as key and pandas DataFrame as value
                 2. with  str 'paired' as key and a common annotaion/metadata as DataFrame
        """
        common_samples: Optional[Set[str]] = None
        bulk_dfs: Dict[str, pd.DataFrame] = {}
        annotation_df = pd.DataFrame()
        has_annotation = False

        # First pass: read all data files and track common samples
        for key, info in self.config.data_config.data_info.items():
            if info.data_type == "IMG":
                continue

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
        """Read data without enforcing sample alignment across modalities.

        Returns:
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

                if hasattr(info, "extra_anno_file") and info.extra_anno_file:
                    extra_anno_file = os.path.join(info.extra_anno_file)
                    extra_anno_df = self._read_tabular_data(
                        file_path=extra_anno_file, sep=info.sep
                    )
                    if extra_anno_df is not None:
                        annotations[key] = extra_anno_df

            elif info.data_type == "ANNOTATION":
                annotations[key] = df

        bulk_dfs, annotations = self._validate_and_filter_unpaired(
            bulk_dfs, annotations
        )

        return bulk_dfs, annotations

    def _validate_and_filter_unpaired(
        self,
        bulk_dfs: Dict[str, pd.DataFrame],
        annotations: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Validates that all samples in bulk data have a corresponding annotation.

        If a single global annotation file is provided, it creates a perfectly
        matched annotation dataframe for each bulk dataframe.

        Warns and drops samples that do not have a corresponding annotation.

        Args:
            bulk_dfs: Dictionary of bulk data modalities and their dataframes.
            annotations: Dictionary of annotation dataframes, possibly one global one.

        Returns:
            A tuple of two dictionaries:
            1. The filtered bulk dataframes.
            2. The new, synchronized annotation dataframes, with keys matching the bulk dataframes.
        """
        if not annotations:
            warnings.warn(
                "No annotation files were provided. Cannot validate sample annotations."
            )
            return bulk_dfs, {}

        # If annotations have keys that match bulk_dfs, we assume they are already paired.
        # This logic focuses on the case where one annotation file is meant for all bulk files.
        # A simple heuristic: if there is one annotation file and its key is not in bulk_dfs.
        annotation_keys = set(annotations.keys())
        bulk_keys = set(bulk_dfs.keys())

        # Check for the global annotation case
        if len(annotation_keys) == 1 and not annotation_keys.intersection(bulk_keys):
            global_annotation_key = list(annotation_keys)[0]
            global_annotation_df = annotations[global_annotation_key]

            filtered_bulk_dfs = {}
            synchronized_annotations = {}

            for key, data_df in bulk_dfs.items():
                data_samples = data_df.index
                annotation_samples = global_annotation_df.index

                # Find the intersection of valid sample IDs
                valid_ids = data_samples.intersection(annotation_samples)

                # Check for and warn about dropped samples
                if len(valid_ids) < len(data_samples):
                    missing_ids = sorted(list(set(data_samples) - set(valid_ids)))
                    warnings.warn(
                        f"For data modality '{key}', {len(missing_ids)} sample(s) "
                        f"were found without a corresponding annotation and will be dropped: {missing_ids}"
                    )

                # Filter both the data and the annotation to the valid IDs
                filtered_bulk_dfs[key] = data_df.loc[valid_ids]
                synchronized_annotations[key] = global_annotation_df.loc[valid_ids]

            return filtered_bulk_dfs, synchronized_annotations
        else:
            # Handle the case where annotations are already meant to be paired by key
            # (Or a more complex case we are not handling yet)
            warnings.warn(
                "Proceeding without global annotation synchronization. Assuming annotations are pre-aligned by key."
            )
            return bulk_dfs, annotations

    def _read_tabular_data(
        self, file_path: str, sep: Union[str, None] = None
    ) -> pd.DataFrame:
        """Read tabular data from a file with error handling.

        Args:
        file_path: Path to the data file.
        sep: Separator character for CSV/TSV files.

        Returns:
            The loaded DataFrame.
        """
        try:
            if file_path.endswith(".parquet"):
                print(f"reading parquet: {file_path}")
                return pd.read_parquet(file_path)
            elif file_path.endswith((".csv", ".txt", ".tsv")):
                return pd.read_csv(file_path, sep=sep, index_col=0)
            else:
                raise ValueError(
                    f"Unsupported file type for {file_path}. Supported formats: .parquet, .csv, .txt, .tsv"
                )
        except Exception as e:
            raise e
