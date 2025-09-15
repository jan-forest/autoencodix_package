import itertools
from typing import Dict, Optional, Any, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from autoencodix.configs.default_config import DefaultConfig

import pandas as pd
from functools import reduce
from autoencodix.data.datapackage import DataPackage


# internal check done
# write tests: done
class DataSplitter:
    """
    Splits data into train, validation, and test sets. And validates the splits.

    Also allows for custom splits to be provided.
    Here we allow empty splits (e.g. test_ratio=0), this might raise an error later
    in the pipeline, when this split is expected to be non-empty. However, this allows
    are more flexible usage of the pipeline (e.g. when the user only wants to run the fit step).

    Constraints:
    1. Split ratios must sum to 1
    2. Each non-empty split must have at least min_samples_per_split samples
    3. Any split ratio must be <= 1.0
    4. Custom splits must contain 'train', 'valid', and 'test' keys and non-overlapping indices

    Attributes:
        _config: Configuration object containing split ratios

        _custom_splits: Optional pre-defined split indices
        _test_ratio
        _valid_ratio

    """

    def __init__(
        self,
        config: DefaultConfig,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize DataSplitter with configuration and optional custom splits.

        Args:
            config (DefaultConfig): Configuration object containing split ratios
            custom_splits (Optional[Dict[str, np.ndarray]]): Pre-defined split indices
        """
        self._config = config
        self._test_ratio = self._config.test_ratio
        self._valid_ratio = self._config.valid_ratio
        self._train_ratio = self._config.train_ratio
        self._min_samples = self._config.min_samples_per_split
        self._custom_splits = custom_splits

        self._validate_ratios()
        if self._custom_splits:
            self._validate_custom_splits(self._custom_splits)

    def _validate_ratios(self) -> None:
        """
        Validate that the splitting ratios meet required constraints.
        Returns:
            None
        Raises:
            ValueError: If ratios violate constraints

        """
        if not 0 <= self._test_ratio <= 1:
            raise ValueError(
                f"Test ratio must be between 0 and 1, got {self._test_ratio}"
            )
        if not 0 <= self._valid_ratio <= 1:
            raise ValueError(
                f"Validation ratio must be between 0 and 1, got {self._valid_ratio}"
            )
        if not 0 <= self._train_ratio <= 1:
            raise ValueError(
                f"Train ratio must be between 0 and 1, got {self._train_ratio}"
            )

        if np.sum([self._test_ratio, self._valid_ratio, self._train_ratio]) != 1:
            raise ValueError("Split ratios must sum to 1")

    def _validate_split_sizes(self, n_samples: int) -> None:
        """
        Validate that each non-empty split will have sufficient samples.

        Args:
            n_samples: Total number of samples in dataset
        Returns:
            None
        Raises:
            ValueError: If any non-empty split would have too few samples

        """

        # Calculate expected sizes
        n_train = int(n_samples * (1 - self._test_ratio - self._valid_ratio))
        n_valid = int(n_samples * self._valid_ratio) if self._valid_ratio > 0 else 0
        n_test = int(n_samples * self._test_ratio) if self._test_ratio > 0 else 0

        if self._train_ratio > 0 and n_train < self._min_samples:
            raise ValueError(
                f"Training set would have {n_train} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

        if self._valid_ratio > 0 and n_valid < self._min_samples:
            raise ValueError(
                f"Validation set would have {n_valid} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

        if self._test_ratio > 0 and n_test < self._min_samples:
            raise ValueError(
                f"Test set would have {n_test} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

    def _validate_custom_splits(self, splits: Dict[str, np.ndarray]) -> None:
        """
        Validate custom splits for correctness.

        Args:
            splits: Custom split indices
        Returns:
            None
        Raises:
            ValueError: If custom splits violate constraints

        """
        required_keys = {"train", "valid", "test"}
        if not all(key in splits for key in required_keys):
            raise ValueError(
                f"Custom splits must contain all of: {required_keys} \ Got: {splits.keys()} \ if you want to pass empty splits, pass an empty array"
            )

        # check for index out of bounds
        if len(splits["train"]) < self._min_samples:
            raise ValueError(
                f"Custom training split has {len(splits['train'])} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

        # For non-empty validation and test splits, check minimum size
        if len(splits["valid"]) > 0 and len(splits["valid"]) < self._min_samples:
            raise ValueError(
                f"Custom validation split has {len(splits['valid'])} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

        if len(splits["test"]) > 0 and len(splits["test"]) < self._min_samples:
            raise ValueError(
                f"Custom test split has {len(splits['test'])} samples, "
                f"which is less than minimum required ({self._min_samples})"
            )

        # Check for overlap between splits
        for k1, k2 in itertools.combinations(required_keys, 2):
            intersection = set(splits[k1]) & set(splits[k2])
            if intersection:
                raise ValueError(
                    f"Overlapping indices found between splits '{k1}' and '{k2}': {intersection}"
                )

    def split(
        self,
        n_samples: int,
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            n_samples: Total number of samples in the dataset

        Returns:
            Dictionary containing indices for each split, with empty arrays for splits with ratio=0

        Raises:
            ValueError: If resulting splits would violate size constraints
        """
        self._validate_split_sizes(n_samples)
        indices = np.arange(n_samples)

        if self._custom_splits:
            max_index = n_samples - 1
            for split in self._custom_splits.values():
                if len(split) > 0:
                    if np.max(split) > max_index:
                        raise AssertionError(
                            f"Custom split indices must be within range [0, {max_index}]"
                        )
                    elif np.min(split) < 0:
                        raise AssertionError(
                            f"Custom split indices must be within range [0, {max_index}]"
                        )
            return self._custom_splits

        # all three 0 case already handled in _validate_ratios (sum to 1)
        if self._test_ratio == 0 and self._valid_ratio == 0:
            return {
                "train": indices,
                "valid": np.array([], dtype=int),
                "test": np.array([], dtype=int),
            }
        if self._train_ratio == 0 and self._valid_ratio == 0:
            return {
                "train": np.array([], dtype=int),
                "valid": np.array([], dtype=int),
                "test": indices,
            }
        if self._train_ratio == 0 and self._test_ratio == 0:
            return {
                "train": np.array([], dtype=int),
                "valid": indices,
                "test": np.array([], dtype=int),
            }

        if self._train_ratio == 0:
            valid_indices, test_indices = train_test_split(
                indices,
                test_size=self._test_ratio,
                random_state=self._config.global_seed,
            )
            return {
                "train": np.array([], dtype=int),
                "valid": valid_indices,
                "test": test_indices,
            }

        if self._test_ratio == 0:
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=self._valid_ratio,
                random_state=self._config.global_seed,
            )
            return {
                "train": train_indices,
                "valid": valid_indices,
                "test": np.array([], dtype=int),
            }

        if self._valid_ratio == 0:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=self._test_ratio,
                random_state=self._config.global_seed,
            )
            return {
                "train": train_indices,
                "valid": np.array([], dtype=int),
                "test": test_indices,
            }

        # Normal case: split into all three sets
        train_valid_indices, test_indices = train_test_split(
            indices, test_size=self._test_ratio, random_state=self._config.global_seed
        )

        train_indices, valid_indices = train_test_split(
            train_valid_indices,
            test_size=self._valid_ratio / (1 - self._test_ratio),
            random_state=self._config.global_seed,
        )

        return {"train": train_indices, "valid": valid_indices, "test": test_indices}


class PairedUnpairedSplitter:
    """
    Performs a pairing-aware split of a DataPackage dynamically, handling any
    data type that provides sample IDs.

    Attributes:
        data_package: The DataPackage to be split.
        config: Configuration object containing split ratios.
        all_modalities: Discovered modalities in the DataPackage.
        paired_ids: Sample IDs present in all modalities.
        unpaired_ids: Sample IDs present in at least one, but not all, modalities.
    """

    def __init__(self, data_package: DataPackage, config: DefaultConfig):
        if not isinstance(data_package, DataPackage):
            raise TypeError("data_package must be an instance of DataPackage.")
        self.data_package = data_package
        self.config = config

        # Dynamically discover all modalities and their data objects
        self.all_modalities: Dict[str, Any] = self._get_all_modalities()

        self.paired_ids: pd.Index
        self.unpaired_ids: pd.Index
        self.paired_ids, self.unpaired_ids = self._identify_sample_groups()

    def _get_all_modalities(self) -> Dict[str, Any]:
        """Dynamically discovers all data-containing attributes from the DataPackage.
        Returns:
            A dictionary mapping modality names to their corresponding data objects.
        """
        modalities = {}
        for key, value in self.data_package:
            # The iterator yields keys like "multi_bulk.RNA".
            # We exclude the 'annotation' parent key from being a modality to be split,
            # as it is the reference, but we need its samples for intersection.
            parent_key = key.split(".")[0]
            if parent_key != "annotation":
                modalities[key] = value
        return modalities

    def _identify_sample_groups(self) -> Tuple[pd.Index, pd.Index]:
        """Identifies fully-paired and unpaired sample IDs across all modalities.
        Returns:
            A tuple containing:
            - paired_ids: Sample IDs present in all modalities.
            - unpaired_ids: Sample IDs present in at least one, but not all, modalities.

        """
        if not self.all_modalities:
            return pd.Index([]), pd.Index([])

        # Use the helper to get ID sets for all data modalities AND the annotation
        all_data_id_sets = [
            set(self.data_package._get_sample_ids(mod))
            for mod in self.all_modalities.values()
        ]

        # Also get the annotation IDs to ensure everything is grounded
        if self.data_package.annotation:
            # Assuming one primary annotation dictionary for now
            main_anno_key = next(iter(self.data_package.annotation.keys()))
            all_data_id_sets.append(
                set(self.data_package.annotation[main_anno_key].index)
            )

        # Find the intersection (paired) and union (all)
        paired_ids_set = reduce(lambda s1, s2: s1.intersection(s2), all_data_id_sets)
        all_ids_set = reduce(lambda s1, s2: s1.union(s2), all_data_id_sets)
        unpaired_ids_set = all_ids_set - paired_ids_set

        print(
            f"Identified {len(paired_ids_set)} fully paired samples across all modalities."
        )
        print(
            f"Identified {len(unpaired_ids_set)} samples present in at least one, but not all, modalities."
        )

        return pd.Index(sorted(list(paired_ids_set))), pd.Index(
            sorted(list(unpaired_ids_set))
        )

    def split(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Executes the two-stage split and returns the final integer indices
        in the format expected by DataPackageSplitter.
        Returns:
            A dictionary mapping each modality to its train/valid/test integer indices.
        """
        splitter = DataSplitter(config=self.config)
        empty_split = {
            "train": np.array([], dtype=int),
            "valid": np.array([], dtype=int),
            "test": np.array([], dtype=int),
        }

        # --- Step 1: Conditionally Split the Paired and Unpaired groups ---

        # Only split the paired group if it's not empty
        if len(self.paired_ids) > 0:
            paired_splits_indices = splitter.split(n_samples=len(self.paired_ids))
        else:
            paired_splits_indices = empty_split

        # Only split the unpaired group if it's not empty
        if len(self.unpaired_ids) > 0:
            unpaired_splits_indices = splitter.split(n_samples=len(self.unpaired_ids))
        else:
            unpaired_splits_indices = empty_split

        # --- Step 2: Combine the SAMPLE IDs for each split ---
        # This logic remains the same and works correctly with empty splits.
        final_split_ids: Dict[str, Set[str]] = {
            "train": set(),
            "valid": set(),
            "test": set(),
        }

        for split_name in ["train", "valid", "test"]:
            # Get IDs from the paired group
            p_indices = paired_splits_indices[split_name]
            final_split_ids[split_name].update(self.paired_ids[p_indices])

            # Get IDs from the unpaired group
            u_indices = unpaired_splits_indices[split_name]
            final_split_ids[split_name].update(self.unpaired_ids[u_indices])

        # --- Step 3: Generate the final integer index map for each original DataFrame ---
        # This logic also remains the same.
        final_indices: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for (
            full_key,
            data_obj,
        ) in self.data_package:  # Iterate through the original full package
            parent_key, child_key = full_key.split(".")

            original_modality_ids = self.data_package._get_sample_ids(data_obj)
            if not original_modality_ids:
                continue

            id_to_pos = {id_val: i for i, id_val in enumerate(original_modality_ids)}

            final_indices.setdefault(parent_key, {})[child_key] = {}

            for split_name in ["train", "valid", "test"]:
                split_ids_in_df = final_split_ids[split_name].intersection(
                    original_modality_ids
                )

                int_indices = [id_to_pos[id_val] for id_val in split_ids_in_df]
                final_indices[parent_key][child_key][split_name] = np.array(
                    sorted(int_indices)
                )

        return final_indices
