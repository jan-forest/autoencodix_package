import itertools
import pandas as pd
from typing import Dict, Optional, Any, Set, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data.datapackage import DataPackage
from collections import defaultdict


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
    """Performs pairing-aware data splitting across multiple modalities.

    Handles any number of data modalities and automatically identifies
    fully paired and partially paired samples. Each sample is assigned
    to exactly one split (train, valid, or test). If a sample appears in
    multiple modalities, it is guaranteed to appear in the same split
    across all of them.

    Each modality can have a corresponding annotation file. Samples in modalities
    that are not in their corresponding annotation file are dropped.

    Attributes:
        data_package: The input data package containing modalities and annotations.
        config: Configuration object with split ratios and random seed.
        annotation_ids_per_modality: Mapping from modality keys to their valid sample IDs.
        modalities: Mapping of modality keys to their data objects.
        membership_groups: Mapping of modality combinations to the set of
            sample IDs belonging to each combination (e.g., RNA+Protein pairs).
    """

    def __init__(self, data_package, config):
        """Initializes the splitter and computes modality membership groups.

        Args:
            data_package: The full data package to split. Must implement
                `_get_sample_ids` and iterable access yielding (key, object) pairs.
            config: Split configuration object defining ratios and random seed.

        Raises:
            TypeError: If `data_package` is not a valid DataPackage instance.
        """
        if not hasattr(data_package, "_get_sample_ids"):
            raise TypeError("data_package must be an instance of DataPackage.")

        self.datapackage = data_package
        self.config = config
        self.membership_groups: Dict[Tuple[str, ...], Set[str]] = (
            self._compute_membership_groups()
        )

    def _compute_membership_groups(self) -> Dict[Tuple[str, ...], Set[str]]:
        """Groups samples by the set of modalities in which they appear.

        Returns:
            A mapping from modality combinations (tuples of modality keys)
            to the set of sample IDs belonging to each combination.

        Examples:
            This mapping could look like:
            {("multi_bulk.rna", "multi_bulk.cna"): {"id1", id3", ...},
              ()"multi_bulk.rna, "img.img"): {"id2", "id4 ,...}
            }
        """
        sample_to_modalities: Dict[str, Set[str]] = defaultdict(set)
        for modality_key, obj in self.datapackage:
            if "annotation" in modality_key:
                continue
            ids: List[str] = self.datapackage._get_sample_ids(obj)

            for sid in ids:
                sample_to_modalities[sid].add(modality_key)

        # Group samples by identical modality membership
        groups: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
        for sid, mods in sample_to_modalities.items():
            groups[tuple(sorted(mods))].add(sid)
        return groups

    def _split_group(self, ids: Sequence[str]) -> Dict[str, np.ndarray]:
        """Splits a homogeneous group of sample IDs into train, valid, and test subsets.

        Args:
            ids: Collection of sample IDs belonging to the same modality group.

        Returns:
            A mapping with keys ``train``, ``valid``, and ``test``, where each value
            is an array of sample IDs assigned to that split.
        """
        ids = list(ids)
        rng = np.random.default_rng(self.config.global_seed)
        ids = rng.permutation(ids)
        n = len(ids)
        n_train = int(n * self.config.train_ratio)
        n_valid = int(n * self.config.valid_ratio)
        return {
            "train": np.array(ids[:n_train], dtype=object),
            "valid": np.array(ids[n_train : n_train + n_valid], dtype=object),
            "test": np.array(ids[n_train + n_valid :], dtype=object),
        }

    def split(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """Performs the complete pairing-aware split across modalities and annotations.

        Ensures that:
        - Each sample appears in exactly one split across all modalities it belongs to.
        - Fully paired samples are synchronized across modalities.
        - Each annotation table is split consistently with its corresponding modality.

        Returns:
            Nested mapping of ``{parent_key -> {child_key -> {split_name -> np.ndarray(indices)}}}``,
            suitable for use with ``DataPackageSplitter``. Includes splits for both
            modalities and their corresponding annotation files.
        """
        # Sort groups by descending number of modalities (most-paired first)
        # This ensures fully-paired samples are assigned before partially-paired ones
        sorted_groups = sorted(
            self.membership_groups.items(), key=lambda kv: -len(kv[0])
        )
        assigned_ids: Set[str] = set()

        # Initialize split storage for each modality
        per_modality_splits: Dict[str, Dict[str, Set[str]]] = {
            mod: {"train": set(), "valid": set(), "test": set()}
            for mod, _ in self.datapackage
        }

        # Assign each membership group to splits
        for mods_tuple, sids in sorted_groups:
            sids_to_assign = [sid for sid in sids if sid not in assigned_ids]
            if not sids_to_assign:
                continue

            group_splits = self._split_group(sids_to_assign)
            for split_name, split_ids in group_splits.items():
                for mod in mods_tuple:
                    per_modality_splits[mod][split_name].update(split_ids)
                assigned_ids.update(split_ids)

        final_indices: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for full_key, data_obj in self.datapackage:
            parent_key, child_key = full_key.split(".")
            if parent_key == "annotation":
                continue  # Handle annotations separately below
            original_ids = self.datapackage._get_sample_ids(data_obj)

            id_to_pos = {sid: i for i, sid in enumerate(original_ids)}
            final_indices.setdefault(parent_key, {})[child_key] = {}

            for split_name in ["train", "valid", "test"]:
                split_ids_for_mod = per_modality_splits.get(full_key, {}).get(
                    split_name, set()
                )
                final_indices[parent_key][child_key][split_name] = np.array(
                    sorted([id_to_pos[sid] for sid in split_ids_for_mod]), dtype=int
                )

        if self.datapackage.annotation:
            for anno_key, anno_df in self.datapackage.annotation.items():
                print(f"anno key: {anno_key}")
                anno_id_to_pos = {sid: i for i, sid in enumerate(anno_df.index)}
                final_indices.setdefault("annotation", {})[anno_key] = {}
                if anno_key == "paired":
                    # For each split, take the union of all sample IDs across modalities
                    for split_name in ["train", "valid", "test"]:
                        split_ids_union = set().union(
                            *[
                                splits[split_name]
                                for splits in per_modality_splits.values()
                            ]
                        )
                        anno_indices = sorted(
                            [
                                anno_id_to_pos[sid]
                                for sid in split_ids_union
                                if sid in anno_id_to_pos
                            ]
                        )
                        final_indices["annotation"][anno_key][split_name] = np.array(
                            anno_indices, dtype=int
                        )

                    continue  # Skip normal annotation handling for "paired"


                for split_name in ["train", "valid", "test"]:
                    for mod_name in per_modality_splits:
                        parent_key, child_key = mod_name.split(".")
                        # in per_modality_splits are no annotation ids
                        # thats what we do here, so we need to skip annotation
                        # to not get empty arrays in final_indices
                        if parent_key == "annotation":
                            continue
                        if child_key == anno_key:
                            # Take split IDs from the corresponding modality only
                            split_ids = per_modality_splits.get(mod_name, {}).get(
                                split_name, set()
                            )
                            anno_indices = sorted(
                                [
                                    anno_id_to_pos[sid]
                                    for sid in split_ids
                                    if sid in anno_id_to_pos
                                ]
                            )
                            final_indices["annotation"][anno_key][split_name] = (
                                np.array(anno_indices, dtype=int)
                            )

        return final_indices
