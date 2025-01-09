import itertools
from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split  # type: ignore

from autoencodix.utils.default_config import DefaultConfig


# internal check done
# write tests: TODO
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
    -----------
    _config : DefaultConfig
        Configuration object containing split ratios

    _custom_splits : Optional[Dict[str, np.ndarray]]
    _test_ratio : float
    _valid_ratio : float

    Methods:
    --------
    _validate_ratios(test_ratio: float, valid_ratio: float) -> None
        Validate that the splitting ratios meet required constraints.
    _validate_split_sizes(n_samples: int) -> None
        Validate that each non-empty split will have sufficient samples.
    _validate_custom_splits(splits: Dict[str, np.ndarray]) -> None
        Validate custom splits for correctness.
    split(X: Union[torch.Tensor, np.ndarray]) -> Dict[str, np.ndarray]
        Split data into train, validation, and test sets. Returns indices for each split.
        and None for empty splits.

    """

    def __init__(
        self,
        config: DefaultConfig,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize DataSplitter with configuration and optional custom splits.

        Parameters:
            config (DefaultConfig): Configuration object containing split ratios
            custom_splits (Optional[Dict[str, np.ndarray]]): Pre-defined split indices

        Raises:
            ValueError: If ratios violate constraints or custom splits are malformed
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

        Parameters:
            n_samples (int): Total number of samples in dataset
        Returns:
            None
        Raises:
            ValueError: If any non-empty split would have too few samples

        """

        # Calculate expected sizes
        n_train = int(n_samples * (1 - self._test_ratio - self._valid_ratio))
        n_valid = int(n_samples * self._valid_ratio) if self._valid_ratio > 0 else 0
        n_test = int(n_samples * self._test_ratio) if self._test_ratio > 0 else 0

        if n_train < self._min_samples:
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

        Parameters:
            splits (Dict[str, np.ndarray]): Custom split indices
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
        X: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Parameters:
            X: Input data to split (torch.Tensor or np.ndarray)

        Returns:
            Dict[str, np.ndarray]: Dictionary containing indices for each split,
                                 with empty arrays for splits with ratio=0

        Raises:
            ValueError: If resulting splits would violate size constraints
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        if self._custom_splits:
            return self._custom_splits

        n_samples = len(X)
        self._validate_split_sizes(n_samples)
        indices = np.arange(n_samples)

        if self._test_ratio == 0 and self._valid_ratio == 0:
            return {
                "train": indices,
                "valid": np.array([], dtype=int),
                "test": np.array([], dtype=int),
            }

        if self._test_ratio == 0:
            train_indices, valid_indices = train_test_split(
                indices,
                test_ratio=self._valid_ratio,
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
                test_ratio=self._test_ratio,
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
