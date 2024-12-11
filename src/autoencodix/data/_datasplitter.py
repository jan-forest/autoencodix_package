from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split  # type: ignore

# TODO check implemetnation and test
# TODO work with config object
# TODO allow custom splits
# TODO add tests
# TODO add default class docstring
class DataSplitter:
    """
    Receives the processed data as np.ndarray and splits it into train, validation, and test sets.
    Uses the default train, test, and valid ratios from the default config. If the user overwrites the default config,
    these ratios will be used. The user can also provide custom indices for each split.
    """

    def __init__(
        self,
        test_size: float = 0.2,  # TODO get from config
        valid_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize DataSplitter

        Parameters:
        -----------
        test_size : float, optional
            Proportion of data to use for testing (default is 0.2).
        valid_size : float, optional
            Proportion of data to use for validation (default is 0.1).
        random_state : int, optional
            Random seed for reproducibility (default is 42).
        """
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_state = random_state

    def split(
        self,
        X: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Create train, validation, and test indices.

        Parameters:
        -----------
        X : torch.Tensor
            Input features
        custom_splits : Optional[Dict[str, Union[List[int], np.ndarray]]]

        Returns:
        --------
        Dict[str, np.ndarray]
            Indices for each split: 'train', 'valid', 'test'
        """
        X = X.numpy()
        train_valid_indices, test_indices = train_test_split(
            np.arange(len(X)), test_size=self.test_size, random_state=self.random_state
        )

        train_indices, valid_indices = train_test_split(
            train_valid_indices,
            test_size=self.valid_size / (1 - self.test_size),
            random_state=self.random_state,
        )

        return {"train": train_indices, "valid": valid_indices, "test": test_indices}
