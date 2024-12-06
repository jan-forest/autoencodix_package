from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# TODO check implemetnation and test (see old repo)
# TODO add tests
# TODO add default class docstring
class NumericDataset(Dataset):
    """
    A custom PyTorch dataset that can handle numpy arrays or tensors
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor, None]] = None,
    ):
        """
        Initialize the dataset

        Parameters:
        -----------
        data : Union[np.ndarray, torch.Tensor]
            Input features
        labels : Optional[Union[np.ndarray, torch.Tensor]]
            Optional labels for supervised learning
        """
        # Convert to tensor if numpy array
        self.data = torch.tensor(data, dtype=torch.float32)

        # Handle labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single item from the dataset

        Parameters:
        -----------
        idx : int
            Index of the item to retrieve

        Returns:
        --------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Single data item, or data with labels if labels exist
        """
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]
