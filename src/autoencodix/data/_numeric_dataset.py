from typing import Optional, Union

import numpy as np
import torch
from src.autoencodix.base._base_dataset import BaseDataset


# TODO check implemetnation and test (see old repo)
# TODO add tests
# TODO add default class docstring
class NumericDataset(BaseDataset):
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
        self.data = torch.tensor(data, dtype=torch.float32)

        # Handle labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def get_input_dim(self) -> int:
        """
        Get the input dimension of the dataset

        Returns:
        --------
        int
            The input dimension
        """
        return self.data.shape[1]
