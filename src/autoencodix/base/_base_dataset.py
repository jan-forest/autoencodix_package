import abc
from typing import Any, Tuple, Optional

from torch.utils.data import Dataset
import torch


# internal check done
# write tests: TODO
class BaseDataset(abc.ABC, Dataset):
    """
    Abstract base class for PyTorch datasets.

    Subclasses must implement the __len__ and __getitem__ methods.

    Attributes
    ----------
    data : Any
        The dataset (can be any type, like a NumPy array, list, or Pandas DataFrame).
    """

    def __init__(self, data: torch.Tensor, ids: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.

        Parameters
        ----------
        data : Any
            The data to be used by the dataset.
        """
        if ids is not None:
            self.ids = ids
        self.data = data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Retrieve a single sample and its corresponding label.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[Any, Any]
            The sample and its label.
        """
        if self.ids:
            label = self.ids[index]
        else:
            label = index
        return self.data[index], label

    def get_input_dim(self) -> int:
        """
        Get the input dimension of the dataset.

        Returns
        -------
        int
            The input dimension of the dataset, of the feature space.
        """
        return self.data.shape[1]
