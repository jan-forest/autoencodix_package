import abc
from typing import Any, Tuple

from torch.utils.data import Dataset


# TODO decide wheter to use this at all and how (abstract class or not)
#
class BaseDataset(abc.ABC, Dataset):
    """
    Abstract base class for PyTorch datasets.

    Subclasses must implement the __len__ and __getitem__ methods.

    Attributes
    ----------
    data : Any
        The dataset (can be any type, like a NumPy array, list, or Pandas DataFrame).
    """

    def __init__(self, data: Any): #TOD specify data type
        """
        Initialize the dataset.

        Parameters
        ----------
        data : Any
            The data to be used by the dataset.
        """
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]: #TODO specify return type
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
        return self.data[index] # TODO return in form of id, data[index], data[index] (for autoencoders, target is the same as input)

    def get_input_dim(self) -> int:
        """
        Get the input dimension of the dataset.

        Returns
        -------
        int
            The input dimension of the dataset, of the feature space.
        """
        return self.data.shape[1]
