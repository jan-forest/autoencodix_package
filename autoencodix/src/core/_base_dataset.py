from typing import Any, Tuple

from torch.utils.data import Dataset

# TODO decide wheter to use this at all and how (abstract class or not)
class BaseDataset(Dataset):
    """
    Abstract base class for PyTorch datasets.

    Subclasses must implement the __len__ and __getitem__ methods.

    Attributes
    ----------
    data : Any
        The dataset (can be any type, like a NumPy array, list, or Pandas DataFrame).
    """

    def __init__(self, data: Any):
        """
        Initialize the dataset.

        Parameters
        ----------
        data : Any
            The data to be used by the dataset.
        """
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
        return self.data[index]
