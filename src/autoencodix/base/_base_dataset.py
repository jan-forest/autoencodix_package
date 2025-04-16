import abc
from typing import Any, Tuple, Optional, Union, List, Dict

from torch.utils.data import Dataset
import torch

# from autoencodix.utils.default_config import DefaultConfig


# internal check done
# write tests: done
class BaseDataset(abc.ABC, Dataset):
    """
    Abstract base class for PyTorch datasets.

    Subclasses must implement the __len__ and __getitem__ methods.

    Attributes
    ----------
    data : Any
        The dataset (can be any type, like a NumPy array, list, or Pandas DataFrame).
    """

    def __init__(
        self,
        data: torch.Tensor,
        config: Optional[Any] = None,
        sample_ids: Union[None, List[Any]] = None,
        feature_ids: Union[None, List[Any]] = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data : Any
            The data to be used by the dataset.
        """
        self.sample_ids = sample_ids
        self.data = data
        self.config = config
        self.feature_ids = feature_ids

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
        if self.sample_ids is not None:
            label = self.sample_ids[index]
        else:
            label = index
        return self.data[index], label

    def get_input_dim(self) -> Union[int, Dict[str, int]]:
        """
        Get the input dimension of the dataset.

        Returns
        -------
        int
            The input dimension of the dataset, of the feature space.
        """
        return self.data.shape[1]
