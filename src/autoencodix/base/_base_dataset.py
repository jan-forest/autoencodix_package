import abc
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from autoencodix.data._imgdataclass import ImgData


class DataSetTypes(str, Enum):
    NUM = "NUM"
    IMG = "IMG"


class BaseDataset(abc.ABC, Dataset):
    """Interface to guide implementation fo custom PyTorch datasets.

    Attributes:
        data: The dataset content (can be a torch.Tensor or other data structure).
        config: Optional configuration object.
        sample_ids: Optional list of identifiers for each sample.
        feature_ids: Optional list of identifiers for each feature.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, List[ImgData]],
        config: Optional[Any] = None,
        sample_ids: Optional[List[Any]] = None,
        feature_ids: Optional[List[Any]] = None,
    ):
        """Initializes the dataset.

        Args:
            data: The data to be used by the dataset.
            config: Optional configuration parameters.
            sample_ids: Optional identifiers for each sample.
            feature_ids: Optional identifiers for each feature.
        """
        self.data = data
        self.config = config
        self.sample_ids = sample_ids
        self.feature_ids = feature_ids
        self.mytype: Enum # set in subclasses

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, Any], Dict[str, Tuple[Any, torch.Tensor, Any]]]:
        """Retrieves a single sample and its corresponding label.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple containing the data sample and its label, or a dictionary
            mapping keys to such tuples.
        """
        if self.sample_ids is not None:
            label = self.sample_ids[index]
        else:
            label = index
        return index, self.data[index], label

    def get_input_dim(self) -> Union[int, Tuple[int, ...]]:
        """Gets the input dimension of the dataset (n_features)

        Returns:
            The input dimension of the dataset's feature space.
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.shape[1]
        elif isinstance(self.data, list) and isinstance(self.data[0], ImgData):
            return self.data[0].img.shape[0]
        else:
            raise ValueError("Unsupported data type for input dimension retrieval.")
