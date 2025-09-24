import abc
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from autoencodix.data._imgdataclass import ImgData


class DataSetTypes(str, Enum):
    NUM = "NUM"
    IMG = "IMG"


class BaseDataset(abc.ABC, Dataset):
    """Interface to guide implementation for custom PyTorch datasets.

    Attributes:
        data: The dataset content (can be a torch.Tensor or other data structure).
        config: Optional configuration object.
        sample_ids: Optional list of identifiers for each sample.
        feature_ids: Optional list of identifiers for each feature.
        mytype: Enum indicating the dataset type (should be set in subclasses).
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
            mytype: Enum indicating the dataset type (should be set in subclasses).
        """
        self.data = data
        self.raw_data = data  # for child class ImageDataset
        self.config = config
        self.sample_ids = sample_ids
        self.feature_ids = feature_ids
        self.mytype: Enum  # Should be set in subclasses to indicate the dataset type (e.g., DataSetTypes.NUM or DataSetTypes.IMG)

        self.metadata: Optional[Union[pd.Series, pd.DataFrame]] = (None,)
        self.datasets: Dict[str, BaseDataset] = {}  # for xmodalix child

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Union[
        Tuple[Union[torch.Tensor, int], Union[torch.Tensor, ImgData], Any],
        Dict[str, Tuple[Any, torch.Tensor, Any]],
    ]:
        """Retrieves a single sample and its corresponding label.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple containing the index, the data sample and its label, or a dictionary
            mapping keys to such tuples in case we have multiple uncombined data at this step.
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
        elif isinstance(self.data, list):
            if len(self.data) == 0:
                raise ValueError(
                    "Dataset is ImgData, and the list of ImgData is empty, cannot determine input dimension."
                )
            if isinstance(self.data[0], ImgData):
                return self.data[0].img.shape[0]
            else:
                raise ValueError(
                    "List data is not of type ImgData, cannot determine input dimension."
                )
        else:
            raise ValueError("Unsupported data type for input dimension retrieval.")

    def _to_df(self, modality: Optional[str] = None) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
            DataFrame representation of the dataset
        """
        if isinstance(self.data, torch.Tensor):
            return pd.DataFrame(
                self.data.numpy(), columns=self.feature_ids, index=self.sample_ids
            )
        else:
            raise TypeError(
                "Data is not a torch.Tensor and cannot be converted to DataFrame."
            )
