import torch
import numpy as np
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._numeric_dataset import TensorAwareDataset
from typing import List, Tuple
from autoencodix.data._imgdataclass import ImgData
from autoencodix.base._base_dataset import DataSetTypes


class ImageDataset(TensorAwareDataset):
    """
    A custom PyTorch dataset that handles image data with proper dtype conversion.
    """

    def __init__(
        self,
        data: List[ImgData],
        config: DefaultConfig,
        split_indices: np.ndarray = None,
        metadata: pd.DataFrame = None,
    ):
        """
        Initialize the dataset

        Parameters:
        -----------
        data : List[ImgData]
            List of image data objects
        config : DefaultConfig
            Configuration object
        """
        self.raw_data = data # image data before conversion to keep original infos
        self.config = config
        self.mytype = DataSetTypes.IMG

        if self.config is None:
            raise ValueError("config cannot be None")

        # Convert all images to tensors with proper dtype once during initialization
        target_dtype = self._get_target_dtype()
        self.data = self._convert_all_images_to_tensors(target_dtype)

        # Extract sample_ids for consistency
        self.sample_ids = [img_data.sample_id for img_data in data]

        self.split_indices = split_indices
        self.feature_ids = None
        self.metadata = metadata

    def _convert_all_images_to_tensors(self, dtype: torch.dtype) -> List[torch.Tensor]:
        """
        Convert all images to tensors with specified dtype during initialization.

        Parameters:
        -----------
        dtype : torch.dtype
            Target dtype for the tensors

        Returns:
        --------
        List[torch.Tensor]
            List of converted image tensors
        """
        print(f"Converting {len(self.raw_data)} images to {dtype} tensors...")
        converted_data = []

        for img_data in self.raw_data:
            tensor = self._to_tensor(img_data.img, dtype)
            converted_data.append(tensor)

        return converted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at index - data is already converted to proper dtype"""
        return idx, self.data[idx], self.sample_ids[idx]

    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Gets the input dimension of the dataset's feature space.

        Returns:
        --------
        Tuple[int, ...]
            The input dimension of the dataset's feature space
        """
        return self.data[0].shape  # All images should have the same shape
