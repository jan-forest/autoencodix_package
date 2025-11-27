import torch
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._numeric_dataset import TensorAwareDataset
from typing import Any, List, Tuple, Optional, Dict
from autoencodix.data._imgdataclass import ImgData
from autoencodix.base._base_dataset import DataSetTypes


class ImageDataset(TensorAwareDataset):
    """
    A custom PyTorch dataset that handles image data with proper dtype conversion.


    Attributes:
        raw_data: List of ImgData objects containing original image data and metadata.
        config: Configuration object for dataset settings.
        mytype: Enum indicating the dataset type (set to DataSetTypes.IMG).
        data: List of image tensors converted to the appropriate dtype.
        sample_ids: List of identifiers for each sample.
        split_indices: Optional numpy array of indices for splitting the dataset.
        feature_ids: Optional list of identifiers for each feature (set to None for images).
        metadata: Optional pandas DataFrame containing additional metadata.
    """

    def __init__(
        self,
        data: List[ImgData],
        config: DefaultConfig,
        split_indices: Optional[Dict[str, Any]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the dataset
        Args:
            data: List of image data objects
            config: Configuration object
        """
        self.raw_data = data  # image data before conversion to keep original infos
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

        Args:
            dtype: Target dtype for the tensors

        Returns:
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
        """Get item at index - data is already converted to proper dtype
        Returns:
            Tuple of (index, image tensor, sample_id)
        """
        return idx, self.data[idx], self.sample_ids[idx]

    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Gets the input dimension of the dataset's feature space.

        Returns:
            The input dimension of the dataset's feature space
        """
        return self.data[0].shape  # All images should have the same shape
