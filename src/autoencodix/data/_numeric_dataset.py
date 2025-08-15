import torch
import numpy as np
from typing import Optional, List, Union, Any
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.base._base_dataset import BaseDataset, DataSetTypes


class TensorAwareDataset(BaseDataset):
    """
    Base class that handles dtype mapping and tensor conversion logic.
    """

    @staticmethod
    def _to_tensor(
        data: Union[torch.Tensor, np.ndarray, Any], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert data to tensor with specified dtype.

        Parameters:
        -----------
        data : Union[torch.Tensor, np.ndarray, Any]
            Input data to convert
        dtype : torch.dtype
            Desired data type

        Returns:
        --------
        torch.Tensor
            Tensor with the specified dtype
        """
        if isinstance(data, torch.Tensor):
            return data.clone().detach().to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)

    @staticmethod
    def _map_float_precision_to_dtype(float_precision: str) -> torch.dtype:
        """
        Map fabric precision types to torch tensor dtypes.

        Parameters:
        -----------
        float_precision : str
            Precision type (e.g., 'bf16-mixed', '16-mixed')

        Returns:
        --------
        torch.dtype
            Corresponding torch dtype
        """
        precision_mapping = {
            "transformer-engine": torch.float32,  # Default for transformer-engine
            "transformer-engine-float16": torch.float16,
            "16-true": torch.float16,
            "16-mixed": torch.float16,
            "bf16-true": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            "32-true": torch.float32,
            "64-true": torch.float64,
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.float16,
            "bf16": torch.bfloat16,
        }
        # Default to torch.float32 if the precision is not recognized
        return precision_mapping.get(float_precision, torch.float32)

    def _to_df(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
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

    def _get_target_dtype(self) -> torch.dtype:
        """Get the target dtype based on config, with MPS compatibility check."""
        target_dtype = self._map_float_precision_to_dtype(self.config.float_precision)

        # MPS doesn't support float64, so fallback to float32
        if target_dtype == torch.float64 and self.config.device == "mps":
            print("Warning: MPS doesn't support float64, using float32 instead")
            target_dtype = torch.float32

        return target_dtype


class NumericDataset(TensorAwareDataset):
    """
    A custom PyTorch dataset that handles tensors.
    """

    def __init__(
        self,
        data: torch.Tensor,
        config: DefaultConfig,
        sample_ids: Union[None, List[Any]] = None,
        feature_ids: Union[None, List[Any]] = None,
        metadata: Optional[pd.DataFrame] = None,
        split_indices: Optional[np.ndarray] = None,
    ):
        """
        Initialize the dataset

        Parameters:
        -----------
        data : torch.Tensor
            Input features
        config : DefaultConfig
            Configuration object
        sample_ids : Union[None, List[Any]]
            Optional sample identifiers
        feature_ids : Union[None, List[Any]]
            Optional feature identifiers
        metadata : Optional[pd.DataFrame]
            Optional metadata
        split_indices : Optional[np.ndarray]
            Optional split indices
        """
        super().__init__(
            data=data, sample_ids=sample_ids, config=config, feature_ids=feature_ids
        )

        if self.config is None:
            raise ValueError("config cannot be None")

        # Convert data to appropriate dtype once during initialization
        target_dtype = self._get_target_dtype()
        self.data = self._to_tensor(data, target_dtype)

        self.metadata = metadata
        self.split_indices = split_indices
        self.mytype = DataSetTypes.NUM

    def __len__(self) -> int:
        """Returns the number of samples (rows) in the dataset"""
        return self.data.shape[0]
