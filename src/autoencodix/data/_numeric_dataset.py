from __future__ import annotations
import torch

import scipy as sp
from scipy.sparse import issparse
import numpy as np
from typing import Optional, List, Union, Any, Dict, Tuple, no_type_check
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.base._base_dataset import BaseDataset, DataSetTypes


class TensorAwareDataset(BaseDataset):
    """
    Handles dtype mapping and tensor conversion logic.
    """

    @staticmethod
    def _to_tensor(
        data: Union[torch.Tensor, np.ndarray, Any], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert data to tensor with specified dtype.

        Args:
            data: Input data to convert
            dtype: Desired data type

        Returns:
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

        Args:
            float_precision: Precision type (e.g., 'bf16-mixed', '16-mixed')

        Returns:
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
            DataFrame representation of the dataset
        """
        if isinstance(self.data, torch.Tensor):
            return pd.DataFrame(
                self.data.numpy(), columns=self.feature_ids, index=self.sample_ids
            )
        elif issparse(self.data):
            return pd.DataFrame(
                self.data.toarray(), columns=self.feature_ids, index=self.sample_ids
            )
        elif isinstance(self.data, list) and all(
            isinstance(item, torch.Tensor) for item in self.data
        ):
            # Handle image modality
            # Get the list of tensors
            tensor_list = self.data

            # Flatten each tensor and collect as rows
            rows = [
                (
                    t.flatten().cpu().numpy()
                    if isinstance(t, torch.Tensor)
                    else t.flatten()
                )
                for t in tensor_list
            ]

            df_flat = pd.DataFrame(
                rows,
                index=self.sample_ids,
                columns=["Pixel_" + str(i) for i in range(len(rows[0]))],
            )
            return df_flat
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
    """A custom PyTorch dataset that handles tensors.


    Attributes:
        data: The input features as a torch.Tensor.
        config: Configuration object containing settings for data processing.
        sample_ids: Optional list of sample identifiers.
        feature_ids: Optional list of feature identifiers.
        metadata: Optional pandas DataFrame containing metadata.
        split_indices: Optional numpy array for data splitting.
        mytype: Enum indicating the dataset type (set to DataSetTypes.NUM).
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, sp.sparse.spmatrix],
        config: DefaultConfig,
        sample_ids: Union[None, List[Any]] = None,
        feature_ids: Union[None, List[Any]] = None,
        metadata: Optional[Union[pd.Series, pd.DataFrame]] = None,
        split_indices: Optional[Union[Dict[str, Any], List[Any], np.ndarray]] = None,
    ):
        """
        Initialize the dataset

        Args:
            data: Input features
            config: Configuration object
            sample_ids: Optional sample identifiers
            feature_ids: Optional feature identifiers
            metadata: Optional metadata
            split_indices: Optional split indices
            Optional split indices
        """
        super().__init__(
            data=data, sample_ids=sample_ids, config=config, feature_ids=feature_ids
        )

        if self.config is None:
            raise ValueError("config cannot be None")

        # Convert data to appropriate dtype once during initialization
        self.target_dtype = self._get_target_dtype()
        # keep data sparce if it is a scipy sparse matrix to be memory
        # efficient for large single cell data, convert at batch level to dense tensor
        if isinstance(self.data, (np.ndarray, torch.Tensor)):
            self.data = self._to_tensor(data, self.target_dtype)

        self.metadata = metadata
        self.split_indices = split_indices
        self.mytype = DataSetTypes.NUM

    @no_type_check
    def __getitem__(self, index: int) -> Union[
        Tuple[
            Union[torch.Tensor, int],
            Union[torch.Tensor, "ImgData"],  # ty: ignore  # noqa: F821
            Any,
        ],
        Dict[str, Tuple[Any, torch.Tensor, Any]],
    ]:
        """Retrieves a single sample and its corresponding label.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple containing the index, the data sample and its label, or a dictionary
            mapping keys to such tuples in case we have multiple uncombined data at this step.
        """

        row = self.data[index]  # idx: int, slice, or list
        if self.sample_ids is not None:
            label = self.sample_ids[index]
        else:
            label = index
        if issparse(row):
            # print("calling to array")

            # print(f"Size of data sparse: {asizeof.asizeof(row)}")
            row = torch.tensor(row.toarray(), dtype=self.target_dtype).squeeze(0)

            # print(f"Size of data dense: {asizeof.asizeof(row)}")

        return index, row, label

    def __len__(self) -> int:
        """Returns the number of samples (rows) in the dataset"""
        return self.data.shape[0]
