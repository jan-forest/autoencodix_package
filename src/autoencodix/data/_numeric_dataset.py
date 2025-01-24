from typing import Optional
import torch
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils.default_config import DefaultConfig

# internal check done
# write tests: done


class NumericDataset(BaseDataset):
    """
    A custom PyTorch dataset that handles tensors.
    """

    def __init__(
        self,
        data: torch.Tensor,
        config: DefaultConfig,
        ids: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the dataset

        Parameters:
        -----------
        data : torch.Tensor
            Input features
        labels : Optional[torch.Tensor]
            Optional labels for supervised learning
        """
        super().__init__(data=data, ids=ids, config=config)
        if self.config is None:
            raise ValueError("config cannot be None")
        dtype = self._map_float_precision_to_dtype(self.config.float_precision)

        # Convert or clone data to the specified dtype
        self.data = self._to_tensor(data, dtype)

    @staticmethod
    def _to_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Ensure a tensor is of the specified dtype.

        Parameters:
        -----------
        tensor : torch.Tensor
            Input tensor
        dtype : torch.dtype
            Desired data type

        Returns:
        --------
        torch.Tensor
            Cloned tensor with the specified dtype
        """
        # Clone and detach to ensure no gradient history is retained
        return tensor.clone().detach().to(dtype)

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

    def __len__(self) -> int:
        """
        Returns the number of samples (rows) in the dataset
        """
        return self.data.shape[0]

    def get_input_dim(self) -> int:
        """
        Get the input dimension of the dataset

        Returns:
        --------
        int
            The input dimension
        """
        return self.data.shape[1]
