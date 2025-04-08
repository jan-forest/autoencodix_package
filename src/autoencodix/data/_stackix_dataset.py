from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils.default_config import DefaultConfig


class StackixDataset(BaseDataset):
    """
    Dataset for handling multiple modalities in Stackix models.

    This dataset holds tensors for multiple data modalities and provides
    a consistent interface for accessing them during training.
    """

    def __init__(
        self,
        data_dict: Dict[str, torch.Tensor],
        config: DefaultConfig,
        ids_dict: Optional[Dict[str, List[Any]]] = None,
        feature_ids_dict: Optional[Dict[str, List[Any]]] = None,
    ):
        # Use first modality for base class initialization
        first_modality = next(iter(data_dict.values()))
        first_ids = None
        if ids_dict:
            first_key = next(iter(ids_dict.keys()))
            first_ids = ids_dict[first_key]

        super().__init__(data=first_modality, ids=first_ids, config=config)

        self.data_dict = data_dict
        self.modality_keys = list(data_dict.keys())
        self.ids_dict = ids_dict or {}
        self.feature_ids_dict = feature_ids_dict or {}

        # Ensure all tensors have same first dimension (number of samples)
        sample_counts = [tensor.shape[0] for tensor in data_dict.values()]
        if not all(count == sample_counts[0] for count in sample_counts):
            raise ValueError(
                "All modality tensors must have the same number of samples"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return next(iter(self.data_dict.values())).shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """
        Returns a dictionary mapping each modality to its (data, label) tuple.
        """
        result = {}
        for key in self.modality_keys:
            tensor = self.data_dict[key][index]
            label = index
            if key in self.ids_dict and self.ids_dict[key]:
                label = self.ids_dict[key][index]
            result[key] = (tensor, label)
        return result

    def get_input_dim(
        self, modality: Optional[str] = None
    ) -> Union[int, Dict[str, int]]:
        """
        Get the input dimension(s) of the dataset.
        """
        if modality is not None:
            if modality not in self.data_dict:
                raise KeyError(f"Modality '{modality}' not found in dataset")
            return self.data_dict[modality].shape[1]

        return {key: tensor.shape[1] for key, tensor in self.data_dict.items()}
