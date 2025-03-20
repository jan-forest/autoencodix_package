from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import issparse

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._datasetcontainer import DatasetContainer
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


class StackixPreprocessor(BasePreprocessor):
    """
    Preprocessor for Stackix architecture, which handles multiple modalities separately.

    Unlike GeneralPreprocessor which combines all modalities, StackixPreprocessor
    keeps modalities separate for individual VAE training in the Stackix architecture.
    """

    def __init__(self, config: DefaultConfig):
        """Initialize the StackixPreprocessor with the given configuration."""
        super().__init__(config=config)
        self._datapackage: Optional[Dict[str, Any]] = None

    def _extract_primary_data(self, modality_data: Any) -> np.ndarray:
        """Extract primary data matrix and convert to dense array if sparse."""
        primary_data = modality_data.X
        if issparse(primary_data):
            primary_data = primary_data.toarray()
        return primary_data

    def _combine_layers(
        self, modality_name: str, modality_data: Any
    ) -> List[np.ndarray]:
        """Combine specified layers from a modality."""
        layer_list: List[np.ndarray] = []
        selected_layers = self.config.data_config.data_info[
            modality_name
        ].selected_layers

        for layer_name in selected_layers:
            if layer_name == "X":
                primary_data = self._extract_primary_data(modality_data)
                layer_list.append(primary_data)
            elif layer_name in modality_data.layers:
                layer_data = modality_data.layers[layer_name]
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                layer_list.append(layer_data)
            else:
                print(
                    f"Layer '{layer_name}' not found in modality '{modality_name}'. Skipping."
                )

        return layer_list

    def _process_multi_bulk(self, data_dict: Dict[str, Any]) -> StackixDataset:
        """Process multi-bulk data for Stackix architecture."""
        data, split_indices = data_dict["data"], data_dict["indices"]
        multi_bulk = data.multi_bulk

        if multi_bulk is None:
            raise ValueError("No multi_bulk data found")

        # Create dictionaries to hold tensors and IDs for each modality
        modality_tensors = {}
        modality_ids = {}

        # Process each modality separately
        for modality_name, df in multi_bulk.items():
            if df is not None:
                tensor_data = torch.from_numpy(df.values).float()
                modality_tensors[modality_name] = tensor_data
                modality_ids[modality_name] = df.index.tolist()

        return StackixDataset(
            data_dict=modality_tensors, config=self.config, ids_dict=modality_ids
        )

    def _process_multi_sc(self, data_dict: Dict[str, Any]) -> StackixDataset:
        """Process multi-single-cell data for Stackix architecture."""
        data = data_dict["data"]
        multi_sc = data.multi_sc

        if multi_sc is None:
            raise ValueError("No multi_sc data found")

        # Create dictionaries to hold tensors and IDs for each modality
        modality_tensors = {}
        modality_ids = {}

        # Process each modality separately
        for modality_name, modality_data in multi_sc.mod.items():
            # Combine layers for this modality
            combined_layers = self._combine_layers(
                modality_name=modality_name, modality_data=modality_data
            )

            if combined_layers:
                # Concatenate along feature axis if multiple layers
                if len(combined_layers) > 1:
                    combined_data = np.concatenate(combined_layers, axis=1)
                else:
                    combined_data = combined_layers[0]

                tensor_data = torch.from_numpy(combined_data).float()
                modality_tensors[modality_name] = tensor_data
                modality_ids[modality_name] = modality_data.obs_names.tolist()

        return StackixDataset(
            data_dict=modality_tensors, config=self.config, ids_dict=modality_ids
        )

    def _process_data_package(self, data_dict: Dict[str, Any]) -> BaseDataset:
        """Process a data package based on its type for Stackix architecture."""
        data = data_dict["data"]

        for key in data.__annotations__.keys():
            attr_val = getattr(data, key)
            if key == "multi_bulk" and attr_val is not None:
                return self._process_multi_bulk(data_dict)
            elif key == "multi_sc" and attr_val is not None:
                return self._process_multi_sc(data_dict)

        raise ValueError("No supported data type found in the split")

    def preprocess(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> DatasetContainer:
        """
        Execute preprocessing steps for Stackix architecture.

        Unlike GeneralPreprocessor, this keeps modalities separate for individual VAE training.

        Returns:
            DatasetContainer with StackixDataset for each split.
        """
        self._datapackage = self._general_preprocess(raw_user_data)
        if self._datapackage is None:
            raise TypeError("Datapackage cannot be None")

        self._dataset_container = DatasetContainer()

        for split in ["train", "valid", "test"]:
            if self._datapackage[split]["data"] is None:
                self._dataset_container[split] = None
                continue
            if split in self._datapackage:
                dataset = self._process_data_package(data_dict=self._datapackage[split])
                self._dataset_container[split] = dataset

        return self._dataset_container
