from typing import Any, Dict, Optional, Tuple, Union
import torch
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils.default_config import DefaultConfig


class StackixDataset(NumericDataset):
    """
    Dataset for handling multiple modalities in Stackix models.

    This dataset holds individual BaseDataset objects for multiple data modalities
    and provides a consistent interface for accessing them during training.
    It's designed to work specifically with StackixTrainer.

    Attributes
    ----------
    dataset_dict : Dict[str, BaseDataset]
        Dictionary mapping modality names to dataset objects
    modality_keys : List[str]
        List of modality names
    data : torch.Tensor
        First modality's tensor (needed for BaseDataset compatibility)
    sample_ids : List[Any]
        First modality's sample IDs (needed for BaseDataset compatibility)
    feature_ids : List[Any]
        First modality's feature IDs (needed for BaseDataset compatibility)
    """

    def __init__(
        self,
        dataset_dict: Dict[str, BaseDataset],
        config: DefaultConfig,
    ):
        """
        Initialize a StackixDataset instance.

        Parameters
        ----------
        dataset_dict : Dict[str, BaseDataset]
            Dictionary mapping modality names to dataset objects
        config : DefaultConfig
            Configuration object

        Raises
        ------
        ValueError
            If the datasets dictionary is empty or if modality datasets have different numbers of samples
        """
        if not dataset_dict:
            raise ValueError("dataset_dict cannot be empty")

        # Use first modality for base class initialization
        first_modality_key = next(iter(dataset_dict.keys()))
        first_modality = dataset_dict[first_modality_key]
        data = torch.cat(
            [v.data for k, v in dataset_dict.items() if hasattr(v, "data")], dim=1
        )
        super().__init__(
            data=data,
            sample_ids=first_modality.sample_ids,
            config=config,
            split_indices=first_modality.split_indices,
            metadata=first_modality.metadata,
            feature_ids=[
                v.feature_ids
                for v in dataset_dict.values()
                if hasattr(v, "feature_ids")
            ],
        )

        self.dataset_dict = dataset_dict
        self.modality_keys = list(dataset_dict.keys())

        # Ensure all datasets have the same number of samples
        sample_counts = [len(dataset) for dataset in dataset_dict.values()]
        if not all(count == sample_counts[0] for count in sample_counts):
            raise ValueError(
                "All modality datasets must have the same number of samples"
            )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(next(iter(self.dataset_dict.values())))

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, Any], Dict[str, Tuple[torch.Tensor, Any]]]:
        """
        Get a single sample and its label from the dataset.

        Returns the data from the first modality to maintain compatibility
        with the BaseDataset interface, while still supporting multi-modality
        access through dataset_dict.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve

        Returns
        -------
        Dict[str, Tuple[torch.Tensor, Any]]
            Dictionary of (data tensor, label) pairs for each modality

        """
        return {
            k: self.dataset_dict[k].__getitem__(index)
            for k in self.dataset_dict.keys()
        }

    def get_modality_item(self, modality: str, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Get a sample for a specific modality.

        Parameters
        ----------
        modality : str
            The modality name to retrieve data from
        index : int
            Index of the sample to retrieve

        Returns
        -------
        Tuple[torch.Tensor, Any]
            Tuple of (data tensor, label) for the specified modality and sample index

        Raises
        ------
        KeyError
            If the requested modality doesn't exist in the dataset
        """
        if modality not in self.dataset_dict:
            raise KeyError(f"Modality '{modality}' not found in dataset")

        return self.dataset_dict[modality][index]
