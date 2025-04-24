from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils.default_config import DefaultConfig


class StackixDataset(BaseDataset):
    """
    Dataset for handling multiple modalities in Stackix models.

    This dataset holds individual BaseDataset objects for multiple data modalities
    and provides a consistent interface for accessing them during training.
    It's designed to work specifically with StackixTrainer.

    Attributes
    ----------
    datasets_dict : Dict[str, BaseDataset]
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
        datasets_dict: Dict[str, BaseDataset],
        config: DefaultConfig,
    ):
        """
        Initialize a StackixDataset instance.

        Parameters
        ----------
        datasets_dict : Dict[str, BaseDataset]
            Dictionary mapping modality names to dataset objects
        config : DefaultConfig
            Configuration object

        Raises
        ------
        ValueError
            If the datasets dictionary is empty or if modality datasets have different numbers of samples
        """
        if not datasets_dict:
            raise ValueError("datasets_dict cannot be empty")

        # Use first modality for base class initialization
        first_modality_key = next(iter(datasets_dict.keys()))
        first_modality = datasets_dict[first_modality_key]

        super().__init__(
            data=first_modality.data, 
            sample_ids=first_modality.sample_ids if hasattr(first_modality, "sample_ids") else None, 
            config=config,
            feature_ids=[v.feature_ids for v in datasets_dict.values() if hasattr(v, "feature_ids")],
        )

        self.datasets_dict = datasets_dict
        self.modality_keys = list(datasets_dict.keys())
        
        # Ensure all datasets have the same number of samples
        sample_counts = [len(dataset) for dataset in datasets_dict.values()]
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
        return len(next(iter(self.datasets_dict.values())))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Get a single sample and its label from the dataset.

        Returns the data from the first modality to maintain compatibility
        with the BaseDataset interface, while still supporting multi-modality
        access through datasets_dict.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve

        Returns
        -------
        Tuple[torch.Tensor, Any]
            Tuple of (data tensor, label) for the sample
        """
        # Return the first modality's data for compatibility with BaseTrainer
        first_key = self.modality_keys[0]
        first_dataset = self.datasets_dict[first_key]
        
        return first_dataset[index]

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
        if modality not in self.datasets_dict:
            raise KeyError(f"Modality '{modality}' not found in dataset")

        return self.datasets_dict[modality][index]

    def get_input_dim(
        self, modality: Optional[str] = None
    ) -> Union[int, Dict[str, int]]:
        """
        Get the input dimension(s) of the dataset.

        Parameters
        ----------
        modality : Optional[str]
            If provided, returns the dimension for the specified modality only

        Returns
        -------
        Union[int, Dict[str, int]]
            Input dimension of the specified modality or dictionary of dimensions for all modalities

        Raises
        ------
        KeyError
            If the requested modality doesn't exist in the dataset
        """
        if modality is not None:
            if modality not in self.datasets_dict:
                raise KeyError(f"Modality '{modality}' not found in dataset")
            return self.datasets_dict[modality].get_input_dim()

        return {key: dataset.get_input_dim() for key, dataset in self.datasets_dict.items()}

    def get_feature_ids(
        self, modality: Optional[str] = None
    ) -> Union[List[Any], Dict[str, List[Any]]]:
        """
        Get feature identifiers for one or all modalities.

        Parameters
        ----------
        modality : Optional[str]
            If provided, returns feature IDs for the specified modality only

        Returns
        -------
        Union[List[Any], Dict[str, List[Any]]]
            Feature IDs for the specified modality or dictionary of IDs for all modalities

        Raises
        ------
        KeyError
            If the requested modality doesn't exist or lacks feature IDs
        """
        feature_ids_dict = {}
        
        for key, dataset in self.datasets_dict.items():
            if hasattr(dataset, "feature_ids") and dataset.feature_ids is not None:
                feature_ids_dict[key] = dataset.feature_ids
        
        if modality is not None:
            if modality not in feature_ids_dict:
                raise KeyError(f"Feature IDs for modality '{modality}' not found")
            return feature_ids_dict[modality]

        return feature_ids_dict
    
    def get_sample_ids(
        self, modality: Optional[str] = None
    ) -> Union[List[Any], Dict[str, List[Any]]]:
        """
        Get sample identifiers for one or all modalities.

        Parameters
        ----------
        modality : Optional[str]
            If provided, returns sample IDs for the specified modality only

        Returns
        -------
        Union[List[Any], Dict[str, List[Any]]]
            Sample IDs for the specified modality or dictionary of IDs for all modalities

        Raises
        ------
        KeyError
            If the requested modality doesn't exist or lacks sample IDs
        """
        sample_ids_dict = {}
        
        for key, dataset in self.datasets_dict.items():
            if hasattr(dataset, "sample_ids") and dataset.sample_ids is not None:
                sample_ids_dict[key] = dataset.sample_ids
        
        if modality is not None:
            if modality not in sample_ids_dict:
                raise KeyError(f"Sample IDs for modality '{modality}' not found")
            return sample_ids_dict[modality]

        return sample_ids_dict

    @classmethod
    def from_datasets(
        cls, datasets: Dict[str, BaseDataset], config: DefaultConfig
    ) -> "StackixDataset":
        """
        Create a StackixDataset from multiple individual datasets.

        Parameters
        ----------
        datasets : Dict[str, BaseDataset]
            Dictionary mapping modality names to individual datasets
        config : DefaultConfig
            Configuration object

        Returns
        -------
        StackixDataset
            A new StackixDataset combining data from all input datasets
        """
        return cls(datasets_dict=datasets, config=config)