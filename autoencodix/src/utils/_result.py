from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import numpy as np
from autoencodix.src.data import DataSetContainer
from autoencodix.src.base import BaseDataset


@dataclass
class Result:
    """
    A dataclass to store results from the pipeline with predefined keys.
    Attributes
    ----------
    latentspaces : Dict[str, np.ndarray]
        Stores latent space representations for 'train', 'valid', and 'test' splits.
    reconstructions : Dict[str, np.ndarray]
        Stores reconstructed outputs for 'train', 'valid', and 'test' splits.
    mus : Dict[str, np.ndarray]
        Stores mean values of latent distributions for 'train', 'valid', and 'test' splits.
    sigmas : Dict[str, np.ndarray]
        Stores standard deviations of latent distributions for 'train', 'valid', and 'test' splits.
    losses : Dict[str, Dict[str, float]]
        Stores loss values for various metrics (e.g., 'recon') and splits ('train', 'valid', 'test').
    """

    latentspaces: Dict[str, np.ndarray] = field(default_factory=dict)
    reconstructions: Dict[str, np.ndarray] = field(default_factory=dict)
    mus: Dict[str, np.ndarray] = field(default_factory=dict)
    sigmas: Dict[str, np.ndarray] = field(default_factory=dict)
    losses: Dict[str, Dict[str, float]] = field(default_factory=dict)
    preprocessed_data: torch.Tensor = field(default_factory=torch.Tensor)
    model: Optional[torch.nn.Module] = None
    datasets: Optional[DataSetContainer] = field(
        default_factory=lambda: DataSetContainer(
            train=BaseDataset(data=None),
            valid=BaseDataset(data=None),
            test=BaseDataset(data=None),
        )
    )


    def __getitem__(self, key: str) -> Any:
        """
        Retrieve the value associated with a specific key.
        Parameters
        ----------
        key : str
            The name of the attribute to retrieve.
        Returns
        -------
        Any
            The value of the specified attribute.
        Raises
        ------
        KeyError
            If the key is not a valid attribute of the Results class.
        """
        if not hasattr(self, key):
            raise KeyError(
                f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}"
            )
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Assign a value to a specific attribute.
        Parameters
        ----------
        key : str
            The name of the attribute to set.
        value : Any
            The value to assign to the attribute.
        Raises
        ------
        KeyError
            If the key is not a valid attribute of the Results class.
        """
        if not hasattr(self, key):
            raise KeyError(
                f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}"
            )
        setattr(self, key, value)

    def update(self, other: "Result") -> None:
        """
        Update the current Result object with non-empty values from another Result object.

        Parameters
        ----------
        other : Result
            The Result object to update from.
        """
        for field_name, field_type in self.__annotations__.items():
            current_value = getattr(self, field_name)
            other_value = getattr(other, field_name)

            # Handle dictionary fields
            if isinstance(current_value, dict):
                for key, value in other_value.items():
                    if key not in current_value or not current_value.get(key):
                        current_value[key] = value

            # Handle tensor/numpy array fields
            elif isinstance(current_value, (torch.Tensor, np.ndarray)):
                if current_value is None or current_value.size == 0:
                    setattr(self, field_name, other_value)

            # Handle other types of fields
            else:
                if current_value is None or current_value == field_type.__origin__():
                    setattr(self, field_name, other_value)