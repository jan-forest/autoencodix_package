from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from autoencodix.data import DataSetContainer

from ._traindynamics import TrainingDynamics


@dataclass
class Result:
    """
    A dataclass to store results from the pipeline with predefined keys.
    Attributes
    ----------
    latentspaces : TrainingDynamics
        Stores latent space representations for 'train', 'valid', and 'test' splits.
    reconstructions : TrainingDynamics
        Stores reconstructed outputs for 'train', 'valid', and 'test' splits.
    mus : TrainingDynamics
        Stores mean values of latent distributions for 'train', 'valid', and 'test' splits.
    sigmas : TrainingDynamics
        Stores standard deviations of latent distributions for 'train', 'valid', and 'test' splits.
    losses : TrainingDynamics
        Stores loss values for various metrics (e.g., 'recon') and splits ('train', 'valid', 'test').
    """

    latentspaces: TrainingDynamics = field(default_factory=TrainingDynamics)
    reconstructions: TrainingDynamics = field(default_factory=TrainingDynamics)
    mus: TrainingDynamics = field(default_factory=TrainingDynamics)
    sigmas: TrainingDynamics = field(default_factory=TrainingDynamics)
    losses: TrainingDynamics = field(default_factory=TrainingDynamics)
    preprocessed_data: torch.Tensor = field(default_factory=torch.Tensor)
    model: torch.nn.Module = field(default_factory=torch.nn.Module)
    model_checkpoints: Dict[int, torch.nn.Module] = field(default_factory=dict)
    datasets: Optional[DataSetContainer] = field(
        default_factory=lambda: DataSetContainer(train=None, valid=None, test=None)
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
        for field_name in self.__annotations__.keys():
            current_value = getattr(self, field_name)
            other_value = getattr(other, field_name)

            # Skip if other_value is None
            if other_value is None:
                continue

            # For TrainingDynamics, merge data
            if isinstance(
                current_value, TrainingDynamics
            ):  # TODO check if merge is the correct operation
                for epoch, split_data in other_value._data.items():
                    for split, value in split_data.items():
                        current_value.add(epoch, value, split)

            # Handle tensor/numpy array fields
            elif isinstance(current_value, (torch.Tensor, np.ndarray)):
                if current_value is None or current_value.size == 0:
                    setattr(self, field_name, other_value)

            # Handle torch.nn.Module specifically
            elif isinstance(current_value, torch.nn.Module):
                if current_value is None:
                    setattr(self, field_name, other_value)

            # Default case: replace if current is None
            else:
                if current_value is None:
                    setattr(self, field_name, other_value)
