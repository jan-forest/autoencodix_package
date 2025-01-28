from dataclasses import dataclass, field
from typing import Any, Optional, Dict

import torch

from autoencodix.data import DatasetContainer

from ._traindynamics import TrainingDynamics


@dataclass
class LossRegistry:
    _losses: Dict[str, TrainingDynamics] = field(default_factory=dict)

    def add(self, data: Dict[str, torch.Tensor], split: str, epoch: int) -> None:
        for key, value in data.items():
            if key not in self._losses:
                self._losses[key] = TrainingDynamics()
            self._losses[key].add(epoch=epoch, data=value, split=split)

    def get(self, key: str) -> TrainingDynamics:
        return self._losses[key]

    def losses(self):
        return self._losses.items()

    def set(self, key: str, value: TrainingDynamics) -> None:
        self._losses[key] = value

    def keys(self):
        return self._losses.keys()


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
        Stores the total loss for different epochs and splits ('train', 'valid', 'test').
    recon_losses : TrainingDynamics
        Stores the reconstruction loss for different epochs and splits ('train', 'valid', 'test').
    var_losses : TrainingDynamics
        Stores the variational i.e. kl divergence loss for different epochs and splits ('train', 'valid', 'test').
    """

    latentspaces: TrainingDynamics = field(default_factory=TrainingDynamics)
    reconstructions: TrainingDynamics = field(default_factory=TrainingDynamics)
    mus: TrainingDynamics = field(default_factory=TrainingDynamics)
    sigmas: TrainingDynamics = field(default_factory=TrainingDynamics)
    losses: TrainingDynamics = field(default_factory=TrainingDynamics)
    sub_losses: LossRegistry = field(default_factory=LossRegistry)
    preprocessed_data: torch.Tensor = field(default_factory=torch.Tensor)
    model: torch.nn.Module = field(default_factory=torch.nn.Module)
    model_checkpoints: TrainingDynamics = field(default_factory=TrainingDynamics)
    datasets: Optional[DatasetContainer] = field(
        default_factory=lambda: DatasetContainer(train=None, valid=None, test=None)
    )

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve the value associated with a specific key.
        Parameters:
            key (str): The name of the attribute to retrieve.
        Returns:
            Any: The value of the specified attribute.
        Raises:
            KeyError - If the key is not a valid attribute of the Results class.

        """
        if not hasattr(self, key):
            raise KeyError(
                f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}"
            )
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Assign a value to a specific attribute.
        Parameters:
            key (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.
        Raises:
            KeyError
                If the key is not a valid attribute of the Results class.

        """
        if not hasattr(self, key):
            raise KeyError(
                f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}"
            )
        setattr(self, key, value)

    def _is_empty_value(self, value: Any) -> bool:
        """
        Helper method to check if an attribute of the Result object is empty.

        Parameters:
            value (Any): The value to check
        Returns:
            bool: True if the value is empty, False otherwise

        """

        if isinstance(value, TrainingDynamics):
            return len(value._data) == 0
        elif isinstance(value, torch.Tensor):
            return value.numel() == 0
        elif isinstance(value, torch.nn.Module):
            return sum(p.numel() for p in value.parameters()) == 0
        elif isinstance(value, DatasetContainer):
            return all(v is None for v in [value.train, value.valid, value.test])
        elif isinstance(value, LossRegistry):
            # single Nones are handled in update method (skipped)
            return all(v is None for _, v in value.losses())

        return False

    def update(self, other: "Result") -> None:
        """
        Update the current Result object with values from another Result object.
        For TrainingDynamics, merges the data across epochs and splits and overwrites if already exists.
        For all other attributes, replaces the current value with the other value.

        Parameters:
            other : Result
                The Result object to update from.
        Raises:
            TypeError
                If the input object is not a Result instance
        Returns:
            None

        """
        if not isinstance(other, Result):
            raise TypeError(f"Expected Result object, got {type(other)}")

        for field_name in self.__annotations__.keys():
            current_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            if self._is_empty_value(other_value):
                continue

            # Handle TrainingDynamics - merge data
            if isinstance(current_value, TrainingDynamics):
                current_value = self._update_traindynamics(current_value, other_value)
            # For all other types - replace with other value
            if isinstance(current_value, LossRegistry):
                for key, value in other_value.losses():
                    if value is None:
                        continue
                    if not isinstance(value, TrainingDynamics):
                        raise ValueError(
                            f"Expected TrainingDynamics object, got {type(value)}"
                        )
                    updated_dynamic = self._update_traindynamics(
                        current_value=current_value.get(key=key), other_value=value
                    )
                    current_value.set(key=key, value=updated_dynamic)
            else:
                setattr(self, field_name, other_value)

    def _update_traindynamics(self, current_value, other_value):
        for epoch, split_data in other_value._data.items():
            for split, value in split_data.items():
                current_value.add(
                    epoch=epoch, data=value, split=split
                )  # overwrites if already exists
        return current_value

    def __str__(self) -> str:
        """
        Provide a readable string representation of the Result object's public attributes.

        Returns
        -------
        str
            Formatted string showing all public attributes and their values
        """
        output = ["Result Object Public Attributes:", "-" * 30]

        for name in self.__annotations__.keys():
            if name.startswith("__"):
                continue

            value = getattr(self, name)
            if isinstance(value, TrainingDynamics):
                output.append(f"{name}: TrainingDynamics object")
            elif isinstance(value, torch.nn.Module):
                output.append(f"{name}: {value.__class__.__name__}")
            elif isinstance(value, dict):
                output.append(f"{name}: Dict with {len(value)} items")
            elif isinstance(value, torch.Tensor):
                output.append(f"{name}: Tensor of shape {tuple(value.shape)}")
            else:
                output.append(f"{name}: {value}")

        return "\n".join(output)

    def __repr__(self) -> str:
        """
        Return the same representation as __str__ for consistency.
        """
        return self.__str__()
