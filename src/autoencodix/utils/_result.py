from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Union
from anndata import AnnData

import torch

from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage

from mudata import MuData
from ._traindynamics import TrainingDynamics


@dataclass
class LossRegistry:
    """
    Dataclass to store multiple TrainingDynamics objects for different losses.
    Objective is to make the Result class extensible for multiple losses that might
    be needed in the future for diffrent autoencoder architectures.

    Attributes
    ----------
    _losses : Dict[str, TrainingDynamics]
        A dictionary to store TrainingDynamics objects for different losses.
    Methods
    -------
    add(data: Dict, split: str, epoch: int) -> None
        Add a new loss value to the registry, calls the add method of TrainingDynamics.
    get(key: str) -> TrainingDynamics
        Retrieve a specific TrainingDynamics object from the registry, based on the loss name.
    losses() -> Dict[str, TrainingDynamics]
        Return all stored losses as a dictionary.
    set(key: str, value: TrainingDynamics) -> None
        Set a specific TrainingDynamics object in the registry.
    keys() -> List[str]
        Return all keys (loss names) in the registry as a list.

    """

    _losses: Dict[str, TrainingDynamics] = field(default_factory=dict)

    def __post_init__(self):
        for name, data in self._losses.items():
            if not isinstance(data, TrainingDynamics):
                raise ValueError(
                    f"Expected TrainingDynamics object, got {type(data)} for loss {name}"
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LossRegistry):
            return False
        if self._losses.keys() != other._losses.keys():
            return False
        return all(self._losses[key] == other._losses[key] for key in self._losses)

    def add(self, data: Dict, split: str, epoch: int) -> None:
        for key, value in data.items():
            if key not in self._losses:
                self._losses[key] = TrainingDynamics()
            self._losses[key].add(epoch=epoch, data=value, split=split)

    def get(self, key: str) -> TrainingDynamics:
        if key not in self._losses:
            self._losses[key] = TrainingDynamics()
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
    new_datasets: Optional[DatasetContainer] = field(
        default_factory=lambda: DatasetContainer(train=None, valid=None, test=None)
    )

    adata_latent: Optional[AnnData] = field(default_factory=AnnData)
    final_reconstruction: Optional[Union[DataPackage, MuData]] = field(default=None)
     # for stackix only
    sub_results: Optional[Dict[str, Any]] = field(default=None)
    sub_reconstructions: Optional[Dict[str, Any]] = field(default=None)

    # plots: Dict[str, Any] = field(
    #     default_factory=nested_dict
    # )  ## Nested dictionary of plots as figure handles

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

    def _update_traindynamics(
        self, current_value: TrainingDynamics, other_value: TrainingDynamics
    ) -> TrainingDynamics:
        """
        Update TrainingDynamics object with values from another TrainingDynamics object.

        Parameters
        ----------
        current_value : TrainingDynamics
            The current TrainingDynamics object to update.
        other_value : TrainingDynamics
            The TrainingDynamics object to update from.

        Returns
        -------
        TrainingDynamics
            Updated TrainingDynamics object.

        Examples
        --------
        >>> current = TrainingDynamics()
        >>> current._data = {1: {"train": np.array([1, 2, 3])},
        ...                   2: None}

        >>> other = TrainingDynamics()
        >>> other._data = {1: {"train": np.array([4, 5, 6])},
        ...                 2: {"train": np.array([7, 8, 9])}}
        >>> # after update
        >>> print(current._data)
        {1: {"train": np.array([4, 5, 6])}, # updated
         2: {"train": np.array([7, 8, 9])}} # kept, because other was None

        """

        if current_value is None:
            return other_value
        if current_value._data is None:
            return other_value

        for epoch, split_data in other_value._data.items():
            if split_data is None:
                continue
            if len(split_data) == 0:
                continue

            # If current epoch is None, it should be updated
            if epoch in current_value._data and current_value._data[epoch] is None:
                current_value._data[epoch] = {}
                for split, data in split_data.items():
                    if data is None:
                        continue
                    current_value.add(epoch=epoch, data=data, split=split)
                continue

            if epoch not in current_value._data:
                for split, data in split_data.items():
                    if data is None:
                        continue
                    current_value.add(epoch=epoch, data=data, split=split)
                continue
            # case when current epoch exists, then update all but None values
            for split, value in split_data.items():
                if value is not None:
                    current_value.add(epoch=epoch, data=value, split=split)

        # Ensure ordering
        current_value._data = dict(sorted(current_value._data.items()))

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
