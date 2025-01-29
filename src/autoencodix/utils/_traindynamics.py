from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any

import numpy as np


# internal check done
# write tests: done
@dataclass
class TrainingDynamics:
    """
    A type-safe, structured approach to storing training dynamics in the form
    epoch -> split -> data.

    Attributes:
    ----------
    _data : Dict[int, Dict[str, np.ndarray]]
        A dictionary to store numpy arrays for each epoch and split

    Methods:
    --------
    add(epoch: int, data: Optional[Union[float, np.ndarray]], split: str) -> None
        Add a numpy array for a specific epoch and split.
    get(epoch: Optional[int] = None, split: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]
        Retrieve stored numpy arrays with flexible filtering, based on epoch and/or split.
    epochs() -> list
        Return all recorded epochs.

    """

    _data: Dict[int, Dict[str, Union[np.ndarray, Dict]]] = field(
        default_factory=dict, repr=False
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingDynamics):
            return False
        if self._data.keys() != other._data.keys():
            return False
        for epoch, splits in self._data.items():
            if epoch not in other._data:
                return False
            for split, data in splits.items():
                if split not in other._data[epoch]:
                    return False
                other_data = other._data[epoch][split]
                if isinstance(data, np.ndarray) and isinstance(other_data, np.ndarray):
                    if not np.array_equal(data, other_data):
                        return False
                elif isinstance(data, dict) and isinstance(other_data, dict):
                    if data != other_data:  # Shallow dict comparison
                        return False
                else:
                    return False  # Mismatched types
        return True

    def add(
        self,
        epoch: int,
        data: Optional[Union[float, np.ndarray, Dict]],
        split: str = "train",
    ) -> None:
        """
        Add a numpy array for a specific epoch and split.

        Parameters:
        ----------
        epoch : int
            The epoch number.
        value : np.ndarray
            The numpy array to store.
        split : str, optional
            The data split (default: 'train').

        """
        if data is None:
            return
        if split not in ["train", "valid", "test"]:
            raise KeyError(
                f"Invalid split type: {split}, we only support 'train', 'valid', and 'test' splits."
            )
        if isinstance(data, (int, float)):
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            if not isinstance(data, Dict):
                raise TypeError(
                    f"Expected value to be of type numpy.ndarray or Dict, got {type(data)}."
                )

        if epoch not in self._data:
            self._data[epoch] = {}
        self._data[epoch][split] = data

    def get(
        self, epoch: Optional[int] = None, split: Optional[str] = None
    ) -> Union[
        np.ndarray,
        Dict[str, np.ndarray],
        Dict[int, Dict[str, np.ndarray]],
        Dict[Any, Any],
    ]:
        """
        Retrieve stored numpy arrays with flexible filtering.

        Parameters:
            epoch : int, optional
                Specific epoch to retrieve. If None, returns data for all epochs.
            split : str, optional
                Specific split to retrieve (e.g., 'train', 'valid', 'test').
                If None, returns data for all splits.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]
            - If epoch is None and split is None:
                Returns complete data dictionary {epoch: {split: data}}
            - If epoch is None and split is provided:
                Returns numpy array of values for the specified split across all epochs
            - If epoch is provided and split is None:
                Returns dictionary of all splits for that epoch {split: data}
            - If both epoch and split are provided:
                Returns numpy array for specific epoch and split

        Examples
        --------
        >>> dynamics = TrainingDynamics()
        >>> dynamics.add(0, np.array([0.1, 0.2]), "train")
        >>> dynamics.add(1, np.array([0.2, 0.3]), "train")
        >>>
        >>> # Get all data
        >>> dynamics.get()  # Returns {0: {"train": array([0.1, 0.2])}, 1: {"train": array([0.2, 0.3])}}
        >>>
        >>> # Get train split across all epochs
        >>> dynamics.get(split="train")  # Returns array([[0.1, 0.2], [0.2, 0.3]])
        >>>
        >>> # Get specific epoch
        >>> dynamics.get(epoch=0)  # Returns {"train": array([0.1, 0.2])}
        """
        # Case 1: No epoch specified
        if split not in ["train", "valid", "test", None]:
            raise KeyError(
                f"Invalid split type: {split}, we only support 'train', 'valid', and 'test' splits."
            )
        if epoch is None:
            # Case 1a: Split specified - return array of values across epochs
            if split is not None:
                epochs = sorted(self._data.keys())
                data = []
                for e in epochs:
                    if split in self._data[e]:
                        data.append(self._data[e][split])
                return np.array(data) if data else np.array([])
            # Case 1b: No split specified - return complete data dictionary
            return self._data

        # we need the not in self._data check here, because in the predict step we save
        # the model outputs with epoch -1
        if epoch < 0 and epoch not in self._data:
            # -1 equals to the highest epoch, -2 to the second highest, etc.
            epoch = max(self._data.keys()) + (epoch + 1)
        # Case 2: Epoch specified
        if epoch not in self._data:
            if split is not None:
                return np.array([])
            return {}
        # handle reverse indexing
        epoch_data = self._data[epoch]

        # Case 2a: No split specified - return all splits for epoch
        if split is None:
            return epoch_data

        # Case 2b: Both epoch and split specified
        return epoch_data.get(split, np.array([]))

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[np.ndarray, Dict[int, Dict[str, np.ndarray]], Any]:
        """
        Allow dictionary-style and slice-based access.

        Examples:
        ---------
        dynamics[100]  # Get data for epoch 100.
        dynamics[50:100]  # Get data for epochs 50-100.
        """
        if isinstance(key, int):
            return self.get(key)
        elif isinstance(key, slice):
            start = key.start or min(self._data.keys())
            stop = key.stop or max(self._data.keys())
            return {epoch: self.get(epoch) for epoch in range(start, stop)}
        raise KeyError(f"Invalid key type: {type(key)}")

    def epochs(self) -> list:
        """
        Return all recorded epochs.
        """
        return sorted(self._data.keys())
