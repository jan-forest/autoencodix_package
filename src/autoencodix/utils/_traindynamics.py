from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np


@dataclass
class TrainingDynamics:
    """
    A type-safe, structured approach to storing training dynamics,
    where all values are numpy arrays.
    """

    _data: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict, repr=False)
    def add(
        self, epoch: int, data: Optional[Union[float, np.ndarray]], split: str = "train"
    ):
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
        if isinstance(data, (int, float)):
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Expected value to be of type numpy.ndarray, got {type(data)}."
            )

        if epoch not in self._data:
            self._data[epoch] = {}
        self._data[epoch][split] = data

    def get(
        self, epoch: Optional[int] = None, split: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
        """
        Retrieve stored numpy arrays with flexible filtering.

        Parameters:
        ----------
        epoch : int, optional
            Specific epoch to retrieve.
        split : str, optional
            Specific split to retrieve.

        Returns:
        -------
        Filtered data matching the specified criteria.
        """
        if epoch is None:
            return self._data

        epoch_data = self._data.get(epoch, {})

        if split is None:
            return epoch_data

        return epoch_data.get(split)

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
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
