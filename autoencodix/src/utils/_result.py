from dataclasses import dataclass, field
from typing import Dict, Any
import torch
import numpy as np

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
            raise KeyError(f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}")
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
            raise KeyError(f"Invalid key: '{key}'. Allowed keys are: {', '.join(self.__annotations__.keys())}")
        setattr(self, key, value)