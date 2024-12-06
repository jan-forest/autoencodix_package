import copy
import functools
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class DefaultConfig:
    """
    A dataclass for storing default configuration parameters.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the model (default: 0.001).
    batch_size : int
        The batch size for training (default: 32).
    epochs : int
        The number of epochs for training (default: 100).
    """
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs of configuration parameters to update.
        
        Raises
        ------
        ValueError
            If an unknown configuration parameter is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


def config_method(func: Callable) -> Callable:
    """
    A decorator to manage configurations for class methods.

    Parameters
    ----------
    func : Callable
        The method to be wrapped.

    Returns
    -------
    Callable
        The wrapped method.
    """
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Check if a config object is explicitly passed
        user_config = kwargs.get("config", None)

        if user_config is not None:
            # Validate the passed config
            if not isinstance(user_config, DefaultConfig):
                raise TypeError("The 'config' parameter must be of type DefaultConfig.")
            # Use the provided config directly
            method_config = copy.deepcopy(user_config)
        else:
            # Use the class config and apply overrides
            method_config = copy.deepcopy(self.config)
            config_overrides = {k: v for k, v in kwargs.items() if hasattr(method_config, k)}
            method_config.update(**config_overrides)
            # Remove config-related kwargs
            kwargs = {k: v for k, v in kwargs.items() if k not in config_overrides}

        # Call the original method with the resolved config
        return func(self, *args, config=method_config, **kwargs)

    return wrapper
