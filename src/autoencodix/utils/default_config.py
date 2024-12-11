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
    latent_dim: int = 16
    n_layers: int = 3
    enc_factor: int = 4
    input_dim: int = 10000
    drop_p: float = 0.1

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
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Extract the user-provided config or default to the class config
        user_config = kwargs.pop("config", None)
        print(f"args and kwargs: {args}, {kwargs}")

        if user_config:
            print("User config found")
            if not isinstance(user_config, DefaultConfig):
                raise TypeError("The 'config' parameter must be of type DefaultConfig.")
            # Use a deepcopy of the provided config
            method_config = copy.deepcopy(user_config)
            # add values from self.config that are not in user_config
            for k, v in self.config.__dict__.items():
                if not hasattr(method_config, k):
                    setattr(method_config, k, v)
        else:
            print("No user config found")
            # Use the class's default config and apply overrides
            method_config = copy.deepcopy(self.config)
            print(f"default config: {method_config}")
            config_overrides = {
                k: v for k, v in kwargs.items() if hasattr(method_config, k)
            }
            method_config.update(**config_overrides)
            print(f"Method config after updates: {method_config}")

        # Forward the updated config to the wrapped function
        return func(self, *args, config=method_config, **kwargs)

    return wrapper
