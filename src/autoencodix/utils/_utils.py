"""
Stores utility functions for the autoencodix package.
Only use of OOP would be overkill for the simple functions in this module.
"""

import copy
import functools
from typing import Any, Callable

import torch

from .default_config import DefaultConfig


def get_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"


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
