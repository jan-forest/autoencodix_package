"""
Stores utility functions for the autoencodix package.
Uuse of OOP would be overkill for the simple functions in this module.
"""

import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints

import torch

from .default_config import DefaultConfig


def get_device(use_gpu: bool) -> str:
    """
    Handle device selection based on user preference and availability.
    Supports CPU, CUDA, and MPS devices.

    Parameters:
       use_gpu : bool
           Flag to enable GPU usage
    Returns:
         str
              Device string ('cpu', 'cuda', 'mps')

    """
    if not use_gpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"


# def config_method(func: Callable) -> Callable:
#     @functools.wraps(func)
#     def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
#         # Extract the user-provided config or default to the class config
#         user_config = kwargs.pop("config", None)
#         print(f"args and kwargs: {args}, {kwargs}")

#         if user_config:
#             print("User config found")
#             if not isinstance(user_config, DefaultConfig):
#                 raise TypeError("The 'config' parameter must be of type DefaultConfig.")
#             # Use a deepcopy of the provided config
#             method_config = copy.deepcopy(user_config)
#             # add values from self.config that are not in user_config
#             for k, v in self.config.__dict__.items():
#                 if not hasattr(method_config, k):
#                     setattr(method_config, k, v)
#         else:
#             print("No user config found")
#             # Use the class's default config and apply overrides
#             method_config = copy.deepcopy(self.config)
#             print(f"default config: {method_config}")
#             config_overrides = {
#                 k: v for k, v in kwargs.items() if hasattr(method_config, k)
#             }
#             method_config.update(**config_overrides)
#             print(f"Method config after updates: {method_config}")

#         # Forward the updated config to the wrapped function
#         return func(self, *args, config=method_config, **kwargs)

#     return wrapper


def config_method(valid_params: set[str] = None):
    """
    Decorator for methods that accept configuration parameters.

    Parameters
    ----------
    valid_params : set[str]
        Set of valid parameter names for this method. If None, all config parameters are valid.
    """

    def decorator(func: Callable) -> Callable:
        # Get the function's type hints
        hints = get_type_hints(func)  ## noqa: F841
        # Get the function's signature
        sig = inspect.signature(func)  ## noqa: F841

        # Document valid parameters in the function's docstring
        param_docs = "\nValid configuration parameters:\n"
        if valid_params:
            param_docs += "\n".join(f"- {param}" for param in sorted(valid_params))
        else:
            param_docs += "All configuration parameters are valid for this method."

        if func.__doc__ is None:
            func.__doc__ = ""
        func.__doc__ += param_docs

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract config
            func_sig = inspect.signature(func)
            param_names = list(func_sig.parameters.keys())

            if "config" in param_names and len(args) > param_names.index("config"):
                raise TypeError(
                    "The 'config' parameter must be passed as a keyword argument, not as a positional argument."
                )

            user_config = kwargs.pop("config", None)

            if user_config is None:
                # Use class config and apply overrides
                config = self.config.model_copy(deep=True)
                # Filter kwargs to only valid parameters if specified
                if valid_params:
                    config_overrides = {
                        k: v for k, v in kwargs.items() if k in valid_params
                    }
                else:
                    config_overrides = kwargs
                config.update(**config_overrides)
            else:
                if not isinstance(user_config, DefaultConfig):
                    raise TypeError(
                        "The 'config' parameter must be of type DefaultConfig"
                    )
                config = user_config

            return func(self, *args, config=config, **kwargs)

        wrapper.valid_params = valid_params

        return wrapper

    return decorator
