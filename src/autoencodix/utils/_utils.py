"""
Stores utility functions for the autoencodix package.
Uuse of OOP would be overkill for the simple functions in this module.
"""
from collections import defaultdict
import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_type_hints

from matplotlib import pyplot as plt

from .default_config import DefaultConfig

def nested_dict():
    """
    Creates a nested defaultdict.

    This function returns a defaultdict where each value is another defaultdict
    of the same type. This allows for the creation of arbitrarily deep nested
    dictionaries without having to explicitly define each level.

    Returns:
        defaultdict: A nested defaultdict where each value is another nested defaultdict.
    """
    return defaultdict(nested_dict)

def nested_to_tuple(d, base=()):
    """
    Recursively converts a nested dictionary into tuples.

    Args:
        d (dict): The dictionary to convert.
        base (tuple, optional): The base tuple to start with. Defaults to ().

    Yields:
        tuple: Tuples representing the nested dictionary structure, where each tuple
               contains the keys leading to a value and the value itself.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from nested_to_tuple(v, base + (k,))
        else:
            yield base + (k, v)

def show_figure(fig):
    """
    Display a given Matplotlib figure in a new window.

    Parameters:
    fig (matplotlib.figure.Figure): The figure to be displayed.

    Returns:
    None
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

# internal check done
# write tests: done
def config_method(valid_params: Optional[set[str]] = None):
    """
    Decorator for methods that accept configuration parameters.
    Parameters
    ----------
    valid_params : set[str]
        Set of valid parameter names for this method. If None, all config parameters are valid.
    """

    def decorator(func: Callable) -> Callable:
        get_type_hints(func)  ## noqa: F841
        inspect.signature(func)  ## noqa: F841

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
            user_config = kwargs.pop("config", None)

            if user_config is None:
                config = self.config.model_copy(deep=True)

                # Check for invalid parameters and warn user
                if valid_params:
                    invalid_params = set(kwargs.keys()) - valid_params
                    if invalid_params:
                        print(
                            f"\nWarning: The following parameters are not valid for {func.__name__}:"
                        )
                        print(f"Invalid parameters: {', '.join(invalid_params)}")
                        print(
                            f"Valid parameters are: {', '.join(sorted(valid_params))}"
                        )
                    # check if parameter is a config parameter

                    # Filter out invalid parameters
                    valid_config_params = {
                        p for p in valid_params if p in config.model_dump()
                    }
                    config_overrides = {
                        k: v for k, v in kwargs.items() if k in valid_config_params
                    }
                else:
                    config_overrides = kwargs

                config = config.model_copy(update=config_overrides)
            else:
                if not isinstance(user_config, DefaultConfig):
                    raise TypeError(
                        "The 'config' parameter must be of type DefaultConfig"
                    )
                config = user_config

            return func(self, *args, config=config, **kwargs)

        setattr(wrapper, "valid_params", valid_params)
        return wrapper

    return decorator
