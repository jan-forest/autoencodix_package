"""
Stores utility functions for the autoencodix package.
Uuse of OOP would be overkill for the simple functions in this module.
"""

import inspect
import os
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, get_type_hints

import dill as pickle
import torch
from matplotlib import pyplot as plt

from .default_config import DefaultConfig


class BasePipeline:
    pass


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

    Parameters:
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


class Saver:
    """
    Handles the saving of BasePipeline objects.
    """

    def __init__(self, file_path: str):
        """
        Initializes the Saver with the base file path.

        Args:
            file_path: The base file path (without extensions).
        """
        self.file_path = file_path
        self.preprocessor_path = f"{file_path}_preprocessor.pkl"
        self.model_state_path = f"{file_path}_model.pth"

    def save(self, pipeline: "BasePipeline"):
        """
        Saves the BasePipeline object.

        Args:
            pipeline: The BasePipeline object to save.
        """
        self._save_pipeline_object(pipeline)
        self._save_preprocessor(pipeline._preprocessor)
        self._save_model_state(pipeline)

    def _save_pipeline_object(self, pipeline: "BasePipeline"):
        try:
            with open(self.file_path, "wb") as f:
                pickle.dump(pipeline, f)
            print("Pipeline object saved successfully.")
        except (pickle.PicklingError, OSError) as e:
            print(f"Error saving pipeline object: {e}")
            raise e

    def _save_preprocessor(self, preprocessor):
        if preprocessor is not None:
            try:
                with open(self.preprocessor_path, "wb") as f:
                    pickle.dump(preprocessor, f)
                print("Preprocessor saved successfully.")
            except (pickle.PickleError, OSError) as e:
                print(f"Error saving preprocessor: {e}")
                raise e

    def _save_model_state(self, pipeline: "BasePipeline"):
        if pipeline.result is not None and pipeline.result.model is not None:
            try:
                torch.save(pipeline.result.model.state_dict(), self.model_state_path)
                print("Model state saved successfully.")
            except (TypeError, OSError) as e:
                print(f"Error saving model state: {e}")
                raise e


class Loader:
    """
    Handles the loading of BasePipeline objects.
    """

    def __init__(self, file_path: str):
        """
        Initializes the Loader with the base file path.

        Args:
            file_path: The base file path (without extensions).
        """
        self.file_path = file_path
        self.preprocessor_path = f"{file_path}_preprocessor.pkl"
        self.model_state_path = f"{file_path}_model.pth"

    def load(self) -> Any:
        """
        Loads the BasePipeline object.

        Returns:
            The loaded BasePipeline object, or None on error.
        """
        loaded_obj = self._load_pipeline_object()
        if loaded_obj is None:
            return None  # Exit if the main object fails to load

        loaded_obj._preprocessor = (
            self._load_preprocessor()
        )  # Load even if it doesn't exist
        loaded_obj.result.model = self._load_model_state(loaded_obj)
        return loaded_obj

    def _load_pipeline_object(self) -> Any:
        print(f"Attempting to load a pipeline from {self.file_path}...")
        try:
            if not os.path.exists(self.file_path):
                print(f"Error: File not found at {self.file_path}")
                return None
            with open(self.file_path, "rb") as f:
                loaded_obj = pickle.load(f)
            print(
                f"Pipeline object loaded successfully. Actual type: {type(loaded_obj).__name__}"
            )
            return loaded_obj
        except (pickle.UnpicklingError, EOFError, OSError, FileNotFoundError) as e:
            print(f"Error loading pipeline object: {e}")
            return None

    def _load_preprocessor(self):
        if os.path.exists(self.preprocessor_path):
            try:
                with open(self.preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                print("Preprocessor loaded successfully.")
                return preprocessor
            except (pickle.UnpicklingError, EOFError, OSError, FileNotFoundError) as e:
                print(f"Error loading preprocessor: {e}")
                return None
        else:
            print("Preprocessor file not found. Skipping preprocessor load.")
            return None

    def _load_model_state(self, loaded_obj: "BasePipeline"):
        if os.path.exists(self.model_state_path):
            try:
                if (
                    loaded_obj.result is not None
                    and loaded_obj.result.model is not None
                ):
                    model_state = torch.load(self.model_state_path)
                    try:
                        loaded_obj.result.model.load_state_dict(model_state)

                        print("Model state loaded successfully.")
                        return loaded_obj.result.model
                    except Exception as e:
                        print(
                            f"Error when loading model, filling obj.result.model with None: {e}"
                        )
                        return None

                else:
                    return None
                    print("Warning: Model not initialized. Skipping model state load.")
            except (RuntimeError, OSError) as e:
                print(f"Error loading model state: {e}")
        else:
            print("Model state file not found. Skipping model state load.")
            return None
