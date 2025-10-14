"""
Stores utility functions for the autoencodix package.
Use of OOP would be overkill for the simple functions in this module.
"""

import inspect
import os
from collections import defaultdict
from dataclasses import MISSING, fields, is_dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import dill as pickle  # type: ignore
import torch
from matplotlib import pyplot as plt

from ..configs.default_config import DefaultConfig


# only for type hints, to avoid circual import
class BasePipeline:
    """Only for type hints in utils, not real BasePipeline class"""

    pass


def nested_dict():
    """Creates a nested defaultdict.

    This function returns a defaultdict where each value is another defaultdict
    of the same type. This allows for the creation of arbitrarily deep nested
    dictionaries without having to explicitly define each level.

    Returns:
        :A nested defaultdict where each value is another nested defaultdict.
    """
    return defaultdict(nested_dict)


def nested_to_tuple(d, base=()):
    """Recursively converts a nested dictionary into tuples.

    Args:
        d: The dictionary to convert.
        base: The base tuple to start with. Defaults to ().

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
    """Display a given Matplotlib figure in a new window.

    Args:
        fig: The figure to be displayed.

    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig  # ty: ignore[possibly-unbound-attribute]
    fig.set_canvas(new_manager.canvas)  # ty: ignore[possibly-unbound-attribute]


def config_method(valid_params: Optional[set[str]] = None):
    """Decorator for methods that accept configuration parameters via kwargs or an explicit 'config' object.

    It separates kwargs intended for the function's signature from those
    intended as configuration overrides, validates the latter against
    `valid_params`, applies valid overrides to a copy of `self.config`,
    and passes the appropriate arguments to the decorated function.

    Args:
        valid_params: Set of valid configuration parameter names that can be overridden
            via kwargs for this method. If None, all kwargs not matching the
            function signature are considered potentially valid config overrides.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        param_docs = "\n\nValid configuration parameters (passed via **kwargs):\n"
        if valid_params:
            param_docs += "\n".join(f"- `{param}`" for param in sorted(valid_params))
        else:
            param_docs += (
                "All keyword arguments not matching the function's "
                "signature are treated as potential configuration overrides."
            )

        if func.__doc__ is None:
            func.__doc__ = ""
        # Avoid duplicating if decorator is applied multiple times (though unlikely)
        if "Valid configuration parameters" not in func.__doc__:
            func.__doc__ += param_docs
        # --- End Docstring Modification ---

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Pop the explicit config object if provided
            user_config = kwargs.pop("config", None)

            # Get names of parameters in the decorated function's signature
            # that can accept keyword arguments (excluding self and config)
            func_sig_kwarg_names = {
                name
                for name, param in sig.parameters.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and name not in ("self", "config")
            }

            # Separate kwargs into those matching the signature and potential config overrides
            func_specific_kwargs = {}
            potential_config_kwargs = {}
            for k, v in kwargs.items():
                if k in func_sig_kwarg_names:
                    func_specific_kwargs[k] = v
                else:
                    potential_config_kwargs[k] = v

            # Determine the configuration object to use
            if user_config is None:
                # No explicit config object, use self.config and apply overrides
                if not hasattr(self, "config"):
                    raise AttributeError(
                        f"{type(self).__name__} instance is missing 'config' attribute."
                    )
                # Ensure self.config is the right type (or duck-types model_copy/model_dump)
                if not hasattr(self.config, "model_copy") or not hasattr(
                    self.config, "model_dump"
                ):
                    raise TypeError(
                        f"'self.config' on {type(self).__name__} must have 'model_copy' and 'model_dump' methods."
                    )

                final_config = self.config.model_copy(deep=True)  # Start with a copy

                # Validate potential_config_kwargs against valid_params
                if valid_params:
                    # Check for invalid *config* parameters among the potential overrides
                    invalid_config_params = (
                        set(potential_config_kwargs.keys()) - valid_params
                    )
                    if invalid_config_params:
                        print(
                            f"\nWarning: The following parameters are not valid "
                            f"configuration overrides for {func.__name__}:"  # type: ignore
                        )
                        print(
                            f"Invalid config parameters: {', '.join(invalid_config_params)}"
                        )
                        print(
                            f"Valid config parameters are: {', '.join(sorted(valid_params))}"
                        )
                        print("These parameters will be ignored.")

                    # Filter potential overrides to only include those listed in valid_params
                    # and that actually exist as fields in the config object
                    # (prevents adding arbitrary attributes if DefaultConfig uses Pydantic's extra='forbid')
                    valid_config_fields = {
                        p
                        for p in valid_params
                        if hasattr(final_config, p)  # Safer check
                    }
                    config_overrides = {
                        k: v
                        for k, v in potential_config_kwargs.items()
                        if k in valid_config_fields
                    }

                else:  # valid_params is None: Allow all potential config kwargs as overrides
                    # Optional: Add a check here if you want to ensure they exist in the config model
                    config_overrides = potential_config_kwargs

                # Apply the valid overrides
                if config_overrides:
                    final_config = final_config.model_copy(update=config_overrides)

            else:
                # User provided an explicit config object
                # Add type check if DefaultConfig class is available
                # Note: Replace 'object' with 'DefaultConfig' if it's imported/defined
                if not isinstance(user_config, DefaultConfig):
                    # Trying to be robust if DefaultConfig is not strictly enforced type
                    if hasattr(user_config, "model_copy") and hasattr(
                        user_config, "model_dump"
                    ):
                        pass  # Looks like a valid config object duck-typing Pydantic
                    else:
                        raise TypeError(
                            "The 'config' parameter must be a valid configuration object "
                            "(e.g., an instance of DefaultConfig or similar)."
                        )
                final_config = user_config
                # Decide what to do with potential_config_kwargs when user_config is provided.
                # Option 1: Ignore them (current implementation below)
                # Option 2: Raise an error if any exist
                # Option 3: Apply them even to the user_config (might be unexpected)
                if potential_config_kwargs:
                    print(
                        f"\nWarning: Additional keyword arguments provided "
                        f"({', '.join(potential_config_kwargs.keys())}) "
                        f"while an explicit 'config' object was also passed to {func.__name__}. "  # type: ignore
                        f"These additional arguments will be ignored as configuration overrides."
                    )

            # Call the original function with the correct arguments
            # Pass: self, original *args, the determined config object,
            # and only the **kwargs that matched the function's signature.
            return func(self, *args, config=final_config, **func_specific_kwargs)

        # Preserve information about valid_params on the wrapper if needed elsewhere
        setattr(wrapper, "valid_params", valid_params)
        return wrapper

    return decorator


class Saver:
    """Handles the saving of BasePipeline objects.

    Atrributes:
        file_path: path to save file.
        preprocessor_path: path where pickle object of preprocesser should be saved.
        model_state_path: path where model state dict should be saved.
        save_all: indicator if all results should be save, or only core pipeline functionalty

    """

    def __init__(self, file_path: str, save_all: bool):
        """Initializes the Saver with the base file path.

        Args:
            file_path: The base file path (without extensions).
        """

        self.file_path = file_path
        self.save_all = save_all
        self.preprocessor_path = f"{file_path}_preprocessor.pkl"
        self.model_state_path = f"{file_path}_model.pth"

        folder = os.path.dirname(self.file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    def save(self, pipeline: "BasePipeline"):
        """Saves the BasePipeline object.

        Args:
            pipeline: The BasePipeline object to save.
        """

        self._save_preprocessor(pipeline._preprocessor)  # type: ignore
        if not self.save_all:
            print("saving memory efficient")
            self.reset_to_defaults(pipeline.result)  # ty: ignore
            pipeline.preprocessed_data = None  # ty: ignore
            pipeline.raw_user_data = None  # ty: ignore
            pipeline._preprocessor = type(pipeline._preprocessor)(  # ty: ignore
                config=pipeline.config  # ty: ignore
            )  # ty: ignore
            pipeline._visualizer = type(pipeline._visualizer)()  # ty: ignore

        self._save_pipeline_object(pipeline)
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
        if pipeline.result is not None and pipeline.result.model is not None:  # type: ignore
            try:
                torch.save(pipeline.result.model.state_dict(), self.model_state_path)  # type: ignore
                print("Model state saved successfully.")
            except (TypeError, OSError) as e:
                print(f"Error saving model state: {e}")
                raise e

    def reset_to_defaults(self, obj):
        if not is_dataclass(obj):
            raise ValueError("Object must be a dataclass")

        for f in fields(obj):
            if f.name == "adata_latent" or f.name == "model":
                print(f)
                continue
            if f.default_factory is not MISSING:
                setattr(obj, f.name, f.default_factory())
            elif f.default is not MISSING:
                setattr(obj, f.name, f.default)
            else:
                setattr(obj, f.name, None)


class Loader:
    """Handles the loading of BasePipeline objects.

    Atrributes:
        file_path: path of saved pipeline object.
        preprocessor_path: path where pickle object of preprocesser should was saved.
        model_state_path: path where model state dict was saved.
    """

    def __init__(self, file_path: str):
        """Initializes the Loader with the base file path.

        Args:
            file_path: The base file path (without extensions).
        """
        self.file_path = file_path
        self.preprocessor_path = f"{file_path}_preprocessor.pkl"
        self.model_state_path = f"{file_path}_model.pth"

    def load(self) -> Any:
        """Loads the BasePipeline object.

        Returns:
            The loaded BasePipeline object, or None on error.
        """
        loaded_obj = self._load_pipeline_object()
        if loaded_obj is None:
            return None  # Exit if the main object fails to load

        loaded_obj._preprocessor = self._load_preprocessor()
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
                    loaded_obj.result is not None  # type: ignore
                    and loaded_obj.result.model is not None  # type: ignore
                ):
                    model_state = torch.load(self.model_state_path)
                    try:
                        loaded_obj.result.model.load_state_dict(model_state)  # type: ignore

                        print("Model state loaded successfully.")
                        return loaded_obj.result.model  # type: ignore
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


def flip_labels(labels: torch.Tensor) -> torch.Tensor:
    """Randomly flip modality labels with probability (1 - 1/n_modalities), vectorized.

    This is mainly used for advers training in multi modal xmodalix.

    Args:
        labels: tensor of labels
    Returns:
        flipped tensor

    """
    device = labels.device
    n_modalities = labels.unique().numel()
    batch_size = labels.size(0)
    flip_prob = 1.0 - 1.0 / n_modalities

    # Decide which labels to flip
    flip_mask = torch.rand(batch_size, device=device) < flip_prob

    # Sample random labels for flipping
    rand_labels = torch.randint(0, n_modalities, size=(batch_size,), device=device)

    # Ensure sampled labels are different from original labels
    needs_resample = (rand_labels == labels) & flip_mask
    while needs_resample.any():
        rand_labels[needs_resample] = torch.randint(
            0, n_modalities, size=(needs_resample.sum(),), device=device
        )
        needs_resample = (rand_labels == labels) & flip_mask

    # Apply flipped labels where needed
    flipped = labels.clone()
    flipped[flip_mask] = rand_labels[flip_mask]

    return flipped


def find_translation_keys(
    config: DefaultConfig,
    trained_modalities: List[str],
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
) -> Dict[str, str]:  # type: ignore
    """Find translation source and target modalities.

    Determines which modalities serve as the "from" and "to" directions for
    cross-modal prediction, either from explicit arguments or from the
    configuration.

    Args:
        config: Experiment configuration containing data information.
        trained_modalities: List of trained modality names.
        from_key: Optional name of the source modality.
        to_key: Optional name of the target modality.

    Returns:
        A dictionary with two entries:
            - "from": Name of the source modality.
            - "to": Name of the target modality.

    Raises:
        ValueError: If no valid "from" or "to" modality is found, or if
        multiple conflicting directions are specified.
    """
    from_key_final: Optional[str] = None
    to_key_final: Optional[str] = None
    simple_names: List[str] = [tm.split(".")[1] for tm in trained_modalities]

    if from_key and to_key:
        for name in trained_modalities:
            simple_name = name.split(".")[1]
            if from_key == simple_name or from_key == name:
                from_key_final = name
            # use if instead of elif to allow for reference prediciton where from_key == to_key
            if to_key == simple_name or from_key == name:
                to_key_final = name
        # if the users passes from_key and to_key and we don't find them, we raise an error
        if not (from_key_final and to_key_final):
            raise ValueError(
                f"Invalid translation keys: {from_key} => {to_key}, valid keys are: {simple_names}"
            )
        return {"from": from_key_final, "to": to_key_final}

    data_info = config.data_config.data_info
    for name in trained_modalities:
        simple_name = name.split(".")[1]

        cur_info = data_info.get(simple_name)
        if not cur_info or not hasattr(cur_info, "translate_direction"):
            continue

        direction = cur_info.translate_direction
        if direction == "to":
            if to_key_final is not None:
                raise ValueError(
                    f"Multiple 'to' directions found: '{to_key_final}' and '{name}'"
                )
            to_key_final = name
        elif direction == "from":
            if from_key_final is not None:
                raise ValueError(
                    f"Multiple 'from' directions found: '{from_key_final}' and '{name}'"
                )
            from_key_final = name

    if from_key_final is None:
        raise ValueError(
            "No modality with a 'from' direction was specified in the config."
        )
    if to_key_final is None:
        raise ValueError(
            "No modality with a 'to' direction was specified in the config."
        )
    assert from_key_final is not None and to_key_final is not None

    return {"from": from_key_final, "to": to_key_final}
