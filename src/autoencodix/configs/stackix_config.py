from .default_config import DefaultConfig
from pydantic import Field, model_validator

import warnings


class StackixConfig(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.
    For Stackix, `save_memory` is always False (feature not supported).
    """

    beta: float = Field(
        default=0.1,
        ge=0,
        description="Beta weighting factor for VAE loss",
    )

    save_memory: bool = Field(
        default=False,
        description="Always False — not supported for Stackix.",
    )

    @model_validator(mode="before")
    def _force_save_memory_false(cls, values):
        if values.get("save_memory") is True:
            warnings.warn(
                "`save_memory=True` is not supported for StackixConfig — forcing to False., Set the checkpoint_interval to number of epochs if you want to save memory",
                UserWarning,
                stacklevel=2,
            )
            values["save_memory"] = False
        return values
