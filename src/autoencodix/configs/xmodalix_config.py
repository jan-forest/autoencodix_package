from .default_config import DefaultConfig

from pydantic import Field, model_validator
import warnings


class XModalixConfig(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.

    This class overrides specific training parameters like pretrain_epochs and beta
    for the XModalix model, while inheriting all other settings.
    """

    pretrain_epochs: int = Field(
        default=5,  # Overridden default (was 0)
        ge=0,
        description="Number of pretraining epochs, can be overwritten in DataInfo to have different number of pretraining epochs for each data modality",
    )

    beta: float = Field(
        default=0.1,  # Overridden default (was 1.0)
        ge=0,
        description="Beta weighting factor for VAE loss",
    )
    requires_paired: bool = Field(default=False)
    save_memory: bool = Field(
        default=False,
        description="Always False — not supported for Stackix.",
    )

    @model_validator(mode="before")
    def _force_save_memory_false(cls, values):
        if values.get("save_memory") is True:
            warnings.warn(
                "`save_memory=True` is not supported for XModalixConfig — forcing to False., Set the checkpoint_interval to number of epochs if you want to save memory",
                UserWarning,
                stacklevel=2,
            )
            values["save_memory"] = False
        return values

    # TODO find sensible defaults for XModalix
