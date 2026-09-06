from .default_config import DefaultConfig

from pydantic import Field, model_validator
import warnings
from typing import Optional, Literal


class XModalix_Ontix_Config(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.

    This class overrides specific training parameters like pretrain_epochs and beta
    for the XModalix model, while inheriting all other settings.
    """

    scaling: Literal["MINMAX", "NONE", "NOTSET", "LOG1P"] = Field(
        default="MINMAX",
        description="Global scaling method. For Ontix, only 'MINMAX' and 'NONE' are allowed, because we need positive values only",
    )

    pretrain_epochs: Optional[int] = Field(
        default=None,  # Overridden default (was 0)
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

    @model_validator(mode="after")
    def validate_nested_scaling(self) -> "XModalix_Ontix_Config":
        """
        Ensures that any scaling method set within DataInfo is also a valid
        positive-value scaler.
        """
        # Define the set of allowed scaling methods
        allowed_scalers = {"MINMAX", "NONE", "NOTSET"}

        # Loop through each data modality defined in the data_config
        for modality_name, data_info in self.data_config.data_info.items():
            if data_info.scaling not in allowed_scalers:
                raise ValueError(
                    f"Invalid scaling '{data_info.scaling}' for modality '{modality_name}'. "
                    f"OntixConfig only permits {list(allowed_scalers)}."
                )
        return self

    # TODO find sensible defaults for XModalix
