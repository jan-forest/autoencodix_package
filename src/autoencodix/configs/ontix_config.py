from typing import Literal
from pydantic import Field, model_validator
from .default_config import DefaultConfig


class OntixConfig(DefaultConfig):
    """
    A specialized configuration for Ontix that only allows scaling methods
    guaranteeing non-negative outputs.
    """

    # 1. Override the top-level 'scaling' attribute
    scaling: Literal["MINMAX", "NONE", "NOTSET", "LOG1P"] = Field(
        default="MINMAX",
        description="Global scaling method. For Ontix, only 'MINMAX' and 'NONE' are allowed, because we need positive values only",
    )

    # 2. Add a validator for the nested 'scaling' attributes
    @model_validator(mode="after")
    def validate_nested_scaling(self) -> "OntixConfig":
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
