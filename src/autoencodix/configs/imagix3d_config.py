
from .default_config import DefaultConfig
from pydantic import Field, model_validator
from typing import Literal, Optional, Tuple, Union 


class Imagix3DConfig(DefaultConfig):
    """
    A specialized configuration for Imagix3D, inheriting from DefaultConfig.
    """

    beta: float = Field(
        default=0.1,  # Overridden default (was 1.0)
        ge=0,
        description="Beta weighting factor for VAE loss",
    )
    
    n_conv_layers_3d: int = Field(
        default=5,
        ge=1,
        description="Number of 3D convolutional layers."
    )

    spatial_shape_policy: Literal[
        "pad_to_multiple",
        "crop_to_multiple",
        "crop_or_pad_to_shape",
    ] = Field(
        default="pad_to_multiple",
        description="Policy for making input volumes spatially compatible with the 3D CNN."
    )

    target_multiple: Optional[int] = Field(
        default=None,
        ge=1,
        description="If None, derived from 2 ** n_conv_layers_3d."
    )

    target_shape_3d: Optional[Tuple[int, int, int]] = Field(
        default=None,
        description="Optional fixed target shape (D, H, W) for crop/pad workflows."
    )

    padding_mode_3d: Union[int, str] = Field(
        default=0,
        description="Padding mode/value for spatial padding."
    )

    normalize_nonzero_only: bool = Field(
        default=True,
        description="Whether intensity normalization should ignore zero background voxels."
    )
    
    ## Validation
    @model_validator(mode="after")
    def set_target_multiple(self) -> "Imagix3DConfig":
        """
        Ensures that targeted 3D image dimensions are compatible with the number of CNN layers.
        """
        if self.target_multiple is None:
            self.target_multiple = 2**self.n_conv_layers_3d
            return self
        elif self.target_multiple % 2**self.n_conv_layers_3d != 0:
            raise ValueError(
                f"'target_multiple' ({self.target_multiple}) must be divisible by 2^{self.n_conv_layers_3d}"
                )
        return self

    # TODO (maybe): find more sensible defaults for IMagix3DConfig