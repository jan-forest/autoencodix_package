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
        "crop_or_pad_to_multiple",
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

    @model_validator(mode="after")
    def check_target_shape_3d(self) -> "Imagix3DConfig":
        """
        Ensures that the fixed 3D target image dimensions (if provided) are compatible with the number of CNN layers.
        """
        if self.target_shape_3d is not None:
            if not all(x % 2 ** self.n_conv_layers_3d == 0 for x in self.target_shape_3d):
                raise ValueError(
                    f"All dimensions in 'target_shape_3d' ({self.target_shape_3d}) must be divisible by 2^{self.n_conv_layers_3d}"
                )
            if len(self.target_shape_3d) != 3:
                raise ValueError(
                    "'target_shape_3d' must receive exactly 3 integer values."
                    f"You have only provided {len(self.target_shape_3d)}"
                )
        return self
    
    @model_validator(mode="after")
    def validate_spatial_shape_settings(self):
        if self.spatial_shape_policy in {
            "pad_to_multiple", 
            "crop_to_multiple",
            "crop_or_pad_to_multiple"
        }: 
            if self.target_shape_3d is not None:
                raise ValueError(
                "'target_shape_3d' must be None when spatial_shape_policy is "
                "'pad_to_multiple' or 'crop_to_multiple'."
            )

        if self.spatial_shape_policy == "crop_or_pad_to_shape":
            if self.target_shape_3d is None:
                raise ValueError(
                    "'target_shape_3d' must be provided when spatial_shape_policy "
                    "is 'crop_or_pad_to_shape'."
                )

        return self

    # TODO (maybe): find more sensible defaults for IMagix3DConfig