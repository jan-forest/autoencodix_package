from .default_config import DefaultConfig
from pydantic import Field


class VanillixConfig(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.
    """

    beta: float = Field(
        default=0.1,  # Overridden default (was 1.0)
        ge=0,
        description="Beta weighting factor for VAE loss",
    )
    epoch: int = Field(
        default=30, ge=0, description="How many epochs should the model train for."
    )
    # TODO find sensible defaults for Vanillix
