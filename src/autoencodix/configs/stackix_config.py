from .default_config import DefaultConfig
from pydantic import Field


class StackixConfig(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.
    """

    beta: float = Field(
        default=0.1,  # Overridden default (was 1.0)
        ge=0,
        description="Beta weighting factor for VAE loss",
    )
    # TODO find sensible defaults for Stackix
