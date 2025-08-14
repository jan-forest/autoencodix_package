
from .default_config import DefaultConfig
from pydantic import Field

class XModalixConfig(DefaultConfig):
    """
    A specialized configuration inheriting from DefaultConfig.
    
    This class overrides specific training parameters like pretrain_epochs and beta
    for the XModalix model, while inheriting all other settings.
    """

    pretrain_epochs: int = Field(
        default=50, # Overridden default (was 0)
        ge=0,
        description="Number of pretraining epochs, can be overwritten in DataInfo to have different number of pretraining epochs for each data modality",
    )

    beta: float = Field(
        default=0.1, # Overridden default (was 1.0)
        ge=0, 
        description="Beta weighting factor for VAE loss"
    )

    # TODO find sensible defaults for XModalix