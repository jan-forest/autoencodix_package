from .default_config import DefaultConfig
from pydantic import Field


class MaskixConfig(DefaultConfig):
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
    maskix_hidden_dim: int = Field(
        default=128,
        ge=8,
        description="The Maskix implementation follows https://doi.org/10.1093/bioinformatics/btae020. The authors use a hidden dimension 0f 256 for their neural network, so we set this as default",
    )
    maskix_swap_prob: float = Field(
        default=0.2,
        ge=0,
        description="For the Maskix input_data masinkg, we sample a probablity if samples within one gene should be swapt. This is done with a Bernoulli distribution, maskix_swap_prob is the probablity passed to the bernoulli distribution ",
    )
    delta_mask_predictor: float = Field(
        default=0.7,
        ge=0.0,
        description="Delt weighting factor of the mask predictin loss term for the Maskix",
    )
    delta_mask_corrupted: float = Field(
        default=0.75,
        ge=0.0,
        description="For the Maskix: if >0.5 this gives more weight for the correct reconstruction of corrupted input",
    )

    # TODO find sensible defaults for Maskix
