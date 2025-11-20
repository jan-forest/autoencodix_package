from typing import Optional
from dataclasses import dataclass
import torch


# internal check done
# write tests: done
@dataclass
class ModelOutput:
    """A structured output dataclass for autoencoder models.

    This class is used to encapsulate the outputs of autoencoder models in a
    consistent format, allowing for flexibility in the type of outputs returned
    by different architectures.

    Attributes:
        reconstruction: The reconstructed input data.
        latent_mean: The mean of the latent space distribution, applicable for models like VAEs.
        latent_logvar: The log variance of the latent space distribution, applicable for models like VAEs.
        additional_info:  A dictionary to store any additional information or intermediate outputs.
    """

    reconstruction: torch.Tensor
    latentspace: torch.Tensor
    latent_mean: Optional[torch.Tensor] = None
    latent_logvar: Optional[torch.Tensor] = None
    additional_info: Optional[dict] = None

    def __iter__(self):
        yield self
