import torch
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput


class CaptumForward(torch.nn.Module):
    def __init__(self, model: BaseAutoencoder, dim: int):
        self.model = model
        self.dim = dim

    def forward(self, x: torch.Tensor):
        mp: ModelOutput = self.model(x=x)
        latent = mp.latentspace
        output = latent[:, self.dim]
        return output.unsqueeze(1)  # general-leipzig
