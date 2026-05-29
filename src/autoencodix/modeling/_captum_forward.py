import torch
import torch.nn as nn
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput


class CaptumForward(nn.Module):
    def __init__(self, model: BaseAutoencoder, dim: int):
        super(CaptumForward, self).__init__()  # <-- REQUIRED
        self.model = model  # (Registered as a submodule)
        self.dim = dim
        self.device = next(
            model.parameters()
        ).device  # Get the device of the model parameters

    def forward(self, x: torch.Tensor):
        mp: ModelOutput = self.model(x=x.to(self.device))
        latent = mp.latentspace
        output = latent[:, self.dim]
        return output.unsqueeze(1).to("cpu")
