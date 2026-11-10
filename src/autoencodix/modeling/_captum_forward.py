import torch
import torch.nn as nn
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput


class CaptumForward(nn.Module):
    """Wrap an autoencoder so that its output is a single latent dimension.

    Captum attribution methods need a model that maps inputs to one scalar
    output per sample. This wrapper returns latent dimension ``dim`` of the
    model's latent space, which allows attributing that dimension to the
    input features (used by ``.explain()``).

    Attributes:
        model: The wrapped autoencoder.
        dim: Index of the latent dimension to return.
        device: Device of the wrapped model's parameters.

    (LLM-generated docstring, pending human review.)
    """

    def __init__(self, model: BaseAutoencoder, dim: int):
        """Store the model and the latent dimension to explain.

        Args:
            model: Trained autoencoder.
            dim: Index of the latent dimension to return.

        (LLM-generated docstring, pending human review.)
        """
        super(CaptumForward, self).__init__()  # <-- REQUIRED
        self.model = model  # (Registered as a submodule)
        self.dim = dim
        self.device = next(
            model.parameters()
        ).device  # Get the device of the model parameters

    def forward(self, x: torch.Tensor):
        """Return latent dimension ``dim`` for a batch of inputs.

        Args:
            x: Input batch; moved to the model's device.

        Returns:
            Tensor of shape (batch_size, 1) on the CPU.

        (LLM-generated docstring, pending human review.)
        """
        mp: ModelOutput = self.model(x=x.to(self.device))
        latent = mp.latentspace
        output = latent[:, self.dim]
        return output.unsqueeze(1).to("cpu")
