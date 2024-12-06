from typing import Optional

import torch
import torch.nn as nn
from ._layer_factory import LayerFactory
from ._base_autoencoder import BaseAutoencoder
from autoencodix.src.utils.default_config import DefaultConfig


# TODO check if correct and implemnet new config logic#
# TODO add tests
class VanillixArchitecture(BaseAutoencoder):
    """
    Vanilla Autoencoder implementation that consists of an encoder and a decoder.

    This model maps input data to a latent space using the encoder and reconstructs
    the input data from the latent space using the decoder.

    Attributes
    ----------
    encoder : nn.Sequential
        The encoder network, consisting of layers created by LayerFactory.
    decoder : nn.Sequential
        The decoder network, consisting of layers created by LayerFactory.

    Methods
    -------
    encode(x: torch.Tensor) -> torch.Tensor
        Encodes the input data into a latent space representation.
    decode(x: torch.Tensor) -> torch.Tensor
        Decodes the latent space representation back to the input space.
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass (encoding followed by decoding).
    """

    def __init__(self, config: Optional[DefaultConfig] = None) -> None:
        """
        Constructs all the necessary attributes for the Vanillix Autoencoder
        object using the provided configuration.

        Parameters
        ----------
        config : DefaultConfig, optional
            The configuration object containing model parameters.
            If not provided, default configuration is used.
        """

        super().__init__()

        self.input_dim = 784  # Example default value
        self.latent_dim = 16
        self.n_layers = 3
        self.enc_factor = 4
        self.drop_p = 0.1

        # Building the encoder and decoder using the configuration
        self.encoder = nn.Sequential(
            *LayerFactory.create_encoder(
                self.input_dim,
                self.latent_dim,
                self.n_layers,
                self.enc_factor,
                self.drop_p,
            )
        )
        self.decoder = nn.Sequential(
            *LayerFactory.create_decoder(
                self.input_dim,
                self.latent_dim,
                self.n_layers,
                self.enc_factor,
                self.drop_p,
            )
        )
        self.apply(self._init_weights)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data into a latent space representation.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be encoded.

        Returns
        -------
        torch.Tensor
            The latent space representation of the input data.
        """
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space representation back to the input data space.

        Parameters
        ----------
        x : torch.Tensor
            The latent space representation to be decoded.

        Returns
        -------
        torch.Tensor
            The reconstructed input data.
        """
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the autoencoder (encoding followed by decoding).

        Parameters
        ----------
        x : torch.Tensor
            The input data to be passed through the model.

        Returns
        -------
        torch.Tensor
            The reconstructed input data from the forward pass.
        """
        latent = self.encode(x)
        return self.decode(latent)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
