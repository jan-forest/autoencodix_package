from typing import Optional, Union

import torch
import torch.nn as nn

from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig

from autoencodix.base._base_autoencoder import BaseAutoencoder
from ._layer_factory import LayerFactory


class VanillixArchitecture(BaseAutoencoder):
    """
    Vanilla Autoencoder implementation with separate encoder and decoder construction.
    """

    def __init__(
        self, config: Optional[Union[None, DefaultConfig]], input_dim: int
    ) -> None:
        """
        Initialize the Vanilla Autoencoder with the given configuration.

        Parameters
        ----------
        config : Optional[Union[None, DefaultConfig]]
            Configuration object containing model parameters.
        """
        if config is None:
            config = DefaultConfig()
        super().__init__(config, input_dim)
        self.input_dim = input_dim
        print("input_dim", input_dim)
        self.latent_dim = config.latent_dim
        self.n_layers = config.n_layers
        self.enc_factor = config.enc_factor
        self.drop_p = config.drop_p

        # populate self.encoder and self.decoder
        self._build_network()
        self.apply(self._init_weights)

    def _build_network(self) -> None:
        """
        Construct the encoder with linear layers.

        Returns
        -------
        nn.Sequential
            Encoder model.
        """
        # Calculate layer dimensions
        enc_dim = LayerFactory.get_layer_dimensions(
            self.input_dim, self.latent_dim, self.n_layers, self.enc_factor
        )

        encoder_layers = []
        for i in range(len(enc_dim) - 1):
            # Last layer flag for final layer
            last_layer = i == len(enc_dim) - 2
            encoder_layers.extend(
                LayerFactory.create_layer(
                    enc_dim[i], enc_dim[i + 1], self.drop_p, last_layer=last_layer
                )
            )

        dec_dim = enc_dim[::-1]  # Reverse the dimensions
        decoder_layers = []
        for i in range(len(dec_dim) - 1):
            # Last layer flag for final layer
            last_layer = i == len(dec_dim) - 2
            decoder_layers.extend(
                LayerFactory.create_layer(
                    dec_dim[i], dec_dim[i + 1], self.drop_p, last_layer=last_layer
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input data."""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the latent space representation."""
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Performs a forward pass."""
        latent = self.encode(x)
        return ModelOutput(
            reconstruction=self.decode(latent),
            latentspace=latent,
            latent_mean=None,
            latent_logvar=None,
            additional_info=None,
        )
