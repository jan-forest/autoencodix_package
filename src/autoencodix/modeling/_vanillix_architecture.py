from typing import Optional, Union

import torch
import torch.nn as nn

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig

from ._layer_factory import LayerFactory


# internal check done
# write tests: done
class VanillixArchitecture(BaseAutoencoder):
    """
    Vanilla Autoencoder implementation with separate encoder and decoder construction.

    Attributes
    ----------
    self.input_dim : int
        number of input features
    self.config: DefaultConfig
        Configuration object containing model architecture parameters
    self._encoder: nn.Module
        Encoder network of the autoencoder
    self._decoder: nn.Module
        Decoder network of the autoencoder

    Methods
    -------
    _build_network()
        Construct the encoder and decoder networks via the LayerFactory
    encode(x: torch.Tensor) -> torch.Tensor
        Encode the input tensor x
    decode(x: torch.Tensor) -> torch.Tensor
        Decode the latent tensor x
    forward(x: torch.Tensor) -> ModelOutput
        Forward pass of the model, fills in the reconstruction and latentspace attributes of ModelOutput class.

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
        self._config = config
        super().__init__(config, input_dim)
        self.input_dim = input_dim

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
            feature_dim=self.input_dim,
            latent_dim=self._config.latent_dim,
            n_layers=self._config.n_layers,
            enc_factor=self._config.enc_factor,
        )

        encoder_layers = []
        for i, (in_features, out_features) in enumerate(zip(enc_dim[:-1], enc_dim[1:])):
            last_layer = i == len(enc_dim) - 2
            encoder_layers.extend(
                LayerFactory.create_layer(
                    in_features=in_features,
                    out_features=out_features,
                    dropout_p=self._config.drop_p,
                    last_layer=last_layer,
                )
            )

        dec_dim = enc_dim[::-1]  # Reverse the dimensions and copy
        decoder_layers = []
        for i, (in_features, out_features) in enumerate(zip(dec_dim[:-1], dec_dim[1:])):
            last_layer = i == len(dec_dim) - 2
            decoder_layers.extend(
                LayerFactory.create_layer(
                    in_features=in_features,
                    out_features=out_features,
                    dropout_p=self._config.drop_p,
                    last_layer=last_layer,
                )
            )
        self._encoder = nn.Sequential(*encoder_layers)
        self._decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data.

        Parameters:
            x : torch.Tensor
        Returns:
            torch.Tensor

        """
        return self._encoder(x)

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input data.

        Parameters:
            x : torch.Tensor
        Returns:
            torch.Tensor

        """
        return self.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation.

        Parameters:
            x : torch.Tensor
        Returns:
            torch.Tensor

        """
        return self._decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the model.

        Parameters:
            x : torch.Tensor
        Returns:
            ModelOutput

        """
        latent = self.encode(x)
        return ModelOutput(
            reconstruction=self.decode(latent),
            latentspace=latent,
            latent_mean=None,
            latent_logvar=None,
            additional_info=None,
        )
