from typing import Optional, Union, Tuple, Dict

import torch
import torch.nn as nn

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig

from ._layer_factory import LayerFactory


# internal check done
# write tests: done
class VanillixArchitecture(BaseAutoencoder):
    """Vanilla Autoencoder implementation with separate encoder and decoder construction.

    Attributes:
        input_dim: number of input features
        config: Configuration object containing model architecture parameters
        encoder: Encoder network of the autoencoder
        decoder: Decoder network of the autoencoder

    """

    def __init__(
        self,
        config: Optional[Union[None, DefaultConfig]],
        input_dim: int,
        ontologies: Optional[Union[Tuple, Dict]] = None,
        feature_order: Optional[Union[Tuple, Dict]] = None,
    ) -> None:
        """Initialize the Vanilla Autoencoder with the given configuration.

        Args:
            config: Configuration object containing model parameters.
            input_dim: Number of input features.
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
        """Construct the encoder with linear layers."""
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
        """Encodes the input data.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self._encoder(x)

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent space representation of the input data.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self._decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass of the model.

        Args:
            x: input Tensor
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
