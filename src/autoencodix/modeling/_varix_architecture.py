from typing import Optional, Union, Tuple, Dict

import torch
import torch.nn as nn

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig

from ._layer_factory import LayerFactory


class VarixArchitecture(BaseAutoencoder):
    """Variational Autoencoder implementation with separate encoder and decoder construction.

    Attributes:
        input_dim: number of input features
        config: Configuration object containing model architecture parameters
        encoder: Encoder network of the autoencoder
        decoder: Decoder network of the autoencoder
        mu: Linear layer to compute the mean of the latent distribution
    logvar: Linear layer to compute the log-variance of the latent distribution

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
        self._config: DefaultConfig = config
        super().__init__(config=config, input_dim=input_dim)
        self.input_dim: int = input_dim
        self._mu: nn.Module
        self._logvar: nn.Module

        # populate self.encoder and self.decoder
        self._build_network()
        self.apply(self._init_weights)

    def _build_network(self) -> None:
        """Construct the encoder and decoder networks.

        Handles cases where `n_layers=0` by skipping the encoder and using only mu/logvar.
        """
        enc_dim = LayerFactory.get_layer_dimensions(
            feature_dim=self.input_dim,
            latent_dim=self._config.latent_dim,
            n_layers=self._config.n_layers,
            enc_factor=self._config.enc_factor,
        )
        #

        # Case 1: No Hidden Layers (Direct Mapping)
        self._encoder = nn.Sequential()
        self._mu = nn.Linear(self.input_dim, self._config.latent_dim)
        self._logvar = nn.Linear(self.input_dim, self._config.latent_dim)

        # Case 2: At Least One Hidden Layer
        if self._config.n_layers > 0:
            encoder_layers = []
            # print(enc_dim)
            for i, (in_features, out_features) in enumerate(
                zip(enc_dim[:-1], enc_dim[1:])
            ):
                # since we add mu and logvar, we will remove the last layer
                if i == len(enc_dim) - 2:
                    break
                encoder_layers.extend(
                    LayerFactory.create_layer(
                        in_features=in_features,
                        out_features=out_features,
                        dropout_p=self._config.drop_p,
                        last_layer=False,  # only for decoder relevant
                    )
                )

            self._encoder = nn.Sequential(*encoder_layers)
            self._mu = nn.Linear(enc_dim[-2], self._config.latent_dim)
            self._logvar = nn.Linear(enc_dim[-2], self._config.latent_dim)

        # Construct Decoder (Same for Both Cases)
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

        self._decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor x.

        Args:
            x: Input tensor

        Returns:
            Encoded tensor

        """

        latent = x  # for case where n_layers=0
        if len(self._encoder) > 0:
            latent = self._encoder(x)
        mu = self._mu(latent)
        logvar = self._logvar(latent)
        # numeric stability
        logvar = torch.clamp(logvar, 0.01, 20)
        mu = torch.where(mu < 0.0000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE

        Args:
            mu: torch.Tensor
            logvar: torch.Tensor

        Returns:
            torch.Tensor

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent space representation of the input.

        Args:
            x: Input tensor

        Returns:
            Latent space representation

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the latent tensor x

        Args:
            x: Latent tensor

        Returns:
            Decoded tensor
        """

        return self._decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass of the model, fill

        Args:
            x: Input tensor

        Returns:
            ModelOutput object containing the reconstructed tensor and latent tensor

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return ModelOutput(
            reconstruction=x_hat,
            latentspace=z,
            latent_mean=mu,
            latent_logvar=logvar,
            additional_info=None,
        )
