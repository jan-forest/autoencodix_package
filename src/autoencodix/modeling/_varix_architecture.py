from typing import Optional, Union

import torch
import torch.nn as nn

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig

from ._layer_factory import LayerFactory


class VarixArchitecture(BaseAutoencoder):
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
    reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor
        Reparameterization trick for VAE
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
        for i in range(len(enc_dim) - 1):
            # Last layer flag for final layer
            last_layer = i == len(enc_dim) - 2
            encoder_layers.extend(
                LayerFactory.create_layer(
                    in_features=enc_dim[i],
                    out_features=enc_dim[i + 1],
                    dropout_p=self._config.drop_p,
                    last_layer=last_layer,
                )
            )

        dec_dim = enc_dim[::-1]  # Reverse the dimensions and copy
        decoder_layers = []
        for i in range(len(dec_dim) - 1):
            # Last layer flag for final layer
            last_layer = i == len(dec_dim) - 2
            decoder_layers.extend(
                LayerFactory.create_layer(
                    in_features=dec_dim[i],
                    out_features=dec_dim[i + 1],
                    dropout_p=self._config.drop_p,
                    last_layer=last_layer,
                )
            )
        self._encoder = nn.Sequential(*encoder_layers)
        self._decoder = nn.Sequential(*decoder_layers)
        self._mu = nn.Linear(enc_dim[-2], self._config.latent_dim)
        self._logvar = nn.Linear(enc_dim[-2], self._config.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor x

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Encoded tensor

        """
        latent = self._encoder(x)
        mu = self._mu(latent)
        logvar = self._logvar(latent)
        # numeric stability
        logvar = torch.clamp(logvar, 0.01, 20)
        mu = torch.where(mu < 0.0000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE

        Parameters:
            mu : torch.Tensor
            logvar : torch.Tensor

        Returns:
            torch.Tensor

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent tensor x

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor

        Returns
        -------
        torch.Tensor
            Decoded tensor

        """
        return self._decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the model, fill

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        ModelOutput
            ModelOutput object containing the reconstructed tensor and latent tensor

        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)
        return ModelOutput(
            reconstruction=x_hat,
            latentspace=z,
            latent_mean=mu,
            latent_logvar=logvar,
            additional_info=None,
        )
