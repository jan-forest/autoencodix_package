from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from autoencodix.src.utils.default_config import DefaultConfig
from autoencodix.src.utils._model_output import ModelOutput


# TODO add defualt class docstring
class BaseAutoencoder(ABC, nn.Module):
    """
    Abstract BaseAutoencoder class defining the required methods for an autoencoder.
    Provides an interface for weight initialization that subclasses should implement.

    This class inherits from `torch.nn.Module` and is intended to be extended
    by specific autoencoder models.
    """

    def __init__(
        self, config: Optional[Union[DefaultConfig, None]], input_dim: int
    ) -> None:
        """
        Initializes the BaseAutoencoder class.
        """
        super().__init__()
        if config is None:
            config = DefaultConfig()
        self.latent_dim = config.latent_dim
        self.input_dim = input_dim

    @abstractmethod
    def _build_network(self) -> None:
        """
        Builds the encoder and decoder networks for the autoencoder model.

        This method should be implemented by subclasses to define the architecture
        of the encoder and decoder networks.
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be encoded.

        Returns
        -------
        torch.Tensor
            The encoded latent space representation.
        """
        return self.encoder(x)

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the input space.

        Parameters
        ----------
        x : torch.Tensor
            The latent tensor to be decoded.

        Returns
        -------
        torch.Tensor
            The decoded tensor, reconstructed from the latent space.
        """
        return self.decoder(x)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Combines encoding and decoding steps for the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be processed.

        Returns
        -------
        ModelOutput
            The reconstructed input tensor, and any additional information, depending on the model type.
        """
        pass


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
