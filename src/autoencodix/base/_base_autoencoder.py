from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput


# internal check done
# write tests: done
class BaseAutoencoder(ABC, nn.Module):
    """
    Abstract BaseAutoencoder class defining the required methods for an autoencoder.
    Builds the encoder and decoder networks, with the _build_network method,
    Each autoencoder model should implement the encode and decode methods and forward method.
    Weight initalization is also encouraged to be implemented in the _init_weights method.

    This class inherits from `torch.nn.Module` and is intended to be extended
    by specific autoencoder models i.e. a variational autoencoder might add a
    reparmeterization method.

    Attributes
    ----------
    self.input_dim : int
        number of input features
    self.config: DefaultConfig
        Configuration object containing model architecture parameters
    self.encoder: nn.Module
        Encoder network
    self.decoder: nn.Module
        Decoder network

    Methods
    -------
    _build_network()
        Abstract method to build the encoder and decoder networks
    encode(x: torch.Tensor) -> torch.Tensor
        Abstract method to encode input tensor x
    decode(x: torch.Tensor) -> torch.Tensor
        Abstract method to decode latent tensor x
    forward(x: torch.Tensor) -> ModelOutput
        forward pass of model, fills in the reconstruction and latentspace attributes of ModelOutput class.
        For other implementations, additional information can be added to the ModelOutput class.

    """

    def __init__(
        self, config: Optional[Union[DefaultConfig, None]], input_dim: int
    ) -> None:
        """
        Parameters:
           config: Optional[Union[DefaultConfig, None]]
                Configuration object containing model parameters.
            input_dim: int
                Number of input features.
        Returns:
            None

        """
        super().__init__()
        if config is None:
            config = DefaultConfig()
        self.input_dim = input_dim
        self._encoder: nn.Module
        self._decoder: nn.Module
        self.config = config

    @abstractmethod
    def _build_network(self) -> None:
        """
        Builds the encoder and decoder networks for the autoencoder model.
        Populates the self._encoder and self._decoder attributes.

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
        pass

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
        pass

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
        """
        This weight inititalization method worked well in our experiments.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
