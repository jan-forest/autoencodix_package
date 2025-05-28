from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig


class BaseAutoencoder(ABC, nn.Module):
    """Interface for building autoencoder models with standard interfaces.

    Defines standard methods for encoding data to a latent space and decoding
    back to the original space. Includes a weight initialization method for
    stable training. Intended to be extended by specific autoencoder variants
    like VAE.

    Attributes:
        input_dim: Number of input features.
        config: Configuration object containing model architecture parameters.
        _encoder: Encoder network.
        _decoder: Decoder network.
    """

    def __init__(self, config: Optional[DefaultConfig], input_dim: int):
        """Initializes the BaseAutoencoder.

        Args:
            config: Configuration object containing model parameters.
                If None, a default configuration will be used.
            input_dim: Number of input features.
        """
        super().__init__()
        if config is None:
            config = DefaultConfig()
        self.input_dim = input_dim
        self._encoder: Optional[nn.Module] = None
        self._decoder: Optional[nn.Module] = None
        self.config = config

    @abstractmethod
    def _build_network(self) -> None:
        """Builds the encoder and decoder networks for the autoencoder model.

        Populates the self._encoder and self._decoder attributes.
        This method should be implemented by subclasses to define
        the architecture of the encoder and decoder networks.
        """
        pass

    @abstractmethod
    def encode(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input into the latent space.

        Args:
            x: The input tensor to be encoded.

        Returns:
            The encoded latent space representation, or mu and logvar for VAE.
        """
        pass

    @abstractmethod
    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent space representation of the input.

        Method for unification of getting a latent space between Variational
        and Vanilla Autoencoders. This method is a wrapper around the encode
        method, or the reparameterization method for VAE.

        Args:
            x: The input tensor to be encoded.

        Returns:
            The latent space representation of the input tensor.
        """
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation back to the input space.

        Args:
            x: The latent tensor to be decoded.

        Returns:
            The decoded tensor, reconstructed from the latent space.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Combines encoding and decoding steps for the autoencoder.

        Args:
            x: The input tensor to be processed.

        Returns:
            The reconstructed input tensor and any additional information,
            depending on the model type.
        """
        pass

    def _init_weights(self, m):
        """Initializes weights using Xavier uniform initialization.

        This weight initialization method helps maintain the variance of
        activations across layers, preventing gradients from vanishing or
        exploding during training. This approach ensures stable and efficient
        training of the autoencoder model.

        Args:
            m: The module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
