from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseAutoencoder(ABC, nn.Module):
    """
    Abstract BaseAutoencoder class defining the required methods for an autoencoder.
    Provides an interface for weight initialization that subclasses should implement.

    This class inherits from `torch.nn.Module` and is intended to be extended
    by specific autoencoder models.
    """

    def __init__(self):
        """
        Initializes the BaseAutoencoder class.
        """
        super().__init__()

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines encoding and decoding steps for the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be processed.

        Returns
        -------
        torch.Tensor
            The reconstructed input tensor after encoding and decoding.
        """
        pass

    @abstractmethod
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for the model layers. This method should be implemented 
        by subclasses to define how weights are initialized for each layer type.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module whose weights are to be initialized.
        """
        pass
