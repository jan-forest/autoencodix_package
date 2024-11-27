from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def fc_layer_1d(
    in_dim: int,
    out_dim: int,
    drop_p: float = 0.0,
    only_linear: bool = False,
) -> List[nn.Module]:
    """
    Create a fully connected layer with optional batch normalization, dropout, and activation.

    Args:
        in_dim: Input dimension of the layer
        out_dim: Output dimension of the layer
        drop_p: Dropout probability
        only_linear: If True, only create a linear layer without additional modules

    Returns:
        List of PyTorch modules forming the layer
    """
    if not only_linear:
        fc_layer: List[nn.Module] = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_p),
            nn.ReLU(),
        ]
    else:
        fc_layer = [nn.Linear(in_dim, out_dim)]
    return fc_layer


def get_layer_dim(
    feature_dim: int,
    latent_dim: int,
    n_layers: int,
    enc_factor: int,
) -> List[int]:
    """
    Calculate layer dimensions for encoder/decoder.

    Args:
        feature_dim: Initial input dimension
        latent_dim: Target latent dimension
        n_layers: Number of layers
        enc_factor: Encoding factor for dimension reduction

    Returns:
        List of layer dimensions
    """
    layer_dimensions: List[int] = [feature_dim]
    for _ in range(n_layers - 1):
        prev_layer_size = layer_dimensions[-1]
        next_layer_size = max(int(prev_layer_size / enc_factor), latent_dim)
        layer_dimensions.append(next_layer_size)
    layer_dimensions.append(latent_dim)
    return layer_dimensions


class Vanillix(nn.Module):
    """A class to define a Vanilla Autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        global_p: float = 0.1,
        n_layers: int = 2,
        enc_factor: int = 4,
    ) -> None:
        """
        Initialize the Vanilla Autoencoder.

        Args:
            input_dim: Dimension of the input features
            latent_dim: Dimension of the latent space
            global_p: Global dropout probability
            n_layers: Number of layers in encoder/decoder
            enc_factor: Factor for reducing layer dimensions
        """
        super().__init__()

        # Validate input dimensions
        if input_dim <= 4:
            raise ValueError("input_dim must be greater than 4")

        # Model attributes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.global_p = global_p
        self.n_layers = n_layers
        self.enc_factor = enc_factor

        # Building architecture
        enc_dim = get_layer_dim(
            input_dim,
            latent_dim,
            n_layers,
            enc_factor,
        )

        # Create encoder layers
        encoder_layers: List[nn.Module] = []
        for i in range(len(enc_dim) - 2):
            encoder_layers.extend(
                fc_layer_1d(enc_dim[i], enc_dim[i + 1], self.global_p),
            )
        encoder_layers.extend(
            fc_layer_1d(enc_dim[-2], enc_dim[-1], self.global_p, only_linear=True),
        )

        # Create decoder layers
        dec_dim = enc_dim[::-1]
        decoder_layers: List[nn.Module] = []
        for i in range(len(dec_dim) - 2):
            decoder_layers.extend(
                fc_layer_1d(dec_dim[i], dec_dim[i + 1], self.global_p),
            )
        decoder_layers.extend(
            fc_layer_1d(dec_dim[-2], dec_dim[-1], self.global_p, only_linear=True),
        )

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for linear layers.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to latent space.

        Args:
            x: Input tensor

        Returns:
            Encoded tensor
        """
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to original space.

        Args:
            x: Latent representation tensor

        Returns:
            Reconstructed tensor
        """
        return self.decoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        latent, _ = self.encode(x)
        recon = self.decode(latent)
        return recon