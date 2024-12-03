import torch.nn as nn
from typing import List


class LayerFactory:
    """
    A factory class for creating fully connected layers and model architectures.

    This class provides methods to create fully connected layers, encoder, and decoder
    architectures for autoencoders.

    Methods
    -------
    create_fc_layer(in_dim, out_dim, drop_p=0.0, only_linear=False):
        Creates a fully connected layer with optional batch normalization, dropout, and activation.
    
    create_encoder(input_dim, latent_dim, n_layers, enc_factor, drop_p):
        Creates the encoder layers for an autoencoder architecture.

    create_decoder(input_dim, latent_dim, n_layers, enc_factor, drop_p):
        Creates the decoder layers for an autoencoder architecture.
    """

    @staticmethod
    def create_fc_layer(
        in_dim: int, out_dim: int, drop_p: float = 0.0, only_linear: bool = False
    ) -> List[nn.Module]:
        """
        Creates a fully connected layer with optional batch normalization, dropout, and activation.

        Parameters
        ----------
        in_dim : int
            The input dimension of the layer.
        out_dim : int
            The output dimension of the layer.
        drop_p : float, optional
            The dropout probability (default is 0.0).
        only_linear : bool, optional
            If True, only returns the linear layer without batch normalization or activation (default is False).

        Returns
        -------
        List[nn.Module]
            A list of `nn.Module` layers including the linear, batch normalization, dropout, and activation layers.
        """
        layers = [nn.Linear(in_dim, out_dim)]
        if not only_linear:
            layers += [nn.BatchNorm1d(out_dim), nn.Dropout(drop_p), nn.ReLU()]
        return layers

    @staticmethod
    def create_encoder(
        input_dim: int, latent_dim: int, n_layers: int, enc_factor: int, drop_p: float
    ) -> List[nn.Module]:
        """
        Creates encoder layers for an autoencoder architecture.

        Parameters
        ----------
        input_dim : int
            The input dimension of the encoder.
        latent_dim : int
            The latent dimension of the encoder.
        n_layers : int
            The number of layers in the encoder.
        enc_factor : int
            The factor by which the input dimension is reduced at each layer.
        drop_p : float
            The dropout probability.

        Returns
        -------
        List[nn.Module]
            A list of `nn.Module` layers representing the encoder architecture.
        """
        layers = []
        for i in range(n_layers):
            out_dim = max(latent_dim, input_dim // (enc_factor**i))
            layers.extend(LayerFactory.create_fc_layer(input_dim, out_dim, drop_p))
            input_dim = out_dim
        return layers

    @staticmethod
    def create_decoder(
        input_dim: int, latent_dim: int, n_layers: int, enc_factor: int, drop_p: float
    ) -> List[nn.Module]:
        """
        Creates decoder layers for an autoencoder architecture.

        Parameters
        ----------
        input_dim : int
            The input dimension of the decoder.
        latent_dim : int
            The latent dimension of the decoder.
        n_layers : int
            The number of layers in the decoder.
        enc_factor : int
            The factor by which the input dimension is increased at each layer.
        drop_p : float
            The dropout probability.

        Returns
        -------
        List[nn.Module]
            A list of `nn.Module` layers representing the decoder architecture.
        """
        layers = []
        for i in range(n_layers):
            out_dim = min(input_dim * (enc_factor**i), latent_dim)
            layers.extend(LayerFactory.create_fc_layer(input_dim, out_dim, drop_p))
            input_dim = out_dim
        return layers
