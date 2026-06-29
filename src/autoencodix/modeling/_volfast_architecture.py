import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict
from autoencodix.configs import Imagix3DConfig
from autoencodix.utils import ModelOutput
from autoencodix.base._base_autoencoder import BaseAutoencoder


class VolumeVAEFastArchitecture(BaseAutoencoder):
    ## TODO rework the docstring
    """This class defines a VAE, based on a CNN for three-dimensional images
    
    This architecture is identical to VolumeVAEArchitecture, apart from the fact that here:
        - There is no Batch normalization
        - Activation functions occur 'in place'

    It takes as input a 3D image of shape (C, D, H, W) and reconstructs it.
    We ensure to have a latent space of shape <batchsize,1,LatentDim> and img_in.shape = img_out.shape
    We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

    We need to calculate how the volume dimensions change after each 3D convolution.
    In the 3D case, depth (D), height (H), and width (W) are treated separately,
    because they do not have to be equal.

    For a Conv3d layer, the output size along each spatial axis is:
        X_out = (((X_in - kernel_size + 2 * padding) / stride) + 1)
    where X can be D, H, or W.

    Thus, for each axis:
        X_out = ((X - 4 + 2 * 1) / 2) + 1
              = ((X - 2) / 2) + 1
              = X / 2

    So in this configuration, each convolutional step halves every spatial dimension independently.
    
    Example:
        input shape:  (D, H, W) = (64, 96, 32)
        after 1 conv: (32, 48, 16)
        after 2 conv: (16, 24,  8)
        after 3 conv: ( 8, 12,  4)
        after 4 conv: ( 4,  6,  2)
        after 5 conv: ( 2,  3,  1)

    Attributes:
        input_dim: (C, D, H, W) the input image shape
        config: Configuration object containing model architecture parameters
        _encoder: Encoder network of the autoencoder
        _decoder: Decoder network of the autoencoder
        latent_dim: Dimension of the latent space
        nc: number of channels in the input image
        d: depth of the input image
        h: height of the input image
        w: width of the input image
        img_shape: (C, D, H, W), the input image shape
        hidden_dim: number of filters in the first convolutional layer
        """

    def __init__(
        self,
        input_dim: Tuple[int, int, int, int],  # (C, D, H, W) the input image shape
        config: Optional[Imagix3DConfig],
        ontologies: Optional[Union[Tuple, Dict]] = None,
        feature_order: Optional[Union[Tuple, Dict]] = None,
        # the input_dim is the number of channels in the image, e.g. 4
    ):
        """Initialize the ImageVAEArchitecture with the given configuration.

        Args:
            input_dim: (C, D, H, W) the input image shape
            config: Configuration object containing model parameters.
            hidden_dim: number of filters in the first convolutional layer
        """
        if config is None:
            config = Imagix3DConfig()
        self._config: Imagix3DConfig = config
        super().__init__(
            config=config, 
            input_dim=input_dim,
            ontologies=ontologies,
            feature_order=feature_order)
        if self._config.n_conv_layers_3d != 5: # this is just a safety measure for now, as # of conv layers are hardcoded
            raise ValueError(
                "VolumeVAEArchitecture currently supports exactly 5 convolutional layers."
            )
        self.input_dim: Tuple[int, int, int, int] = input_dim
        self.latent_dim: int = self._config.latent_dim
        self.nc, self.d, self.h, self.w = input_dim
        self.img_shape: Tuple[int, int, int, int] = input_dim
        self.hidden_dim: int = self._config.hidden_dim
        self.clamp_logvar: bool = self._config.clamp_logvar
        self.logvar_range: Tuple[float, float] = self._config.logvar_range #type: ignore
        self.keep_mu_positive: bool = self._config.keep_mu_positive
        self._build_network()
        self.apply(self._init_weights)

    def _build_network(self) -> None:
        """Construct the encoder and decoder networks."""
        self._encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.nc,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # to Calculate the image shape after the _encoder, we need to know the number of layers
        # because the shape halfs after every Conv3D layer
        self.num__encoder_layers = sum(
            1 for _ in self._encoder.children() if isinstance(_, nn.Conv3d)
        )
        # So the output shape after all layers is in_shape / 2**N_layers
        # We showed above in the DocString why the shape halfs

        self.reduced_d = self.d // (2**self.num__encoder_layers)
        self.reduced_h = self.h // (2**self.num__encoder_layers)
        self.reduced_w = self.w // (2**self.num__encoder_layers)
        
        # The encoder output is a 5D tensor of shape:
        #   (batch_size, hidden_dim * 8, reduced_d, reduced_h, reduced_w)
        #
        # The subsequent mu and logvar layers are linear layers, so this encoder output
        # must first be flattened per sample into a 2D tensor of shape:
        #   (batch_size, hidden_dim * 8 * reduced_d * reduced_h * reduced_w)
        #
        # Therefore, the input size of the linear mu and logvar layers is:
        #   hidden_dim * 8 * reduced_d * reduced_h * reduced_w
        
        self.mu = nn.Linear(
            self.hidden_dim * 8 * self.reduced_d * self.reduced_h * self.reduced_w, self.latent_dim
        )
        self.logvar = nn.Linear(
            self.hidden_dim * 8 * self.reduced_d * self.reduced_h * self.reduced_w, self.latent_dim
        )

        # The same logic applies at the start of the decoder:
        # the latent vector of size latent_dim is first mapped back to
        #     hidden_dim * 8 * reduced_d * reduced_h * reduced_w
        # via a Linear layer.
        #
        # This reshaped tensor corresponds to the final spatial output shape of the encoder
        # and serves as the input to the first ConvTranspose3d layer.
        
        self.d1 = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                self.hidden_dim * 8 * self.reduced_d * self.reduced_h * self.reduced_w
            ),
            nn.ReLU(inplace=True),
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                in_channels=self.hidden_dim,
                out_channels=self.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def _get_spatial_dim(self) -> Tuple[int, int, int]:
        return self.reduced_d, self.reduced_h, self.reduced_w

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input tensor x.

        Args:
            x: Input tensor
        Returns:
            The encoded latent space representation, or mu and logvar for VAEs.

        """
        h = self._encoder(x) # type: ignore
        # this makes sure we get the <batchsize, 1, latent_dim> shape for our latent space in the next step
        # because we put all dimensionality in the second dimension of the output shape.
        # By covering all dimensionality here, we are sure that the rest is
        h = h.view(-1, self.hidden_dim * 8 * self.reduced_d * self.reduced_h * self.reduced_w)
        logvar = self.logvar(h)
        mu = self.mu(h)
        
        if self.clamp_logvar:
            # prevent  mu and logvar from being too close to zero, this increases numerical stability
            logvar = torch.clamp(logvar, self.logvar_range) #type: ignore
        
        if self.keep_mu_positive:
            # replace mu when mu < 0.00000001 with 0.1
            mu = torch.where(mu < 0.000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.

        Args:
             mu: mean of the latent distribution
             logvar: log-variance of the latent distribution
        Returns:
             z: sampled latent vector
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
        return self.reparameterize(mu, logvar)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the latent tensor x
        Args:
            x: Latent tensor
        Returns:
            Decoded tensor, reconstructed from the latent space
        """
        h = self.d1(x)
        # here we do a similar thing as in the _encoder,
        # but instead of ensuring the correct dimension for the latent space,
        # we ensure the correct dimension for the first Conv3DTranspose layer
        # so we make sure that the last 4 dimensions are (n_filters, reduced_d, reduced_h, reduced_w)
        h = h.view(-1, self.hidden_dim * 8, self.reduced_d, self.reduced_h, self.reduced_w)
        return self._decoder(h) # type: ignore

    def translate(self, z: torch.Tensor) -> torch.Tensor:
        """Reshapes the output to get actual images

        Args:
            z: Latent tensor
        Returns:
            Reconstructed image of shape (C, D, H, W)
        """
        out = self.decode(z)
        return out.view(-1, *self.img_shape)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass of the model.
        Args:
            x: Input tensor
        Returns:
            ModelOutput object containing the reconstructed tensor and latent tensor
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return ModelOutput(
            reconstruction=self.translate(z),
            latentspace=z,
            latent_mean=mu,
            latent_logvar=logvar,
            additional_info=None,
        )
