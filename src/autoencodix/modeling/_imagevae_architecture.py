import torch
import torch.nn as nn
from typing import Tuple, Optional
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput
from autoencodix.base._base_autoencoder import BaseAutoencoder


class ImageVAEArchitecture(BaseAutoencoder):
    """This class defines a VAE, based on a CNN for images

    It takes as input an image and of shape (C,W,H) and reconstructs it.
    We ensure to have a latent space of shape <batchsize,1,LatentDim> and img_in.shape = img_out.shape
    We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

    So we need to calculate how the image dimension changes after each Convolution (we assume W=H)
    Applying the formular:
        W_out = (((W - kernel_size + 2padding)/stride) + 1)
    We get:
        W_out = (((W-4+2*1)/2)+1) =
        = (W-2/2)+1 =
        = (2(0.5W-1)/2) +1 # factor 2 out
        = 0.5W - 1 + 1
        W_out = 0.5W
    So in this configuration the output shape halfs after every convolutional step (assuming W=H)


    Attributes:
        input_dim: (C,W,H) the input image shape
        config: Configuration object containing model architecture parameters
        _encoder: Encoder network of the autoencoder
        _decoder: Decoder network of the autoencoder
        latent_dim: Dimension of the latent space
        nc: number of channels in the input image
        h: height of the input image
        w: width of the input image
        img_shape: (C,W,H) the input image shape
        hidden_dim: number of filters in the first convolutional layer
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],  # (C,W,H) the input image shape
        config: Optional[DefaultConfig],
        # the input_dim is the number of channels in the image, e.g. 3
        hidden_dim: int = 16,
    ):
        """Initialize the ImageVAEArchitecture with the given configuration.

        Args:
            input_dim: (C,W,H) the input image shape
            config: Configuration object containing model parameters.
            hidden_dim: number of filters in the first convolutional layer
        """
        if config is None:
            config = DefaultConfig()
        self._config: DefaultConfig = config
        super().__init__(config=config, input_dim=input_dim)
        self.input_dim: int = input_dim
        self.latent_dim: int = self._config.latent_dim
        self.nc, self.h, self.w = input_dim
        self.img_shape: Tuple[int, int, int] = input_dim
        self.hidden_dim: int = hidden_dim
        self._build_network()
        self.apply(self._init_weights)

    def _build_network(self):
        """Construct the encoder and decoder networks."""
        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nc,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # to Calculate the image shape after the _encoder, we need to know the number of layers
        # because the shape halfs after every Conv2D layer
        self.num__encoder_layers = sum(
            1 for _ in self._encoder.children() if isinstance(_, nn.Conv2d)
        )
        # So the output shape after all layers is in_shape / 2**N_layers
        # We showed above in the DocString why the shape halfs
        self.spatial_dim = self.h // (2**self.num__encoder_layers)
        # In the Linear mu and logvar layer we need to flatten the 3D output to a 2D matrix
        # Therefore we need to multiply the size of every out diemension of the input layer to the Linear layers
        # This is hidden_dim * 8 (the number of filter/channel layer) * spatial dim (the widht of the image) * spatial diem (the height of the image)
        # assuimg width = height
        # The original paper had a fixed spatial dimension of 2, which only worked for images with 64x64 shape
        self.mu = nn.Linear(
            self.hidden_dim * 8 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )
        self.logvar = nn.Linear(
            self.hidden_dim * 8 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )

        # the same logic goes for the first _decoder layer, which takes the latent_dim as inshape
        # which is the outshape of the previous mu/logvar layer
        # and the shape of the first ConvTranspose2D layer is the last outpus shape of the _encoder layer
        # This the same multiplication as above
        self.d1 = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                self.hidden_dim * 8 * self.spatial_dim * self.spatial_dim,
            ),
            nn.ReLU(inplace=False),
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim,
                out_channels=self.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def _get_spatial_dim(self) -> int:
        return self.spatial_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input tensor x.

        Args:
            x: Input tensor
        Returns:
            The encoded latent space representation, or mu and logvar for VAEs.

        """
        h = self._encoder(x)
        # this makes sure we get the <batchsize, 1, latent_dim> shape for our latent space in the next step
        # because we put all dimensionaltiy in the second dimension of the output shape.
        # By covering all dimensionality here, we are sure that the rest is
        h = h.view(-1, self.hidden_dim * 8 * self.spatial_dim * self.spatial_dim)
        logvar = self.logvar(h)
        mu = self.mu(h)
        # prevent  mu and logvar from being too close to zero, this increased
        # numerical stability
        logvar = torch.clamp(logvar, 0.1, 20)
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
        # we ensure the correct dimension for the first Conv2DTranspose layer
        # so we make sure that the last 3 dimension are (n_filters, reduced_img_dim, reduced_img_dim)
        h = h.view(-1, self.hidden_dim * 8, self.spatial_dim, self.spatial_dim)
        return self._decoder(h)

    def translate(self, z: torch.Tensor) -> torch.Tensor:
        """Reshapes the output to get actual images

        Args:
            z: Latent tensor
        Returns:
            Reconstructed image of shape (C,W,H)
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
