import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput
from autoencodix.base._base_autoencoder import BaseAutoencoder


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficiency"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ImageVAEArchitecture(BaseAutoencoder):
    """Optimized VAE architecture with channels-last memory format support"""

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        config: Optional[DefaultConfig],
        ontologies: Optional[Union[Tuple, Dict]] = None,
        feature_order: Optional[Union[Tuple, Dict]] = None,
        use_channels_last: bool = True,  # NEW: Enable channels-last by default
    ):
        if config is None:
            config = DefaultConfig()
        self._config: DefaultConfig = config
        super().__init__(config=config, input_dim=input_dim)
        self.input_dim: int = input_dim
        self.latent_dim: int = self._config.latent_dim
        self.nc, self.h, self.w = input_dim
        self.img_shape: Tuple[int, int, int] = input_dim
        self.hidden_dim: int = self._config.hidden_dim
        self.use_channels_last = use_channels_last
        self._build_network()
        self.apply(self._init_weights)

        # Convert model to channels_last memory format
        if self.use_channels_last:
            self._encoder = self._encoder.to(memory_format=torch.channels_last)
            self._decoder = self._decoder.to(memory_format=torch.channels_last)
            print("Model converted to channels_last memory format")

    def _build_network(self):
        """Construct optimized encoder and decoder networks."""

        # First layer uses standard conv, rest use depthwise separable
        # No BatchNorm for speed - bias=True since we removed BN
        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nc,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,  # Enable bias when no BatchNorm
            ),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim * 3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(
                in_channels=self.hidden_dim * 3,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.num__encoder_layers = 5
        self.spatial_dim = self.h // (2**self.num__encoder_layers)

        # Separate linear layers for mu and logvar
        self.mu = nn.Linear(
            self.hidden_dim * 4 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )
        self.logvar = nn.Linear(
            self.hidden_dim * 4 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )

        # Decoder entry
        self.d1 = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                self.hidden_dim * 4 * self.spatial_dim * self.spatial_dim,
            ),
            nn.ReLU(inplace=True),
        )

        # Decoder with optimized convolutions - no BatchNorm
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 3,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim,
                out_channels=self.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

    def _get_spatial_dim(self) -> int:
        return self.spatial_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input tensor x."""
        # Convert input to channels_last if enabled
        if self.use_channels_last and not x.is_contiguous(
            memory_format=torch.channels_last
        ):
            x = x.to(memory_format=torch.channels_last)

        h = self._encoder(x)
        h = h.view(-1, self.hidden_dim * 4 * self.spatial_dim * self.spatial_dim)

        # Separate computation for mu and logvar
        mu = self.mu(h)
        logvar = self.logvar(h)

        # Simpler, faster clamping
        logvar = logvar.clamp(-10, 10)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent space representation of the input."""
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the latent tensor x"""
        h = self.d1(x)
        h = h.view(-1, self.hidden_dim * 4, self.spatial_dim, self.spatial_dim)

        # Convert to channels_last if enabled
        if self.use_channels_last:
            h = h.to(memory_format=torch.channels_last)

        return self._decoder(h)

    def translate(self, z: torch.Tensor) -> torch.Tensor:
        """Reshapes the output to get actual images"""
        out = self.decode(z)
        return out.view(-1, *self.img_shape)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass of the model."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return ModelOutput(
            reconstruction=self.translate(z),
            latentspace=z,
            latent_mean=mu,
            latent_logvar=logvar,
            additional_info=None,
        )
