import torch
import torch.nn as nn
from autoencodix.configs import DefaultConfig
from typing import Optional, Union, Tuple, Dict
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput


class MaskixArchitectureVanilla(BaseAutoencoder):
    """Masked Autoencoder Architecture that follows https://doi.org/10.1093/bioinformatics/btae020

    To closely mimic the publication, the network is not build with our LayerFactory as in
    other architectures.

    Attributes:
        input_dim: number of input features
        config: Configuration object containing model architecture parameters
        encoder: Encoder network of the autoencoder
        decoder: Decoder network of the autoencoder


    """

    def __init__(
        self,
        config: Optional[DefaultConfig],
        input_dim: Union[int, Tuple[int, ...]],
        ontologies: Optional[Union[Tuple, Dict]] = None,
        feature_order: Optional[Union[Tuple, Dict]] = None,
    ):
        if config is None:
            config = DefaultConfig()
        self._config: DefaultConfig = config
        super().__init__(config, input_dim)
        self.input_dim: Union[int, Tuple[int, ...]] = input_dim
        if not isinstance(self.input_dim, int):
            raise TypeError(
                f"input dim needs to be int for MaskixArchitecture, got {type(self.input_dim)}"
            )
        self.latent_dim: int = self._config.latent_dim

        # populate self.encoder and self.decoder
        self._build_network()
        self.apply(self._init_weights)

    def _build_network(self):
        self._encoder = nn.Sequential(
            nn.Dropout(p=self.config.drop_p),
            nn.Linear(self.input_dim, self._config.maskix_hidden_dim),
            nn.LayerNorm(self._config.maskix_hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(self._config.maskix_hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self._mask_predictor = nn.Linear(self.latent_dim, self.input_dim)
        self._decoder = nn.Linear(
            in_features=self.latent_dim + self.input_dim, out_features=self.input_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input data.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self._encoder(x)

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent space representation of the input data.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation.

        Args:
            x: input Tensor
        Returns:
            torch.Tensor

        """
        return self._decoder(x)

    def forward_mask(self, x):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(torch.cat([latent, predicted_mask], dim=1))

        return latent, predicted_mask, reconstruction

    def forward(self, x: torch.Tensor) -> ModelOutput:
        latent: torch.Tensor = self.encode(x=x)
        predicted_mask: torch.Tensor = self._mask_predictor(latent)
        return ModelOutput(
            reconstruction=self.decode(torch.cat([latent, predicted_mask], dim=1)),
            latentspace=latent,
            latent_mean=None,
            latent_logvar=None,
            additional_info={"predicted_mask": predicted_mask},
        )
