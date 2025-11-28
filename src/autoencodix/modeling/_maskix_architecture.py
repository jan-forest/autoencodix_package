import torch
import torch.nn as nn
from autoencodix.configs import DefaultConfig
from autoencodix.modeling._layer_factory import LayerFactory
from typing import Optional, Union, Tuple, Dict, List
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
        if self._config.maskix_architecture == "scMAE":
            self._build_scMAE()
        elif self._config.maskix_architecture == "custom":
            self._build_custom()
        else:
            raise ValueError(
                f"Got {self.config.maskix_architecture}, but expected 'scMAE' or 'custom'"
                "This happens if you allow a new value in DefaultConfig but did not implement it here."
            )

    def _build_custom(self):
        self._mask_predictor = nn.Linear(self.latent_dim, self.input_dim)  # ty: ignore

        enc_dim = LayerFactory.get_layer_dimensions(
            feature_dim=self.input_dim,  # ty: ignore
            latent_dim=self._config.latent_dim,
            n_layers=self._config.n_layers,
            enc_factor=self._config.enc_factor,
        )
        first_layer = nn.Dropout(p=self.config.drop_p)

        encoder_layers: List[nn.Module] = []
        if self._config.n_layers == 0:
            self._encoder = nn.Sequential(
                nn.Dropout(p=self.config.drop_p),
                nn.Linear(self.input_dim, self.latent_dim),  # ty: ignore
            )
        # print(enc_dim)
        for i, (in_features, out_features) in enumerate(zip(enc_dim[:-1], enc_dim[1:])):
            last_layer = i == len(enc_dim) - 2
            encoder_layers.extend(
                LayerFactory.create_maskix_layer(
                    in_features=in_features,
                    out_features=out_features,
                    last_layer=last_layer,
                )
            )

        self._encoder = nn.Sequential(first_layer, *encoder_layers)

        dec_dimensions = enc_dim[::-1]  # Reverse the dimensions and copy
        decoder_layers: List[nn.Module] = []
        for i, (in_features, out_features) in enumerate(
            zip(dec_dimensions[:-1], dec_dimensions[1:])
        ):
            last_layer = i == len(dec_dimensions) - 2
            decoder_layers.extend(
                LayerFactory.create_maskix_layer(
                    in_features=in_features,
                    out_features=out_features,
                    last_layer=last_layer,
                )
            )

        latent_layer = nn.Linear(
            in_features=self.latent_dim + self.input_dim,  # ty: ignore
            out_features=dec_dimensions[0],
        )  # ty: ignore
        self._decoder = nn.Sequential(latent_layer, *decoder_layers)

    def _build_scMAE(self):
        self._encoder = nn.Sequential(
            nn.Dropout(p=self.config.drop_p),
            nn.Linear(self.input_dim, self._config.maskix_hidden_dim),  # ty: ignore
            nn.LayerNorm(self._config.maskix_hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(self._config.maskix_hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self._mask_predictor = nn.Linear(self.latent_dim, self.input_dim)  # ty: ignore
        self._decoder = nn.Linear(
            in_features=self.latent_dim + self.input_dim,  # ty: ignore
            out_features=self.input_dim,  # ty: ignore
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
