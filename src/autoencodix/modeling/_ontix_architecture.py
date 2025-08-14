from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig

from ._layer_factory import LayerFactory


class OntixArchitecture(BaseAutoencoder):
    """
    Vanilla Autoencoder implementation with separate encoder and decoder construction.

    Attributes
    ----------
    self.input_dim : int
        number of input features
    self.config: DefaultConfig
        Configuration object containing model architecture parameters
    self._encoder: nn.Module
        Encoder network of the autoencoder
    self._decoder: nn.Module
        Decoder network of the autoencoder

    Methods
    -------
    _build_network()
        Construct the encoder and decoder networks via the LayerFactory
    encode(x: torch.Tensor) -> torch.Tensor
        Encode the input tensor x
    decode(x: torch.Tensor) -> torch.Tensor
        Decode the latent tensor x
    forward(x: torch.Tensor) -> ModelOutput
        Forward pass of the model, fills in the reconstruction and latentspace attributes of ModelOutput class.
    reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor
        Reparameterization trick for VAE
    """

    def __init__(
        self,
        config: Optional[Union[None, DefaultConfig]],
        input_dim: int,
        ontologies: tuple,
        feature_order: list,
    ) -> None:
        """
        Initialize the Vanilla Autoencoder with the given configuration.

        Parameters
        ----------
        config : Optional[Union[None, DefaultConfig]]
            Configuration object containing model parameters.
        """
        if config is None:
            config = DefaultConfig()
        self._config = config
        super().__init__(config, input_dim)
        self.input_dim = input_dim
        self._mu: nn.Module
        self._logvar: nn.Module
        # create masks for sparse decoder
        self.ontologies = ontologies
        self.feature_order = feature_order
        self.masks = self._make_masks(config=self._config, feature_order=feature_order)
        self.latent_dim = self.masks[0].shape[1]
        print("Latent Dim: " + str(self.latent_dim))
        # populate self.encoder and self.decoder
        self._build_network()
        self.apply(self._init_weights)
        self._decoder.apply(
            self._positive_dec
        )  # Sparse decoder only has positive weights

        # Apply weight mask to create ontology-based decoder
        with torch.no_grad():
            # Check that the decoder has the same number of layers as masks
            if len(self.masks) != len(self._decoder):
                print(len(self.masks), len(self._decoder))
                print(self._encoder)
                print(self._decoder)
                raise ValueError(
                    "Number of masks does not match number of decoder layers"
                )
            else:
                for i, mask in enumerate(self.masks):
                    self._decoder[i].weight.mul_(mask)

    def _make_masks(
        self, config: DefaultConfig, feature_order: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for sparse decoder based on ontology via config

        Parameters
        ----------
        config : DefaultConfig
            Configuration object containing model parameters

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the masks for the decoder network

        """
        # Read ontology from config

        masks = tuple()
        # feature_names are all values in the last ontology layer
        all_feature_names = set()
        for key, values in self.ontologies[-1].items():
            all_feature_names.update(values)
        all_feature_names = list(all_feature_names)
        print("Ontix checks:")
        print(f"All possible feature names length: {len(all_feature_names)}")
        print(f"Feature order length: {len(feature_order)}")
        # Check if all features in feature_order are present in all_feature_names
        feature_names = [f for f in feature_order]
        missing_features = [f for f in feature_order if f not in all_feature_names]
        if missing_features:
            print(
                f"Features in feature_order not found in all_feature_names: {missing_features}"
            )
        print(f"Feature names without filtering: {len(feature_names)}")

        # Enumerate through the ontologies
        for x, ont_dic in enumerate(self.ontologies):
            prev_lay_dim = len(ont_dic.keys())

            if x == len(self.ontologies) - 1:
                # fixed sort of feature list
                node_list = feature_names
            else:
                node_list = list(self.ontologies[x + 1].keys())
            next_lay_dim = len(node_list)
            # create masks for sparse decoder
            mask = torch.zeros(next_lay_dim, prev_lay_dim)
            p_int = 0
            if len(node_list) == next_lay_dim:
                if len(ont_dic.keys()) == prev_lay_dim:
                    for p_id in ont_dic:
                        feature_list = ont_dic[p_id]
                        for f_id in feature_list:
                            if f_id in node_list:
                                f_int = node_list.index(f_id)
                                mask[f_int, p_int] = 1

                        p_int += 1
                else:
                    print(
                        "Mask layer cannot be calculated. Ontology key list does not match previous layer dimension"
                    )
                    print("Returning zero mask")
            else:
                print(f"node list: {len(node_list)} vs next_lay_dim:{next_lay_dim}")
                print(
                    "Mask layer cannot be calculated. Output layer list does not match next layer dimension"
                )
                print("Returning zero mask")
            print(
                f"Mask layer {x} with shape {mask.shape} and {torch.sum(mask)} connections"
            )
            masks += (mask,)

        if torch.max(mask) < 1:
            print(
                "You provided an ontology with no connections between layers in the decoder. Please check your ontology definition."
            )

        return masks

    def _build_network(self) -> None:
        """
        Construct the encoder and decoder networks.

        Handles cases where `n_layers=0` by skipping the encoder and using only mu/logvar.
        """
        #### Encoder copied from varix architecture ####
        enc_dim = LayerFactory.get_layer_dimensions(
            feature_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_layers=self._config.n_layers,
            enc_factor=self._config.enc_factor,
        )
        #

        # Case 1: No Hidden Layers (Direct Mapping)
        self._encoder = nn.Sequential()
        self._mu = nn.Linear(self.input_dim, self.latent_dim)
        self._logvar = nn.Linear(self.input_dim, self.latent_dim)

        # Case 2: At Least One Hidden Layer
        if self._config.n_layers > 0:
            encoder_layers = []
            # print(enc_dim)
            for i, (in_features, out_features) in enumerate(
                zip(enc_dim[:-1], enc_dim[1:])
            ):
                # since we add mu and logvar, we will remove the last layer
                if i == len(enc_dim) - 2:
                    break
                encoder_layers.extend(
                    LayerFactory.create_layer(
                        in_features=in_features,
                        out_features=out_features,
                        dropout_p=self._config.drop_p,
                        last_layer=False,  # only for decoder relevant
                    )
                )

            self._encoder = nn.Sequential(*encoder_layers)
            self._mu = nn.Linear(enc_dim[-2], self.latent_dim)
            self._logvar = nn.Linear(enc_dim[-2], self.latent_dim)
        #### Encoder copied from varix architecture ####

        # Construct Decoder with Sparse Connections via masks
        # Decoder dimension is determined by the masks
        dec_dim = [self.latent_dim] + [
            mask.shape[0] for mask in self.masks
        ]  # + [self.input_dim]
        decoder_layers = []
        for i, (in_features, out_features) in enumerate(zip(dec_dim[:-1], dec_dim[1:])):
            # last_layer = i == len(dec_dim) - 2
            last_layer = True  ## Only linear layers in sparse decoder
            decoder_layers.extend(
                LayerFactory.create_layer(
                    in_features=in_features,
                    out_features=out_features,
                    dropout_p=0,  ## No dropout in sparse decoder
                    last_layer=last_layer,
                    # only_linear=True,
                )
            )

        self._decoder = nn.Sequential(*decoder_layers)

    def _positive_dec(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data = m.weight.data.clamp(min=0)

    ## Defined in the base class
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)

    ### Copy from Varix Architecture ###
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input tensor x

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Encoded tensor

        """
        latent = x  # for case where n_layers=0
        if len(self._encoder) > 0:
            latent = self._encoder(x)
        mu = self._mu(latent)
        logvar = self._logvar(latent)
        # numeric stability
        logvar = torch.clamp(logvar, 0.01, 20)
        mu = torch.where(mu < 0.0000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE

        Parameters:
            mu : torch.Tensor
            logvar : torch.Tensor

        Returns:
            torch.Tensor

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent tensor x

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor

        Returns
        -------
        torch.Tensor
            Decoded tensor

        """
        # print(self._decoder)
        # print(x.shape)
        return self._decoder(x)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the model, fill

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        ModelOutput
            ModelOutput object containing the reconstructed tensor and latent tensor

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return ModelOutput(
            reconstruction=x_hat,
            latentspace=z,
            latent_mean=mu,
            latent_logvar=logvar,
            additional_info=None,
        )

    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Latent space representation

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
