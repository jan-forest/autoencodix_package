# Modeling Module

## `Classifier`

Bases: `Module`

Multi-class classifier for adversarial training in n-modal latent space alignment.

Attributes:

| Name           | Type | Description                                       |
| -------------- | ---- | ------------------------------------------------- |
| `input_dim`    |      | Dimension of the input features.                  |
| `n_modalities` |      | Number of modalities (classes) to classify.       |
| `n_hidden`     |      | Number of hidden units in the classifier network. |

### `__init__(input_dim: int, n_modalities: int, n_hidden: int = 64) -> None`

Build a three-layer MLP classifier with dropout.

Parameters:

| Name           | Type  | Description                                                   | Default    |
| -------------- | ----- | ------------------------------------------------------------- | ---------- |
| `input_dim`    | `int` | Dimension of the input features (latent dimension).           | *required* |
| `n_modalities` | `int` | Number of modalities (output classes).                        | *required* |
| `n_hidden`     | `int` | Units in the first hidden layer; the second has half as many. | `64`       |

(LLM-generated docstring, pending human review.)

### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass through the classifier.

Parameters:

| Name | Type     | Description                                    | Default    |
| ---- | -------- | ---------------------------------------------- | ---------- |
| `x`  | `Tensor` | Input tensor of shape (batch_size, input_dim). | *required* |

Returns: Output tensor of shape (batch_size, n_modalities) representing class scores.

## `ImageVAEArchitecture`

Bases: `BaseAutoencoder`

This class defines a VAE, based on a CNN for images

It takes as input an image and of shape (C,W,H) and reconstructs it. We ensure to have a latent space of shape and img_in.shape = img_out.shape We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

So we need to calculate how the image dimension changes after each Convolution (we assume W=H) Applying the formular: W_out = (((W - kernel_size + 2padding)/stride) + 1) We get: W_out = (((W-4+2\*1)/2)+1) = = (W-2/2)+1 = = (2(0.5W-1)/2) +1 # factor 2 out = 0.5W - 1 + 1 W_out = 0.5W So in this configuration the output shape halfs after every convolutional step (assuming W=H)

Attributes:

| Name         | Type                   | Description                                                   |
| ------------ | ---------------------- | ------------------------------------------------------------- |
| `input_dim`  | `int`                  | (C,W,H) the input image shape                                 |
| `config`     |                        | Configuration object containing model architecture parameters |
| `_encoder`   | `Optional[Module]`     | Encoder network of the autoencoder                            |
| `_decoder`   | `Optional[Module]`     | Decoder network of the autoencoder                            |
| `latent_dim` | `int`                  | Dimension of the latent space                                 |
| `nc`         | `int`                  | number of channels in the input image                         |
| `h`          | `int`                  | height of the input image                                     |
| `w`          | `int`                  | width of the input image                                      |
| `img_shape`  | `Tuple[int, int, int]` | (C,W,H) the input image shape                                 |
| `hidden_dim` | `int`                  | number of filters in the first convolutional layer            |

### `__init__(input_dim: Tuple[int, int, int], config: Optional[DefaultConfig], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Initialize the ImageVAEArchitecture with the given configuration.

The number of filters in the first convolutional layer is taken from `config.hidden_dim`.

Parameters:

| Name            | Type                           | Description                                                                  | Default    |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- | ---------- |
| `input_dim`     | `Tuple[int, int, int]`         | (C, W, H) the input image shape.                                             | *required* |
| `config`        | `Optional[DefaultConfig]`      | Configuration object containing model parameters; defaults are used if None. | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x Args: x: Latent tensor Returns: Decoded tensor, reconstructed from the latent space

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encodes the input tensor x.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: The encoded latent space representation, or mu and logvar for VAEs.

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model. Args: x: Input tensor Returns: ModelOutput object containing the reconstructed tensor and latent tensor

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: Latent space representation

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE.

Parameters:

| Name     | Type     | Description                             | Default    |
| -------- | -------- | --------------------------------------- | ---------- |
| `mu`     | `Tensor` | mean of the latent distribution         | *required* |
| `logvar` | `Tensor` | log-variance of the latent distribution | *required* |

Returns: z: sampled latent vector

### `translate(z: torch.Tensor) -> torch.Tensor`

Reshapes the output to get actual images

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `z`  | `Tensor` | Latent tensor | *required* |

Returns: Reconstructed image of shape (C,W,H)

## `ImageVAEFastArchitecture`

Bases: `BaseAutoencoder`

This class defines a VAE, based on a CNN for images

It takes as input an image and of shape (C,W,H) and reconstructs it. We ensure to have a latent space of shape and img_in.shape = img_out.shape We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

So we need to calculate how the image dimension changes after each Convolution (we assume W=H) Applying the formular: W_out = (((W - kernel_size + 2padding)/stride) + 1) We get: W_out = (((W-4+2\*1)/2)+1) = = (W-2/2)+1 = = (2(0.5W-1)/2) +1 # factor 2 out = 0.5W - 1 + 1 W_out = 0.5W So in this configuration the output shape halfs after every convolutional step (assuming W=H)

Attributes:

| Name         | Type                   | Description                                                   |
| ------------ | ---------------------- | ------------------------------------------------------------- |
| `input_dim`  | `int`                  | (C,W,H) the input image shape                                 |
| `config`     |                        | Configuration object containing model architecture parameters |
| `_encoder`   | `Optional[Module]`     | Encoder network of the autoencoder                            |
| `_decoder`   | `Optional[Module]`     | Decoder network of the autoencoder                            |
| `latent_dim` | `int`                  | Dimension of the latent space                                 |
| `nc`         | `int`                  | number of channels in the input image                         |
| `h`          | `int`                  | height of the input image                                     |
| `w`          | `int`                  | width of the input image                                      |
| `img_shape`  | `Tuple[int, int, int]` | (C,W,H) the input image shape                                 |
| `hidden_dim` | `int`                  | number of filters in the first convolutional layer            |

### `__init__(input_dim: Tuple[int, int, int], config: Optional[DefaultConfig], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Initialize the ImageVAEFastArchitecture with the given configuration.

The number of filters in the first convolutional layer is taken from `config.hidden_dim`.

Parameters:

| Name            | Type                           | Description                                                                  | Default    |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- | ---------- |
| `input_dim`     | `Tuple[int, int, int]`         | (C, W, H) the input image shape.                                             | *required* |
| `config`        | `Optional[DefaultConfig]`      | Configuration object containing model parameters; defaults are used if None. | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x Args: x: Latent tensor Returns: Decoded tensor, reconstructed from the latent space

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encodes the input tensor x.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: The encoded latent space representation, or mu and logvar for VAEs.

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model. Args: x: Input tensor Returns: ModelOutput object containing the reconstructed tensor and latent tensor

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: Latent space representation

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE.

Parameters:

| Name     | Type     | Description                             | Default    |
| -------- | -------- | --------------------------------------- | ---------- |
| `mu`     | `Tensor` | mean of the latent distribution         | *required* |
| `logvar` | `Tensor` | log-variance of the latent distribution | *required* |

Returns: z: sampled latent vector

### `translate(z: torch.Tensor) -> torch.Tensor`

Reshapes the output to get actual images

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `z`  | `Tensor` | Latent tensor | *required* |

Returns: Reconstructed image of shape (C,W,H)

## `LayerFactory`

Factory for creating configurable neural network layers.

### `create_layer(in_features: int, out_features: int, dropout_p: float = 0.1, last_layer: bool = False) -> List[nn.Module]`

Create a configurable layer with optional components.

Parameters:

| Name           | Type    | Description                                                       | Default    |
| -------------- | ------- | ----------------------------------------------------------------- | ---------- |
| `in_features`  | `int`   | Input feature dimension                                           | *required* |
| `out_features` | `int`   | Output feature dimension                                          | *required* |
| `dropout_p`    | `float` | Dropout probability, by default 0.1                               | `0.1`      |
| `last_layer`   | `bool`  | Flag to skip activation/dropout for final layer, by default False | `False`    |

Returns:

| Type           | Description              |
| -------------- | ------------------------ |
| `List[Module]` | List of layer components |

### `create_maskix_layer(in_features: int, out_features: int, last_layer: bool)`

Create one block of the Maskix network (Linear, LayerNorm, Mish).

Parameters:

| Name           | Type   | Description                            | Default    |
| -------------- | ------ | -------------------------------------- | ---------- |
| `in_features`  | `int`  | Number of input features.              | *required* |
| `out_features` | `int`  | Number of output features.             | *required* |
| `last_layer`   | `bool` | If True, return only the linear layer. | *required* |

Returns:

| Type | Description                             |
| ---- | --------------------------------------- |
|      | list\[nn.Module\]: Layers of the block. |

(LLM-generated docstring, pending human review.)

### `get_layer_dimensions(feature_dim: int, latent_dim: int, n_layers: int, enc_factor: float) -> List[int]`

Calculate progressive layer dimensions.

Args: feature_dim: Input feature dimension latent_dim: Target latent dimension n_layers: Number of layers enc_factor: Reduction factor for layer sizes

Returns:

| Type        | Description                 |
| ----------- | --------------------------- |
| `List[int]` | Calculated layer dimensions |

## `MaskixArchitectureVanilla`

Bases: `BaseAutoencoder`

Masked Autoencoder Architecture that follows https://doi.org/10.1093/bioinformatics/btae020

To closely mimic the publication, the network is not build with our LayerFactory as in other architectures.

Attributes:

| Name        | Type                          | Description                                                   |
| ----------- | ----------------------------- | ------------------------------------------------------------- |
| `input_dim` | `Union[int, Tuple[int, ...]]` | number of input features                                      |
| `config`    |                               | Configuration object containing model architecture parameters |
| `encoder`   |                               | Encoder network of the autoencoder                            |
| `decoder`   |                               | Decoder network of the autoencoder                            |

### `__init__(config: Optional[DefaultConfig], input_dim: Union[int, Tuple[int, ...]], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Build the Maskix encoder, mask predictor and decoder.

Parameters:

| Name            | Type                           | Description                                                                                                 | Default    |
| --------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------- | ---------- |
| `config`        | `Optional[DefaultConfig]`      | Configuration (maskix_architecture, maskix_hidden_dim, latent_dim, drop_p, ...); defaults are used if None. | *required* |
| `input_dim`     | `Union[int, Tuple[int, ...]]`  | Number of input features.                                                                                   | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used.                                | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used.                                | `None`     |

Raises:

| Type        | Description                 |
| ----------- | --------------------------- |
| `TypeError` | If input_dim is not an int. |

(LLM-generated docstring, pending human review.)

### `decode(x: torch.Tensor) -> torch.Tensor`

Decodes the latent representation.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

### `encode(x: torch.Tensor) -> torch.Tensor`

Encodes the input data.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

### `forward(x: torch.Tensor) -> ModelOutput`

Encode the (masked) input, predict the mask and reconstruct the input.

Parameters:

| Name | Type     | Description                                                          | Default    |
| ---- | -------- | -------------------------------------------------------------------- | ---------- |
| `x`  | `Tensor` | Input batch (typically corrupted by the trainer's masking function). | *required* |

Returns:

| Type          | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `ModelOutput` | ModelOutput with reconstruction, latentspace and                 |
| `ModelOutput` | additional_info["predicted_mask"] (mask logits). Latent mean and |
| `ModelOutput` | log-variance are None, since Maskix is not variational.          |

(LLM-generated docstring, pending human review.)

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input data.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

## `OntixArchitecture`

Bases: `BaseAutoencoder`

Ontology Autoencoder implementation with separate encoder and decoder construction.

Attributes: input_dim: number of input features config: Configuration object containing model architecture parameters \_encoder: Encoder network of the autoencoder \_decoder: Decoder network of the autoencoder mu: Linear layer to compute the mean of the latent distribution logvar: Linear layer to compute the log-variance of the latent distribution masks: Tuple of weight masks for the decoder layers based on ontology latent_dim: Dimension of the latent space, inferred from the first mask ontologies: Ontology information. feature_order: Order of features for input data.

### `__init__(config: Optional[Union[None, DefaultConfig]], input_dim: int, ontologies: tuple, feature_order: list) -> None`

Initialize the Vanilla Autoencoder with the given configuration.

Parameters:

| Name            | Type                                   | Description                                       | Default    |
| --------------- | -------------------------------------- | ------------------------------------------------- | ---------- |
| `config`        | `Optional[Union[None, DefaultConfig]]` | Configuration object containing model parameters. | *required* |
| `input_dim`     | `int`                                  | Number of input features.                         | *required* |
| `ontologies`    | `tuple`                                | Ontology information.                             | *required* |
| `feature_order` | `list`                                 | Order of features for input data.                 | *required* |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `x`  | `Tensor` | Latent tensor | *required* |

Returns:

| Type     | Description    |
| -------- | -------------- |
| `Tensor` | torch.Tensor   |
| `Tensor` | Decoded tensor |

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encode the input tensor x

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type                    | Description    |
| ----------------------- | -------------- |
| `Tuple[Tensor, Tensor]` | Encoded tensor |

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model, fill

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type          | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `ModelOutput` | ModelOutput object containing the reconstructed tensor and latent tensor |

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type     | Description                 |
| -------- | --------------------------- |
| `Tensor` | Latent space representation |

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE

Parameters:

| Name     | Type     | Description         | Default    |
| -------- | -------- | ------------------- | ---------- |
| `mu`     | `Tensor` | Mean tensor         | *required* |
| `logvar` | `Tensor` | Log variance tensor | *required* |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Tensor` | Reparameterized latent tensor |

## `VanillixArchitecture`

Bases: `BaseAutoencoder`

Vanilla Autoencoder implementation with separate encoder and decoder construction.

Attributes:

| Name        | Type | Description                                                   |
| ----------- | ---- | ------------------------------------------------------------- |
| `input_dim` |      | number of input features                                      |
| `config`    |      | Configuration object containing model architecture parameters |
| `encoder`   |      | Encoder network of the autoencoder                            |
| `decoder`   |      | Decoder network of the autoencoder                            |

### `__init__(config: Optional[Union[None, DefaultConfig]], input_dim: int, ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None) -> None`

Initialize the Vanilla Autoencoder with the given configuration.

Parameters:

| Name            | Type                                   | Description                                                                  | Default    |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------------- | ---------- |
| `config`        | `Optional[Union[None, DefaultConfig]]` | Configuration object containing model parameters; defaults are used if None. | *required* |
| `input_dim`     | `int`                                  | Number of input features.                                                    | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]`         | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]`         | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decodes the latent representation.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

### `encode(x: torch.Tensor) -> torch.Tensor`

Encodes the input data.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: ModelOutput

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input data.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | input Tensor | *required* |

Returns: torch.Tensor

## `VarixArchitecture`

Bases: `BaseAutoencoder`

Variational Autoencoder implementation with separate encoder and decoder construction.

Attributes:

| Name        | Type  | Description                                                   |
| ----------- | ----- | ------------------------------------------------------------- |
| `input_dim` | `int` | number of input features                                      |
| `config`    |       | Configuration object containing model architecture parameters |
| `encoder`   |       | Encoder network of the autoencoder                            |
| `decoder`   |       | Decoder network of the autoencoder                            |
| `mu`        |       | Linear layer to compute the mean of the latent distribution   |

logvar: Linear layer to compute the log-variance of the latent distribution

### `__init__(config: Optional[Union[None, DefaultConfig]], input_dim: int, ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None) -> None`

Initialize the Variational Autoencoder with the given configuration.

Parameters:

| Name            | Type                                   | Description                                                                  | Default    |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------------- | ---------- |
| `config`        | `Optional[Union[None, DefaultConfig]]` | Configuration object containing model parameters; defaults are used if None. | *required* |
| `input_dim`     | `int`                                  | Number of input features.                                                    | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]`         | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]`         | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `x`  | `Tensor` | Latent tensor | *required* |

Returns:

| Type     | Description    |
| -------- | -------------- |
| `Tensor` | Decoded tensor |

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encode the input tensor x.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type                    | Description    |
| ----------------------- | -------------- |
| `Tuple[Tensor, Tensor]` | Encoded tensor |

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model, fill

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type          | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `ModelOutput` | ModelOutput object containing the reconstructed tensor and latent tensor |

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns:

| Type     | Description                 |
| -------- | --------------------------- |
| `Tensor` | Latent space representation |

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE

Parameters:

| Name     | Type     | Description  | Default    |
| -------- | -------- | ------------ | ---------- |
| `mu`     | `Tensor` | torch.Tensor | *required* |
| `logvar` | `Tensor` | torch.Tensor | *required* |

Returns:

| Type     | Description  |
| -------- | ------------ |
| `Tensor` | torch.Tensor |

## `VolumeVAEArchitecture`

Bases: `BaseAutoencoder`

This class defines a VAE, based on a CNN for three-dimensional images

It takes as input a 3D image of shape (C, D, H, W) and reconstructs it. We ensure to have a latent space of shape and img_in.shape = img_out.shape We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

We need to calculate how the volume dimensions change after each 3D convolution. In the 3D case, depth (D), height (H), and width (W) are treated separately, because they do not have to be equal.

For a Conv3d layer, the output size along each spatial axis is: X_out = (((X_in - kernel_size + 2 * padding) / stride) + 1) where X can be D, H, or W.

Thus, for each axis: X_out = ((X - 4 + 2 * 1) / 2) + 1 = ((X - 2) / 2) + 1 = X / 2

So in this configuration, each convolutional step halves every spatial dimension independently.

Example

input shape: (D, H, W) = (64, 96, 32) after 1 conv: (32, 48, 16) after 2 conv: (16, 24, 8) after 3 conv: ( 8, 12, 4) after 4 conv: ( 4, 6, 2) after 5 conv: ( 2, 3, 1)

Attributes:

| Name         | Type                        | Description                                                   |
| ------------ | --------------------------- | ------------------------------------------------------------- |
| `input_dim`  | `Tuple[int, int, int, int]` | (C, D, H, W) the input image shape                            |
| `config`     |                             | Configuration object containing model architecture parameters |
| `_encoder`   | `Optional[Module]`          | Encoder network of the autoencoder                            |
| `_decoder`   | `Optional[Module]`          | Decoder network of the autoencoder                            |
| `latent_dim` | `int`                       | Dimension of the latent space                                 |
| `nc`         | `int`                       | number of channels in the input image                         |
| `d`          | `int`                       | depth of the input image                                      |
| `h`          | `int`                       | height of the input image                                     |
| `w`          | `int`                       | width of the input image                                      |
| `img_shape`  | `Tuple[int, int, int, int]` | (C, D, H, W), the input image shape                           |
| `hidden_dim` | `int`                       | number of filters in the first convolutional layer            |

### `__init__(input_dim: Tuple[int, int, int, int], config: Optional[Imagix3DConfig], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Initialize the VolumeVAEArchitecture with the given configuration.

The number of filters in the first convolutional layer is taken from `config.hidden_dim`.

Parameters:

| Name            | Type                           | Description                                                                  | Default    |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- | ---------- |
| `input_dim`     | `Tuple[int, int, int, int]`    | (C, D, H, W) the input volume shape.                                         | *required* |
| `config`        | `Optional[Imagix3DConfig]`     | Configuration object containing model parameters; defaults are used if None. | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x Args: x: Latent tensor Returns: Decoded tensor, reconstructed from the latent space

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encodes the input tensor x.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: The encoded latent space representation, or mu and logvar for VAEs.

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model. Args: x: Input tensor Returns: ModelOutput object containing the reconstructed tensor and latent tensor

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: Latent space representation

### `make_norm_3d(num_channels: int) -> nn.Module`

Create a 3D normalization layer of the configured type.

Parameters:

| Name           | Type  | Description                      | Default    |
| -------------- | ----- | -------------------------------- | ---------- |
| `num_channels` | `int` | Number of channels to normalize. | *required* |

Returns:

| Type     | Description                                       |
| -------- | ------------------------------------------------- |
| `Module` | A BatchNorm3d, GroupNorm or InstanceNorm3d layer. |

Raises:

| Type         | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| `ValueError` | If the normalization type is not "batch", "group" or "instance". |

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE.

Parameters:

| Name     | Type     | Description                             | Default    |
| -------- | -------- | --------------------------------------- | ---------- |
| `mu`     | `Tensor` | mean of the latent distribution         | *required* |
| `logvar` | `Tensor` | log-variance of the latent distribution | *required* |

Returns: z: sampled latent vector

### `translate(z: torch.Tensor) -> torch.Tensor`

Reshapes the output to get actual images

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `z`  | `Tensor` | Latent tensor | *required* |

Returns: Reconstructed image of shape (C, D, H, W)

## `VolumeVAEFastArchitecture`

Bases: `BaseAutoencoder`

This class defines a VAE, based on a CNN for three-dimensional images

This architecture is identical to VolumeVAEArchitecture, apart from the fact that here:

- There is no Batch normalization
- Activation functions occur 'in place'

It takes as input a 3D image of shape (C, D, H, W) and reconstructs it. We ensure to have a latent space of shape and img_in.shape = img_out.shape We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-auto_encoders/tree/master)

We need to calculate how the volume dimensions change after each 3D convolution. In the 3D case, depth (D), height (H), and width (W) are treated separately, because they do not have to be equal.

For a Conv3d layer, the output size along each spatial axis is: X_out = (((X_in - kernel_size + 2 * padding) / stride) + 1) where X can be D, H, or W.

Thus, for each axis: X_out = ((X - 4 + 2 * 1) / 2) + 1 = ((X - 2) / 2) + 1 = X / 2

So in this configuration, each convolutional step halves every spatial dimension independently.

Example

input shape: (D, H, W) = (64, 96, 32) after 1 conv: (32, 48, 16) after 2 conv: (16, 24, 8) after 3 conv: ( 8, 12, 4) after 4 conv: ( 4, 6, 2) after 5 conv: ( 2, 3, 1)

Attributes:

| Name         | Type                        | Description                                                   |
| ------------ | --------------------------- | ------------------------------------------------------------- |
| `input_dim`  | `Tuple[int, int, int, int]` | (C, D, H, W) the input image shape                            |
| `config`     |                             | Configuration object containing model architecture parameters |
| `_encoder`   | `Optional[Module]`          | Encoder network of the autoencoder                            |
| `_decoder`   | `Optional[Module]`          | Decoder network of the autoencoder                            |
| `latent_dim` | `int`                       | Dimension of the latent space                                 |
| `nc`         | `int`                       | number of channels in the input image                         |
| `d`          | `int`                       | depth of the input image                                      |
| `h`          | `int`                       | height of the input image                                     |
| `w`          | `int`                       | width of the input image                                      |
| `img_shape`  | `Tuple[int, int, int, int]` | (C, D, H, W), the input image shape                           |
| `hidden_dim` | `int`                       | number of filters in the first convolutional layer            |

### `__init__(input_dim: Tuple[int, int, int, int], config: Optional[Imagix3DConfig], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Initialize the VolumeVAEFastArchitecture with the given configuration.

The number of filters in the first convolutional layer is taken from `config.hidden_dim`.

Parameters:

| Name            | Type                           | Description                                                                  | Default    |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- | ---------- |
| `input_dim`     | `Tuple[int, int, int, int]`    | (C, D, H, W) the input volume shape.                                         | *required* |
| `config`        | `Optional[Imagix3DConfig]`     | Configuration object containing model parameters; defaults are used if None. | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | Accepted for a uniform constructor interface across architectures; not used. | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode the latent tensor x Args: x: Latent tensor Returns: Decoded tensor, reconstructed from the latent space

### `encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Encodes the input tensor x.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: The encoded latent space representation, or mu and logvar for VAEs.

### `forward(x: torch.Tensor) -> ModelOutput`

Forward pass of the model. Args: x: Input tensor Returns: ModelOutput object containing the reconstructed tensor and latent tensor

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `x`  | `Tensor` | Input tensor | *required* |

Returns: Latent space representation

### `reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Reparameterization trick for VAE.

Parameters:

| Name     | Type     | Description                             | Default    |
| -------- | -------- | --------------------------------------- | ---------- |
| `mu`     | `Tensor` | mean of the latent distribution         | *required* |
| `logvar` | `Tensor` | log-variance of the latent distribution | *required* |

Returns: z: sampled latent vector

### `translate(z: torch.Tensor) -> torch.Tensor`

Reshapes the output to get actual images

Parameters:

| Name | Type     | Description   | Default    |
| ---- | -------- | ------------- | ---------- |
| `z`  | `Tensor` | Latent tensor | *required* |

Returns: Reconstructed image of shape (C, D, H, W)
