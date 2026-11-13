# Base Module

## `BaseAutoencoder`

Bases: `ABC`, `Module`

Interface for building autoencoder models.

Defines standard methods for encoding data to a latent space and decoding back to the original space. Includes a weight initialization method for stable training. Intended to be extended by specific autoencoder variants like VAE.

Attributes:

| Name            | Type               | Description                                                    |
| --------------- | ------------------ | -------------------------------------------------------------- |
| `input_dim`     |                    | Number of input features.                                      |
| `config`        |                    | Configuration object containing model architecture parameters. |
| `_encoder`      | `Optional[Module]` | Encoder network.                                               |
| `_decoder`      | `Optional[Module]` | Decoder network.                                               |
| `ontologies`    |                    | Ontology information, if provided for Ontix                    |
| `feature_order` |                    | For Ontix                                                      |

### `__init__(config: Optional[DefaultConfig], input_dim: Union[int, Tuple[int, ...]], ontologies: Optional[Union[Tuple, Dict]] = None, feature_order: Optional[Union[Tuple, Dict]] = None)`

Initializes the BaseAutoencoder.

Parameters:

| Name            | Type                           | Description                                                                                      | Default    |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------ | ---------- |
| `config`        | `Optional[DefaultConfig]`      | Configuration object containing model parameters. If None, a default configuration will be used. | *required* |
| `input_dim`     | `Union[int, Tuple[int, ...]]`  | Number of input features.                                                                        | *required* |
| `ontologies`    | `Optional[Union[Tuple, Dict]]` | Ontology information, if provided for Ontix                                                      | `None`     |
| `feature_order` | `Optional[Union[Tuple, Dict]]` | For Ontix                                                                                        | `None`     |

### `decode(x: torch.Tensor) -> torch.Tensor`

Decodes the latent representation back to the input space.

Parameters:

| Name | Type     | Description                      | Default    |
| ---- | -------- | -------------------------------- | ---------- |
| `x`  | `Tensor` | The latent tensor to be decoded. | *required* |

Returns:

| Type     | Description                                              |
| -------- | -------------------------------------------------------- |
| `Tensor` | The decoded tensor, reconstructed from the latent space. |

### `encode(x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Encodes the input into the latent space.

Parameters:

| Name | Type     | Description                     | Default    |
| ---- | -------- | ------------------------------- | ---------- |
| `x`  | `Tensor` | The input tensor to be encoded. | *required* |

Returns:

| Type                                   | Description                                                         |
| -------------------------------------- | ------------------------------------------------------------------- |
| `Union[Tensor, Tuple[Tensor, Tensor]]` | The encoded latent space representation, or mu and logvar for VAEs. |

### `forward(x: torch.Tensor) -> ModelOutput`

Combines encoding and decoding steps for the autoencoder.

Parameters:

| Name | Type     | Description                       | Default    |
| ---- | -------- | --------------------------------- | ---------- |
| `x`  | `Tensor` | The input tensor to be processed. | *required* |

Returns:

| Type          | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `ModelOutput` | The reconstructed input tensor and any additional information, |
| `ModelOutput` | depending on the model type.                                   |

### `get_latent_space(x: torch.Tensor) -> torch.Tensor`

Returns the latent space representation of the input.

Method for unification of getting a latent space between Variational and Vanilla Autoencoders. This method is a wrapper around the encode method, or the reparameterization method for VAE.

Parameters:

| Name | Type     | Description                     | Default    |
| ---- | -------- | ------------------------------- | ---------- |
| `x`  | `Tensor` | The input tensor to be encoded. | *required* |

Returns:

| Type     | Description                                          |
| -------- | ---------------------------------------------------- |
| `Tensor` | The latent space representation of the input tensor. |

## `BaseDataset`

Bases: `ABC`, `Dataset`

Interface to guide implementation for custom PyTorch datasets.

Attributes:

| Name          | Type   | Description                                                          |
| ------------- | ------ | -------------------------------------------------------------------- |
| `data`        |        | The dataset content (can be a torch.Tensor or other data structure). |
| `config`      |        | Optional configuration object.                                       |
| `sample_ids`  |        | Optional list of identifiers for each sample.                        |
| `feature_ids` |        | Optional list of identifiers for each feature.                       |
| `mytype`      | `Enum` | Enum indicating the dataset type (should be set in subclasses).      |

### `__init__(data: Union[torch.Tensor, List[ImgData], sp.sparse.spmatrix], config: Optional[Any] = None, sample_ids: Optional[List[Any]] = None, feature_ids: Optional[List[Any]] = None)`

Initializes the dataset.

Subclasses must set `self.mytype` to a `DataSetTypes` value.

Parameters:

| Name          | Type                                     | Description                            | Default    |
| ------------- | ---------------------------------------- | -------------------------------------- | ---------- |
| `data`        | `Union[Tensor, List[ImgData], spmatrix]` | The data to be used by the dataset.    | *required* |
| `config`      | `Optional[Any]`                          | Optional configuration parameters.     | `None`     |
| `sample_ids`  | `Optional[List[Any]]`                    | Optional identifiers for each sample.  | `None`     |
| `feature_ids` | `Optional[List[Any]]`                    | Optional identifiers for each feature. | `None`     |

### `__len__() -> int`

Returns the number of samples in the dataset.

Returns:

| Type  | Description                           |
| ----- | ------------------------------------- |
| `int` | The number of samples in the dataset. |

### `get_input_dim() -> Union[int, Tuple[int, ...]]`

Gets the input dimension of the dataset (n_features)

Returns:

| Type                          | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `Union[int, Tuple[int, ...]]` | The input dimension of the dataset's feature space. |

## `BaseEvaluator`

Bases: `ABC`

Interface for evaluating learned embeddings on downstream ML tasks.

Subclasses implement `evaluate`, which trains machine-learning models on latent spaces (and reference embeddings) and stores the scores in the `Result`. Used by `BasePipeline.evaluate`.

(LLM-generated docstring, pending human review.)

### `evaluate(*args)`

Evaluate the Autoencodix pipeline on defined machine learning tasks.

Subclasses must implement this method to perform evaluation using the provided arguments.

Parameters:

| Name    | Type | Description                                              | Default |
| ------- | ---- | -------------------------------------------------------- | ------- |
| `*args` |      | Variable length argument list for evaluation parameters. | `()`    |

Returns:

| Name     | Type | Description            |
| -------- | ---- | ---------------------- |
| `Result` |      | The evaluation result. |

## `BaseLoss`

Bases: `Module`, `ABC`

Provides common loss computation functionality for autoencoders.

Implements standard loss calculations including reconstruction loss, KL divergence, and Maximum Mean Discrepancy (MMD), while requiring subclasses to implement the specific forward method.

Attributes:

| Name                  | Type     | Description                                            |
| --------------------- | -------- | ------------------------------------------------------ |
| `config`              |          | Configuration parameters for the loss function.        |
| `recon_loss`          | `Module` | Module for computing reconstruction loss (MSE or BCE). |
| `reduction_fn`        |          | Function to apply reduction (mean or sum).             |
| `compute_kernel`      |          | Function to compute kernel for MMD loss.               |
| `annealing_scheduler` |          | Helper for loss calculation with annealing.            |

### `__init__(config: DefaultConfig, annealing_scheduler=None)`

Initializes the loss module with the specified configuration.

Parameters:

| Name                  | Type            | Description                                       | Default    |
| --------------------- | --------------- | ------------------------------------------------- | ---------- |
| `config`              | `DefaultConfig` | Configuration parameters for the loss function.   | *required* |
| `annealing_scheduler` |                 | Helper class for loss calculation with annealing. | `None`     |

Raises:

| Type                  | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| `NotImplementedError` | If unsupported loss reduction or reconstruction loss type is specified. |

### `compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`

Computes KL divergence loss between N(mu, logvar) and N(0, 1).

Parameters:

| Name     | Type     | Description          | Default    |
| -------- | -------- | -------------------- | ---------- |
| `mu`     | `Tensor` | Mean tensor.         | *required* |
| `logvar` | `Tensor` | Log variance tensor. | *required* |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Tensor` | The KL divergence loss value. |

Raises:

| Type         | Description                                  |
| ------------ | -------------------------------------------- |
| `ValueError` | If mu and logvar do not have the same shape. |

### `compute_mmd_loss(z: torch.Tensor, true_samples: torch.Tensor) -> torch.Tensor`

Computes Maximum Mean Discrepancy loss.

Parameters:

| Name           | Type     | Description                            | Default    |
| -------------- | -------- | -------------------------------------- | ---------- |
| `z`            | `Tensor` | Samples from the encoded distribution. | *required* |
| `true_samples` | `Tensor` | Samples from the prior distribution.   | *required* |

Returns:

| Type     | Description         |
| -------- | ------------------- |
| `Tensor` | The MMD loss value. |

Raises:

| Type                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| `NotImplementedError` | If unsupported loss reduction type is specified. |

### `compute_paired_loss(latentspaces: dict[str, torch.Tensor], sample_ids: dict[str, list]) -> torch.Tensor`

Calculates the paired distance loss across all pairs of modalities in a batch.

Parameters:

| Name           | Type                | Description                                                                                                       | Default    |
| -------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------- |
| `latentspaces` | `dict[str, Tensor]` | A dictionary mapping modality names to their latent space tensors. e.g., {'RNA': tensor_rna, 'ATAC': tensor_atac} | *required* |
| `sample_ids`   | `dict[str, list]`   | A dictionary mapping modality names to their list of sample IDs.                                                  | *required* |

Returns:

| Type     | Description                                                |
| -------- | ---------------------------------------------------------- |
| `Tensor` | A single scalar tensor representing the total paired loss. |

### `compute_variational_loss(mu: Optional[torch.Tensor], logvar: Optional[torch.Tensor], z: Optional[torch.Tensor] = None, true_samples: Optional[torch.Tensor] = None) -> torch.Tensor`

Computes either KL or MMD loss based on configuration.

Parameters:

| Name           | Type               | Description                               | Default    |
| -------------- | ------------------ | ----------------------------------------- | ---------- |
| `mu`           | `Optional[Tensor]` | Mean tensor for variational loss.         | *required* |
| `logvar`       | `Optional[Tensor]` | Log variance tensor for variational loss. | *required* |
| `z`            | `Optional[Tensor]` | Encoded samples for MMD loss.             | `None`     |
| `true_samples` | `Optional[Tensor]` | Prior samples for MMD loss.               | `None`     |

Returns:

| Type     | Description                    |
| -------- | ------------------------------ |
| `Tensor` | The computed variational loss. |

Raises:

| Type                  | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `ValueError`          | If required parameters are missing or if mu and logvar have shape mismatch. |
| `NotImplementedError` | If unsupported VAE loss type is specified.                                  |

### `forward(*args, **kwargs) -> Any`

Calculates the loss for the autoencoder.

This method must be implemented by subclasses to define the specific loss computation logic for the autoencoder. The implementation should compute the total loss as well as any individual loss components (e.g., reconstruction loss, KL divergence, etc.) based on the model's output and the provided targets.

Parameters:

| Name       | Type | Description                                                                                                       | Default |
| ---------- | ---- | ----------------------------------------------------------------------------------------------------------------- | ------- |
| `*args`    |      | Positional arguments, depending on the loss type and pipeline (typically the model output and the target tensor). | `()`    |
| `**kwargs` |      | Keyword arguments, depending on the loss type and pipeline (e.g. epoch for annealing).                            | `{}`    |

Returns:

| Type  | Description                                                             |
| ----- | ----------------------------------------------------------------------- |
| `Any` | Typically a tuple of the total loss as a scalar tensor and a dictionary |
| `Any` | of individual loss components, where the keys are descriptive strings   |
| `Any` | (e.g. "reconstruction_loss", "kl_loss") and the values are the          |
| `Any` | corresponding loss tensors. Implementation in subclasses is flexible,   |
| `Any` | so for new loss classes this can differ.                                |

Note

Subclasses must implement this method to define the specific loss computation logic for their use case.

## `BasePipeline`

Bases: `ABC`

Provides a standardized interface for building model pipelines.

Implements methods for preprocessing data, training models, making predictions, evaluating performance, and visualizing results. Subclasses customize behavior by providing specific implementations for processing, training, evaluation, and visualization. For example when using the Stackix Model, we would use the StackixPreprocessor Type for preprocessing.

Attributes:

| Name                | Type                                                   | Description                                                                                                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`            |                                                        | Configuration for the pipeline's components and behavior.                                                                                                                                                                                                                    |
| `preprocessed_data` | `Optional[DatasetContainer]`                           | Pre-split and processed data that can be provided by user.                                                                                                                                                                                                                   |
| `raw_user_data`     | `Union[DataPackage, AnnData, MuData, DataFrame, dict]` | Raw input data for processing (DataFrames, MuData, etc.).                                                                                                                                                                                                                    |
| `result`            |                                                        | Storage container for all pipeline outputs.                                                                                                                                                                                                                                  |
| `_preprocessor`     |                                                        | Component that filters, scales, and cleans data.                                                                                                                                                                                                                             |
| `_visualizer`       |                                                        | Component that generates visual representations of results.                                                                                                                                                                                                                  |
| `_dataset_type`     |                                                        | Base class for dataset implementations.                                                                                                                                                                                                                                      |
| `_trainer_type`     |                                                        | Base class for trainer implementations.                                                                                                                                                                                                                                      |
| `_model_type`       |                                                        | Base class for model architecture implementations.                                                                                                                                                                                                                           |
| `_loss_type`        |                                                        | Base class for loss function implementations.                                                                                                                                                                                                                                |
| `_datasets`         | `Optional[DatasetContainer]`                           | Split datasets after preprocessing.                                                                                                                                                                                                                                          |
| `_evaluator`        | `Optional[DatasetContainer]`                           | Component that assesses model performance. Not implemented yet                                                                                                                                                                                                               |
| `_data_splitter`    |                                                        | Component that divides data into train/validation/test sets.                                                                                                                                                                                                                 |
| `_ontologies`       |                                                        | Tuple of dictionaries containing the ontologies to be used to construct sparse decoder layers. If a list is provided, it is assumed to be a list of file paths to ontology files. First item in list or tuple will be treated as first layer (after latent space) and so on. |

### `__init__(dataset_type: Type[BaseDataset], trainer_type: Type[BaseTrainer], model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], datasplitter_type: Type[DataSplitter], preprocessor_type: Type[BasePreprocessor], data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData, pd.DataFrame, dict]], visualizer: Optional[Type[BaseVisualizer]] = None, evaluator: Optional[Type[BaseEvaluator]] = None, result: Optional[Result] = None, config: Optional[DefaultConfig] = None, custom_split: Optional[Dict[str, np.ndarray]] = None, ontologies: Optional[Union[Tuple, Dict[Any, Any]]] = None, masking_fn: Optional[Callable] = None, masking_fn_kwargs: Dict[str, Any] = {}, **kwargs: dict) -> None`

Initializes the pipeline with components and configuration.

Concrete pipelines (e.g. `Vanillix`, `Varix`) pass their component types here; users normally only provide `data`, `config` and, where relevant, `ontologies`, `custom_split` or `masking_fn`.

Parameters:

| Name                | Type                                                                               | Description                                                                                                                                                               | Default    |
| ------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `dataset_type`      | `Type[BaseDataset]`                                                                | Class for dataset implementations.                                                                                                                                        | *required* |
| `trainer_type`      | `Type[BaseTrainer]`                                                                | Class for model training implementations.                                                                                                                                 | *required* |
| `model_type`        | `Type[BaseAutoencoder]`                                                            | Class for model architecture implementations.                                                                                                                             | *required* |
| `loss_type`         | `Type[BaseLoss]`                                                                   | Class for loss function implementations.                                                                                                                                  | *required* |
| `datasplitter_type` | `Type[DataSplitter]`                                                               | Class for data splitting implementation.                                                                                                                                  | *required* |
| `preprocessor_type` | `Type[BasePreprocessor]`                                                           | Class for data preprocessing implementation.                                                                                                                              | *required* |
| `data`              | `Optional[Union[DataPackage, DatasetContainer, AnnData, MuData, DataFrame, dict]]` | Input data: a DataPackage, an already preprocessed DatasetContainer, or raw AnnData, MuData, pandas.DataFrame or dict, which is wrapped into a DataPackage automatically. | *required* |
| `visualizer`        | `Optional[Type[BaseVisualizer]]`                                                   | Class for generating visualizations.                                                                                                                                      | `None`     |
| `evaluator`         | `Optional[Type[BaseEvaluator]]`                                                    | Class for assessing model performance.                                                                                                                                    | `None`     |
| `result`            | `Optional[Result]`                                                                 | Storage container for pipeline outputs. A new Result is created if None.                                                                                                  | `None`     |
| `config`            | `Optional[DefaultConfig]`                                                          | Configuration parameters for all pipeline components. The pipeline's default config is used if None.                                                                      | `None`     |
| `custom_split`      | `Optional[Dict[str, ndarray]]`                                                     | User-provided data splits, mapping "train", "valid" and "test" to sample indices.                                                                                         | `None`     |
| `ontologies`        | `Optional[Union[Tuple, Dict[Any, Any]]]`                                           | Ontology definitions for Ontix (sparse decoder layers); ignored by other pipelines.                                                                                       | `None`     |
| `masking_fn`        | `Optional[Callable]`                                                               | Custom masking function for Maskix; ignored by other pipelines.                                                                                                           | `None`     |
| `masking_fn_kwargs` | `Dict[str, Any]`                                                                   | Keyword arguments passed to masking_fn.                                                                                                                                   | `{}`       |
| `**kwargs`          | `dict`                                                                             | Additional keyword arguments (e.g. model_map).                                                                                                                            | `{}`       |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If the pipeline class does not define \_default_config. |
| `TypeError`  | If inputs have incorrect types.                         |

### `decode(latent: Union[torch.Tensor, ad.AnnData, pd.DataFrame]) -> Union[torch.Tensor, ad.AnnData, pd.DataFrame]`

Transforms latent space representations back to input space.

Handles various input formats for the latent representation and returns the decoded data in a matching format.

Parameters:

| Name     | Type                                | Description                            | Default    |
| -------- | ----------------------------------- | -------------------------------------- | ---------- |
| `latent` | `Union[Tensor, AnnData, DataFrame]` | Latent space representation to decode. | *required* |

Returns:

| Type                                | Description                                  |
| ----------------------------------- | -------------------------------------------- |
| `Union[Tensor, AnnData, DataFrame]` | Decoded data in a format matching the input. |

Raises:

| Type         | Description                                            |
| ------------ | ------------------------------------------------------ |
| `TypeError`  | If no model has been trained or input type is invalid. |
| `ValueError` | If latent dimensions are incompatible with the model.  |

### `evaluate(ml_model_class: ClassifierMixin = linear_model.LogisticRegression(), ml_model_regression: RegressorMixin = linear_model.LinearRegression(), params: Union[list, str] = [], metric_class: str = 'roc_auc_ovo', metric_regression: str = 'r2', reference_methods: list = [], reference_reducer: dict = {}, split_type: str = 'use-split', n_downsample: Optional[int] = 10000, top_k_classes: Optional[int] = 20, exclude_classes: Union[list, None] = None) -> Result`

Benchmark the learned embedding on downstream machine-learning tasks.

Trains scikit-learn models on the latent space (and optionally on reference embeddings such as PCA or UMAP) to predict annotation columns, and stores the performance per task, method and split in `result.embedding_evaluation`. Categorical columns are treated as classification, numeric columns as regression. A summary plot is created afterwards.

Parameters:

| Name                  | Type                | Description                                                                                                                                                                                                            | Default                |
| --------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `ml_model_class`      | `ClassifierMixin`   | Scikit-learn classifier used for categorical parameters.                                                                                                                                                               | `LogisticRegression()` |
| `ml_model_regression` | `RegressorMixin`    | Scikit-learn regressor used for numeric parameters.                                                                                                                                                                    | `LinearRegression()`   |
| `params`              | `Union[list, str]`  | Annotation columns to predict. If empty, the annotation_columns from the data config are used.                                                                                                                         | `[]`                   |
| `metric_class`        | `str`               | Scikit-learn scoring name for classification tasks.                                                                                                                                                                    | `'roc_auc_ovo'`        |
| `metric_regression`   | `str`               | Scikit-learn scoring name for regression tasks.                                                                                                                                                                        | `'r2'`                 |
| `reference_methods`   | `list`              | Additional embeddings to compare against the latent space: "PCA", "UMAP", "TSNE" and/or "RandomFeature" (five random feature subsets of the input). The latent space is always evaluated.                              | `[]`                   |
| `reference_reducer`   | `dict`              | Optional pre-fitted reducers, e.g. {"PCA": pca, "UMAP": umap}, to reuse the same transformation across runs.                                                                                                           | `{}`                   |
| `split_type`          | `str`               | Evaluation strategy: "use-split" (train on the train split, score on valid/test), "CV-N" for N-fold cross-validation (e.g. "CV-5"), or "LOOCV" for leave-one-out. Falls back to "CV-5" if no train split is available. | `'use-split'`          |
| `n_downsample`        | `Optional[int]`     | Maximum number of samples used per evaluation for speed; None disables downsampling.                                                                                                                                   | `10000`                |
| `top_k_classes`       | `Optional[int]`     | Keep only the k most frequent classes per parameter and merge the rest into "other"; None keeps all classes.                                                                                                           | `20`                   |
| `exclude_classes`     | `Union[list, None]` | Class labels to drop before evaluation.                                                                                                                                                                                | `None`                 |

Returns:

| Type     | Description                |
| -------- | -------------------------- |
| `Result` | The updated Result object. |

Raises:

| Type                  | Description                                                                                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NotImplementedError` | If no evaluator is set or the model is not trained.                                                                                                                        |
| `ValueError`          | If split_type is invalid, no parameters are given or configured, latent spaces are missing (run predict first), or "RandomFeature" is requested without preprocessed data. |

(LLM-generated docstring, pending human review.)

### `explain(method: Literal['DeepLiftShap', 'IntegratedGradients'] = 'DeepLiftShap', sel_latent_dim: Union[list, int, str, None] = None, input_type: Literal['random', 'grouped'] = 'grouped', input_group: Optional[str] = None, baseline_type: Literal['mean', 'random', 'grouped'] = 'mean', baseline_group: str = 'all', anno_col: Optional[str] = None, n_subset: int = 100, seed_int: int = 12, split: Literal['train', 'test', 'valid'] = 'train', llm_explain: bool = False, llm_client: Literal['ollama', 'mistral', 'openrouter', 'scads-llm'] = 'mistral', llm_model: str = 'mistral-large-latest', top_n_genes: int = 40, prompt: str = PROMPT, save: Union[str, Path, None] = None) -> pd.DataFrame`

Runs the feature-importance explainer and returns gene-by-latent-dimension attribution scores.

Parameters:

| Name             | Type                                                      | Description                                                                                                                                                                       | Default                  |
| ---------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `method`         | `Literal['DeepLiftShap', 'IntegratedGradients']`          | Specifies which attribution algorithm to use for explaining the model.                                                                                                            | `'DeepLiftShap'`         |
| `sel_latent_dim` | `Union[list, int, str, None]`                             | Specifies which latent dimension(s) to explain. If None, all dimensions will be explained.                                                                                        | `None`                   |
| `input_type`     | `Literal['random', 'grouped']`                            | Specifies whether the feature-importance algorithm should use 'random' samples or 'grouped' samples (see input_group) as input for the attribution computation.                   | `'grouped'`              |
| `input_group`    | `Optional[str]`                                           | If input_type is 'grouped', this specifies which subset in anno_col to filter for to use as input for the attribution computation.                                                | `None`                   |
| `baseline_type`  | `Literal['mean', 'random', 'grouped']`                    | Specifies whether the feature-importance algorithm should use the 'mean' baseline, a 'random'-sample baseline, or samples from a 'grouped' baseline (see baseline_group).         | `'mean'`                 |
| `baseline_group` | `str`                                                     | Specifies whether the baseline is computed using all data (default) or which subset in anno_col to filter for.                                                                    | `'all'`                  |
| `anno_col`       | `Optional[str]`                                           | If baseline_group is not 'all', this specifies the annotation column used to filter the baseline data.                                                                            | `None`                   |
| `n_subset`       | `int`                                                     | Specifies the number of cells to use when subsampling for the attribution computation.                                                                                            | `100`                    |
| `seed_int`       | `int`                                                     | Defines the random seed used for reproducible subsampling and attribution calculations.                                                                                           | `12`                     |
| `split`          | `Literal['train', 'test', 'valid']`                       | The split to use for feature importance calculation (train, valid, test), default is train.                                                                                       | `'train'`                |
| `llm_explain`    | `bool`                                                    | Whether to use LLM explainers for feature importance calculation.                                                                                                                 | `False`                  |
| `llm_client`     | `Literal['ollama', 'mistral', 'openrouter', 'scads-llm']` | The LLM client to use for feature importance calculation.                                                                                                                         | `'mistral'`              |
| `llm_model`      | `str`                                                     | The LLM model to use for feature importance calculation.                                                                                                                          | `'mistral-large-latest'` |
| `top_n_genes`    | `int`                                                     | How many top (contribution to embedding) genes to consider for LLM explanation.                                                                                                   | `40`                     |
| `prompt`         | `str`                                                     | Prompt to pass to the LLM when explaining a latent dimension.                                                                                                                     | `PROMPT`                 |
| `save`           | `Union[str, Path, None]`                                  | Path to directory with or without filename to save the generated .md file to. If set to None the file will be saved to the current working directory as "latent explanations.md". | `None`                   |

Returns:

| Type        | Description                                                                                                                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DataFrame` | pd.DataFrame: The generated samples in the input space. A DataFrame of attribution scores with genes as the index and latent dimensions as columns. Each entry represents the contribution of a given gene to a specific latent dimension. |

### `fit(config: Optional[Union[None, DefaultConfig]] = None, **kwargs)`

Trains the model on preprocessed data.

Creates and configures a trainer instance, then executes the training process using the preprocessed datasets.

Parameters:

| Name       | Type                                   | Description                                               | Default |
| ---------- | -------------------------------------- | --------------------------------------------------------- | ------- |
| `config`   | `Optional[Union[None, DefaultConfig]]` | Optional custom configuration for training.               | `None`  |
| `**kwargs` |                                        | Additional configuration parameters as keyword arguments. | `{}`    |

Raises:

| Type         | Description                                |
| ------------ | ------------------------------------------ |
| `ValueError` | If datasets aren't available for training. |

### `generate(n_samples: Optional[int] = None, latent_prior: Optional[Union[np.ndarray, torch.Tensor]] = None, split: str = 'test', epoch: int = -1) -> torch.Tensor`

Generates new samples from the model's latent space.

This method allows for the generation of new data samples by sampling from the model's latent space. Users can either provide a custom latent prior or specify the number of samples to generate. If a custom latent prior is provided, its batch dimension must be compatible with n_samples.

Parameters:

| Name           | Type                               | Description                                                                                                                                                                                                                  | Default  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `n_samples`    | `Optional[int]`                    | The number of samples to generate.                                                                                                                                                                                           | `None`   |
| `latent_prior` | `Optional[Union[ndarray, Tensor]]` | Optional custom latent prior distribution. If provided, this will be used for sampling instead of the learned distribution. The prior must either be a single latent vector or a batch of latent vectors matching n_samples. | `None`   |
| `split`        | `str`                              | The split to sample from (train, valid, test), default is test.                                                                                                                                                              | `'test'` |
| `epoch`        | `int`                              | The epoch to sample from, default is the last epoch (-1).                                                                                                                                                                    | `-1`     |

Returns:

| Type     | Description                                             |
| -------- | ------------------------------------------------------- |
| `Tensor` | torch.Tensor: The generated samples in the input space. |

Raises:

| Type         | Description                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `ValueError` | If n_samples is not a positive integer or if the latent prior has incompatible dimensions. |
| `TypeError`  | If latent_prior is not a numpy array or tensor.                                            |

### `impute(corrupted_tensor: torch.Tensor)`

Impute missing values in a corrupted input tensor.

Only implemented by the `Maskix` pipeline; see `Maskix.impute`.

Parameters:

| Name               | Type     | Description                                   | Default    |
| ------------------ | -------- | --------------------------------------------- | ---------- |
| `corrupted_tensor` | `Tensor` | Input data with missing or corrupted entries. | *required* |

Raises:

| Type                  | Description                              |
| --------------------- | ---------------------------------------- |
| `NotImplementedError` | Always, for pipelines other than Maskix. |

(LLM-generated docstring, pending human review.)

### `load(file_path) -> Any`

Loads a pipeline from a file.

Parameters:

| Name        | Type | Description                 | Default    |
| ----------- | ---- | --------------------------- | ---------- |
| `file_path` |      | Path to the saved pipeline. | *required* |

Returns:

| Type  | Description                   |
| ----- | ----------------------------- |
| `Any` | The loaded pipeline instance. |

### `predict(data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]] = None, config: Optional[Union[None, DefaultConfig]] = None, from_key: Optional[str] = None, to_key: Optional[str] = None, **kwargs)`

Generates predictions using the trained model.

Uses the trained model to make predictions on test data or new data provided by the user. Processes the results and stores them in the result container.

Parameters:

| Name       | Type                                                              | Description                                                                                  | Default |
| ---------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------- |
| `data`     | `Optional[Union[DataPackage, DatasetContainer, AnnData, MuData]]` | Optional new data for predictions. If None, the test split of the preprocessed data is used. | `None`  |
| `config`   | `Optional[Union[None, DefaultConfig]]`                            | Optional custom configuration for prediction.                                                | `None`  |
| `from_key` | `Optional[str]`                                                   | Source modality for cross-modal translation (XModalix only).                                 | `None`  |
| `to_key`   | `Optional[str]`                                                   | Target modality for cross-modal translation (XModalix only).                                 | `None`  |
| `**kwargs` |                                                                   | Additional configuration parameters as keyword arguments.                                    | `{}`    |

Returns:

| Type | Description                                                       |
| ---- | ----------------------------------------------------------------- |
|      | The updated Result object with latent spaces and reconstructions. |

Raises:

| Type                  | Description                                                                           |
| --------------------- | ------------------------------------------------------------------------------------- |
| `NotImplementedError` | If required components aren't initialized.                                            |
| `ValueError`          | If no test data is available, the data format is invalid, or fit has not been called. |

### `preprocess(config: Optional[Union[None, DefaultConfig]] = None, **kwargs)`

Filters, normalizes and prepares data for model training.

Processes raw input data into the format required by the model and creates train/validation/test splits as needed.

Parameters:

| Name       | Type                                   | Description                                               | Default |
| ---------- | -------------------------------------- | --------------------------------------------------------- | ------- |
| `config`   | `Optional[Union[None, DefaultConfig]]` | Optional custom configuration for preprocessing.          | `None`  |
| `**kwargs` |                                        | Additional configuration parameters as keyword arguments. | `{}`    |

Raises:

| Type                  | Description                         |
| --------------------- | ----------------------------------- |
| `NotImplementedError` | If preprocessor is not initialized. |

### `run(data: Optional[Union[DatasetContainer, DataPackage]] = None) -> Result`

Executes the complete pipeline from preprocessing to visualization.

Runs all pipeline steps in sequence and returns the result.

Parameters:

| Name   | Type                                             | Description                                         | Default |
| ------ | ------------------------------------------------ | --------------------------------------------------- | ------- |
| `data` | `Optional[Union[DatasetContainer, DataPackage]]` | Optional data for prediction (overrides test data). | `None`  |

Returns:

| Type     | Description                |
| -------- | -------------------------- |
| `Result` | Complete pipeline results. |

### `sample_latent_space(n_samples: int, split: str = 'test', epoch: int = -1) -> torch.Tensor`

Samples latent space points from the learned distribution.

If `n_samples` is not provided, this method returns one latent point per sample in the specified split (legacy behavior). If `n_samples` is given, it draws samples from the aggregated posterior distribution of the split.

Parameters:

| Name        | Type  | Description                                                                                                      | Default    |
| ----------- | ----- | ---------------------------------------------------------------------------------------------------------------- | ---------- |
| `split`     | `str` | The split to sample from (train, valid, test), default is test.                                                  | `'test'`   |
| `epoch`     | `int` | The epoch to sample from, default is the last epoch (-1).                                                        | `-1`       |
| `n_samples` | `int` | Optional number of latent points to sample. If None, returns one latent point per available sample in the split. | *required* |

Returns:

| Name | Type     | Description                                     |
| ---- | -------- | ----------------------------------------------- |
| `z`  | `Tensor` | torch.Tensor - The sampled latent space points. |

Raises:

| Type         | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `ValueError` | If the model has not been trained or latent statistics have not been computed. |
| `TypeError`  | If mu or logvar are not numpy arrays.                                          |

### `save(file_path: str, save_all: bool = False)`

Saves the pipeline to a file.

Writes three files next to each other: `<file_path>` (pickled pipeline), `<stem>_preprocessor.pkl` and `<stem>_model.pth` (model weights). Restore with `<PipelineClass>.load(file_path)`.

Warning

With `save_all=False` the pipeline object in memory is slimmed down as a side effect: datasets, raw data and most results are removed and the in-memory model is re-initialized (weights are only kept in the saved files). Reload the pipeline with `load` before continuing to work with it, or use `save_all=True`.

Parameters:

| Name        | Type   | Description                                                                                                                                                                                                | Default    |
| ----------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `file_path` | `str`  | Path where the pipeline should be saved.                                                                                                                                                                   | *required* |
| `save_all`  | `bool` | If True, save the complete pipeline including data and all results. If False (default), save only what is needed for inference (model, config, preprocessor, losses and adata_latent) to reduce file size. | `False`    |

### `show_result(split: str = 'all', **kwargs)`

Displays key visualizations of model results.

This method generates the following visualizations:

1. Loss Curves: Displays the absolute loss curves to provide insights into the model's training and validation performance over epochs.
1. Latent Space Ridgeline Plot: Visualizes the distribution of the latent space representations across different dimensions, offering a high-level overview of the learned embeddings.
1. Latent Space 2D Scatter Plot: Projects the latent space into two dimensions for a detailed view of the clustering or separation of data points.

These visualizations help in understanding the model's performance and the structure of the latent space representations.

Parameters:

| Name       | Type  | Description                                                                                                                      | Default |
| ---------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `split`    | `str` | Data split to plot ("train", "valid", "test" or "all").                                                                          | `'all'` |
| `**kwargs` |       | Optional params: annotation column(s) used to color the latent space plots. Defaults to annotation_columns from the data config. | `{}`    |

### `visualize(config: Optional[Union[None, DefaultConfig]] = None, **kwargs)`

Creates visualizations of model results and performance.

Parameters:

| Name       | Type                                   | Description                                      | Default |
| ---------- | -------------------------------------- | ------------------------------------------------ | ------- |
| `config`   | `Optional[Union[None, DefaultConfig]]` | Optional custom configuration for visualization. | `None`  |
| `**kwargs` |                                        | Additional configuration parameters.             | `{}`    |

Raises:

| Type                  | Description                       |
| --------------------- | --------------------------------- |
| `NotImplementedError` | If visualizer is not initialized. |

## `BasePreprocessor`

Bases: `ABC`

Contains logic for data preprocessing in the Autoencodix framework.

This class defines the general preprocessing workflow and provides methods for handling different data modalities and data cases. Subclasses should implement the `preprocess` method to perform specific preprocessing steps.

Attributes:

| Name                       | Type                                  | Description                                                                                              |
| -------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `config`                   |                                       | A DefaultConfig object containing preprocessing configurations.                                          |
| `processed_data`           |                                       | A dictionary to store processed DataPackage objects for each data split.                                 |
| `bulk_genes_to_keep`       | `Optional[Dict[str, List[str]]]`      | Optional list of genes to keep for bulk data.                                                            |
| `bulk_scalers`             | `Optional[Dict[str, Any]]`            | Optional dictionary of scalers for bulk data.                                                            |
| `sc_genes_to_keep`         | `Optional[Dict[str, List[str]]]`      | Optional dictionary mapping modality keys to lists of genes to keep for single-cell data.                |
| `sc_scalers`               | `Optional[Dict[str, Dict[str, Any]]]` | Optional dictionary mapping modality keys to scalers for single-cell data.                               |
| `sc_general_genes_to_keep` | `Optional[Dict[str, List]]`           | Optional dictionary mapping modality keys to lists of genes to keep filtered by non-SC specific methods. |
| `data_readers`             | `Dict[Enum, Any]`                     | A dictionary mapping DataCase enum values to data reader instances for different modalities.             |
| `_dataset_container`       | `Optional[DatasetContainer]`          | Optional DatasetContainer to hold the processed datasets.                                                |

### `__init__(config: DefaultConfig, ontologies: Optional[Union[Tuple[Any, Any], Dict[Any, Any]]] = None)`

Initializes the BasePreprocessor with a configuration object.

Args

config: A DefaultConfig object containing preprocessing configurations. ontologies: Ontology information, if provided for Ontix.

### `filter_imgdata_list(img_list, ids)`

Keep only the images whose sample ID is in `ids`.

Parameters:

| Name       | Type            | Description         | Default    |
| ---------- | --------------- | ------------------- | ---------- |
| `img_list` | `list[ImgData]` | Images to filter.   | *required* |
| `ids`      | `Collection`    | Sample IDs to keep. | *required* |

Returns:

| Type | Description                                                    |
| ---- | -------------------------------------------------------------- |
|      | list\[ImgData\]: The filtered images, in their original order. |

(LLM-generated docstring, pending human review.)

### `format_reconstruction(reconstruction: Dict[str, torch.Tensor], result: Optional[Result] = None) -> DataPackage`

Convert reconstructed tensors back into the user's input data format.

Parameters:

| Name             | Type                | Description                                                | Default    |
| ---------------- | ------------------- | ---------------------------------------------------------- | ---------- |
| `reconstruction` | `Dict[str, Tensor]` | Reconstructed tensors, keyed by modality.                  | *required* |
| `result`         | `Optional[Result]`  | Result object providing sample and feature IDs, if needed. | `None`     |

Returns:

| Type          | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `DataPackage` | A DataPackage with reconstructions in the original data structure |
| `DataPackage` | (e.g. DataFrames or AnnData).                                     |

(LLM-generated docstring, pending human review.)

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data: bool = False) -> DatasetContainer`

To be implemented by subclasses for specific preprocessing steps. Args: raw_user_data: Users can provide raw data. This is an alternative way of providing data via filepaths in the config. If this param is passed, we skip the data reading step. predict_new_data: Indicates whether the user wants to predict with unseen data. If this is the case, we don't split the data and only prerpocess.

## `BaseTrainer`

Bases: `ABC`

General training logic for all autoencoder models.

This class sets up the model, optimizer, and data loaders. It also handles reproducibility and model-specific configurations. Subclasses must implement model training and prediction logic.

Attributes:

| Name           | Type | Description                                                            |
| -------------- | ---- | ---------------------------------------------------------------------- |
| `_trainset`    |      | The dataset used for training.                                         |
| `_validset`    |      | The dataset used for validation, if provided.                          |
| `_result`      |      | An object to store and manage training results.                        |
| `_config`      |      | Configuration object containing training hyperparameters and settings. |
| `_model_type`  |      | The autoencoder model class to be trained.                             |
| `_loss_fn`     |      | Instantiated loss function specific to the model.                      |
| `_trainloader` |      | DataLoader for the training dataset.                                   |
| `_validloader` |      | DataLoader for the validation dataset, if provided.                    |
| `_model`       |      | The instantiated model architecture.                                   |
| `_optimizer`   |      | The optimizer used for training.                                       |
| `_fabric`      |      | Lightning Fabric wrapper for device and precision management.          |
| `ontologies`   |      | Ontology information, if provided for Ontix                            |

### `__init__(trainset: Optional[BaseDataset], validset: Optional[BaseDataset], result: Result, config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], ontologies: Optional[Union[Tuple, List]] = None, **kwargs)`

Store the training components and set up model, optimizer and loaders.

Parameters:

| Name         | Type                           | Description                                                                 | Default    |
| ------------ | ------------------------------ | --------------------------------------------------------------------------- | ---------- |
| `trainset`   | `Optional[BaseDataset]`        | Dataset used for training; must not be None.                                | *required* |
| `validset`   | `Optional[BaseDataset]`        | Dataset used for validation, or None to train without validation.           | *required* |
| `result`     | `Result`                       | Result container that collects losses, latent spaces and the trained model. | *required* |
| `config`     | `DefaultConfig`                | Configuration with training hyperparameters and device settings.            | *required* |
| `model_type` | `Type[BaseAutoencoder]`        | Autoencoder architecture class to instantiate.                              | *required* |
| `loss_type`  | `Type[BaseLoss]`               | Loss class to instantiate with config.                                      | *required* |
| `ontologies` | `Optional[Union[Tuple, List]]` | Ontology definitions; required for Ontix, ignored by other architectures.   | `None`     |
| `**kwargs`   |                                | Additional keyword arguments for subclasses.                                | `{}`       |

(LLM-generated docstring, pending human review.)

### `decode(x: torch.Tensor) -> torch.Tensor`

Decode latent vectors into the input space with the trained model.

Parameters:

| Name | Type     | Description                                      | Default    |
| ---- | -------- | ------------------------------------------------ | ---------- |
| `x`  | `Tensor` | Latent vectors of shape (n_samples, latent_dim). | *required* |

Returns:

| Type     | Description                            |
| -------- | -------------------------------------- |
| `Tensor` | Reconstructed data in the input space. |

(LLM-generated docstring, pending human review.)

### `predict(data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs) -> Result`

Encode and reconstruct a dataset with a trained model.

Parameters:

| Name       | Type               | Description                                                | Default    |
| ---------- | ------------------ | ---------------------------------------------------------- | ---------- |
| `data`     | `BaseDataset`      | Dataset to predict on.                                     | *required* |
| `model`    | `Optional[Module]` | Trained model to use; defaults to the trainer's own model. | `None`     |
| `**kwargs` |                    | Additional keyword arguments for subclasses.               | `{}`       |

Returns:

| Type     | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| `Result` | The Result object updated with latent spaces and reconstructions. |

(LLM-generated docstring, pending human review.)

### `purge() -> None`

Cleans up any resources used during training, such as cached data or large attributes.

### `setup_trainer(old_model=None)`

Create loss, Lightning Fabric, model, optimizer and data loaders.

Called from `__init__`. Also called before prediction with an already trained model, in which case input validation and data loader creation are skipped and the given model is reused.

Parameters:

| Name        | Type | Description                                                                                         | Default |
| ----------- | ---- | --------------------------------------------------------------------------------------------------- | ------- |
| `old_model` |      | Previously trained model to reuse (e.g. after load). If None, a new model is built from model_type. | `None`  |

(LLM-generated docstring, pending human review.)

### `train(epochs_overwrite: Optional[int] = None) -> Result`

Train the model and return the populated result container.

Parameters:

| Name               | Type            | Description                                         | Default |
| ------------------ | --------------- | --------------------------------------------------- | ------- |
| `epochs_overwrite` | `Optional[int]` | Number of epochs to train instead of config.epochs. | `None`  |

Returns:

| Type     | Description                                                         |
| -------- | ------------------------------------------------------------------- |
| `Result` | The Result object with losses, latent spaces and the trained model. |

(LLM-generated docstring, pending human review.)

## `BaseVisualizer`

Bases: `ABC`

Defines the interface for visualizing training results.

Attributes:

| Name    | Type | Description                                 |
| ------- | ---- | ------------------------------------------- |
| `plots` |      | A nested dictionary to store various plots. |

### `save_plots(path: str, which: Union[str, list] = 'all', format: str = 'png') -> None`

Save specified plots to the given path in the specified format.

Parameters:

| Name     | Type               | Description                                                                                                                                                                                                      | Default    |
| -------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path`   | `str`              | The directory path where the plots will be saved.                                                                                                                                                                | *required* |
| `which`  | `Union[str, list]` | A list of plot names to save or a string specifying which plots to save. If 'all', all plots in the plots dictionary will be saved. If a single plot name is provided as a string, only that plot will be saved. | `'all'`    |
| `format` | `str`              | The file format in which to save the plots (e.g., 'png', 'jpg').                                                                                                                                                 | `'png'`    |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

Raises:

| Type         | Description                                         |
| ------------ | --------------------------------------------------- |
| `ValueError` | If the 'which' parameter is not a list or a string. |

### `show_evaluation(param: str, metric: str, ml_alg: Optional[str] = None) -> None`

Displays the evaluation plot for a specific clinical parameter, metric, and optionally ML algorithm. Args: param: clinical parameter to visualize. metric: metric to visualize. ml_alg: ML algorithm to visualize. If None, plots all available algorithms. Returns: None

### `show_latent_space(result: Result, plot_type: str = '2D-scatter', labels: Optional[Union[list, pd.Series, None]] = None, param: Optional[Union[list, str]] = None, epoch: Optional[Union[int, None]] = None, split: str = 'all') -> None`

Display the latent space, colored by annotation parameters.

Parameters:

| Name        | Type                                  | Description                                             | Default        |
| ----------- | ------------------------------------- | ------------------------------------------------------- | -------------- |
| `result`    | `Result`                              | Result object containing the latent spaces.             | *required*     |
| `plot_type` | `str`                                 | Type of plot, e.g. "2D-scatter" or "Ridgeline".         | `'2D-scatter'` |
| `labels`    | `Optional[Union[list, Series, None]]` | Optional explicit labels per sample used for coloring.  | `None`         |
| `param`     | `Optional[Union[list, str]]`          | Annotation column(s) used for coloring, or "all".       | `None`         |
| `epoch`     | `Optional[Union[int, None]]`          | Epoch to plot; defaults to the last stored epoch.       | `None`         |
| `split`     | `str`                                 | Data split to plot ("train", "valid", "test" or "all"). | `'all'`        |

(LLM-generated docstring, pending human review.)

### `show_loss(plot_type: str = 'absolute') -> None`

Display the loss plot. Args: plot_type: Type of loss plot to display. Options are "absolute" or "relative". Options are "absolute" for the absolute loss plot and "relative" for the relative loss plot. Defaults to "absolute". Returns: None

### `show_weights() -> None`

Display heatmaps of the encoder and decoder weights of the trained model.

(LLM-generated docstring, pending human review.)

### `visualize(result: Result, config: DefaultConfig) -> Result`

Create all standard plots for a pipeline run and store them in the result.

Parameters:

| Name     | Type            | Description                                         | Default    |
| -------- | --------------- | --------------------------------------------------- | ---------- |
| `result` | `Result`        | Result object with losses, latent spaces and model. | *required* |
| `config` | `DefaultConfig` | Configuration of the pipeline run.                  | *required* |

Returns:

| Type     | Description                                         |
| -------- | --------------------------------------------------- |
| `Result` | The result object (plots are stored in self.plots). |

(LLM-generated docstring, pending human review.)
