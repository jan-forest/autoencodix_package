# Evaluate Module

## `GeneralEvaluator`

Bases: `BaseEvaluator`

Evaluate embeddings by training scikit-learn models on annotation tasks.

Used by `BasePipeline.evaluate` for tabular and single-cell pipelines. For each annotation parameter, a classifier (categorical) or regressor (numeric) is trained on the latent space and optional reference embeddings (PCA, UMAP, t-SNE, random features). Scores are appended to `result.embedding_evaluation`.

(LLM-generated docstring, pending human review.)

### `evaluate(datasets: DatasetContainer, result: Result, ml_model_class: ClassifierMixin = linear_model.LogisticRegression(max_iter=1000), ml_model_regression: RegressorMixin = linear_model.LinearRegression(), params: Union[list, str] = 'all', metric_class: str = 'roc_auc_ovo', metric_regression: str = 'r2', reference_methods: list = [], reference_reducer: dict = {}, split_type: str = 'use-split', n_downsample: Union[int, None] = 10000, top_k_classes: Union[int, None] = 20, exclude_classes: Union[list, None] = None) -> Result`

Evaluates the performance of machine learning models on various feature representations and clinical parameters.

This method performs classification or regression tasks using specified machine learning models on different feature sets (e.g., latent space, PCA, UMAP, TSNE, RandomFeature) and clinical annotation parameters. It supports multiple evaluation strategies, including pre-defined train/valid/test splits, k-fold cross-validation, and leave-one-out cross-validation. The results are aggregated and stored in the provided `result` object.

- Samples with missing annotation values for a given parameter are excluded from the corresponding evaluation.
- For "RandomFeature", five random feature sets are evaluated.
- The method appends results to any existing `embedding_evaluation` in the result object.

Parameters:

| Name                  | Type                | Description                                                                                                                                                                         | Default                             |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `datasets`            | `DatasetContainer`  | A DatasetContainer containing train, valid, and test datasets, each with sample_ids and metadata (either a DataFrame or a dictionary with a 'paired' key for clinical annotations). | *required*                          |
| `result`              | `Result`            | An Result object to store the evaluation results. Should have an embedding_evaluation attribute which updated (typically a DataFrame).                                              | *required*                          |
| `ml_model_class`      | `ClassifierMixin`   | The scikit-learn classifier to use for classification tasks (default: sklearn.linear_model.LogisticRegression()).                                                                   | `LogisticRegression(max_iter=1000)` |
| `ml_model_regression` | `RegressorMixin`    | The scikit-learn regressor to use for regression tasks (default: sklearn.linear_model.LinearRegression()).                                                                          | `LinearRegression()`                |
| `params`              | `Union[list, str]`  | List of clinical annotation columns to evaluate, or "all" to use all columns (default: "all").                                                                                      | `'all'`                             |
| `metric_class`        | `str`               | Scoring metric for classification tasks (default: "roc_auc_ovo").                                                                                                                   | `'roc_auc_ovo'`                     |
| `metric_regression`   | `str`               | Scoring metric for regression tasks (default: "r2").                                                                                                                                | `'r2'`                              |
| `reference_methods`   | `list`              | List of feature representations to evaluate (e.g., "PCA", "UMAP", "TSNE", "RandomFeature"). "Latent" is always included (default: []).                                              | `[]`                                |
| `reference_reducer`   | `dict`              | Optional dictionary of pre-fitted dimensionality reduction objects for PCA, UMAP, or TSNE to ensure consistent transformations across runs (default: {}).                           | `{}`                                |
| `split_type`          | `str`               | which split to use use-split" for pre-defined splits, "CV-N" for N-fold cross-validation, or "LOOCV" for leave-one-out cross-validation (default: "use-split").                     | `'use-split'`                       |
| `n_downsample`        | `Union[int, None]`  | If provided, downsample the data to this number of samples for faster evaluation. Default is 10000. Set to None to disable downsampling.                                            | `10000`                             |
| `top_k_classes`       | `Union[int, None]`  | If provided, restrict classification tasks to the top k classes, combining others into "other" (default: 20).                                                                       | `20`                                |
| `exclude_classes`     | `Union[list, None]` | List of classes to exclude from evaluation (default: None).                                                                                                                         | `None`                              |

Returns: The updated result object with evaluation results stored in `embedding_evaluation`. Raises ValueError: If required annotation data is missing or improperly formatted, or if an unsupported split type is specified.

## `Imagix3DEvaluator`

Bases: `GeneralEvaluator`

Evaluator for `Imagix3D` with latent-space diagnostics for 3D VAEs.

In addition to the downstream ML evaluation of `GeneralEvaluator`, it offers `compute_latent_activity` (active units / posterior collapse) and `compute_latent_traversal` (decode volumes while varying one latent dimension). Results are stored in `result.sub_results`.

(LLM-generated docstring, pending human review.)

### `compute_latent_activity(result: Result, threshold: float = 0.01, splits: tuple[str, ...] = ('train', 'valid'), include_test: bool = True) -> Result`

Compute the activity of each latent dimension (active units).

The latent activity diagnostic follows the common "active units" idea: a latent dimension is considered active if the variance of its posterior mean across samples exceeds a threshold.

Conceptually

activity_j = Var_x(mu_j(x))

where mu_j(x) is the encoder's posterior mean for latent dimension j.

The active-units diagnostic is used as a proxy for latent-space utilization and posterior collapse.

Parameters:

| Name           | Type              | Description                                                                                              | Default              |
| -------------- | ----------------- | -------------------------------------------------------------------------------------------------------- | -------------------- |
| `result`       | `Result`          | Result object with stored posterior means (result.mus) and, optionally, log-variances (result.sigmas).   | *required*           |
| `threshold`    | `float`           | Variance above which a dimension counts as active (0.01 as in Burda et al. 2015 and Sinha & Dieng 2021). | `0.01`               |
| `splits`       | `tuple[str, ...]` | Training splits evaluated for every stored epoch.                                                        | `('train', 'valid')` |
| `include_test` | `bool`            | If True, also evaluate the final test prediction.                                                        | `True`               |

Returns:

| Type     | Description                                                      |
| -------- | ---------------------------------------------------------------- |
| `Result` | The result object with the DataFrames                            |
| `Result` | sub_results["latent_activity_summary"] (per epoch and split) and |
| `Result` | sub_results["latent_activity_by_dim"] (per latent dimension).    |

Raises:

| Type         | Description                       |
| ------------ | --------------------------------- |
| `ValueError` | If no posterior means are stored. |

### `compute_latent_traversal(result: Result, latent_dims: Literal['all', 'top_kl'] | Sequence[int] = 'top_kl', n_latent_dims: int = 8, value_mode: Literal['prior', 'percentile'] = 'percentile', n_values: int = 5, prior_range: tuple[float, float] = (-2.0, 2.0), percentile_range: tuple[float, float] = (5.0, 95.0)) -> Result`

Compute latent traversal.

Latent traversal varies one latent dimension at a time while keeping all other latent dimensions fixed at the posterior mean of one selected sample. Uses the test split of the last epoch; latent vectors are decoded in batches of `config.batch_size`.

Parameters:

| Name               | Type                             | Description                                                                                                                                                            | Default                                                                                                                                                                                                                   |
| ------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `result`           | `Result`                         | Result object containing the trained model and stored latent means.                                                                                                    | *required*                                                                                                                                                                                                                |
| `latent_dims`      | \`Literal['all', 'top_kl']       | Sequence[int]\`                                                                                                                                                        | Which latent dimensions to traverse. Options: "all": traverse all latent dimensions. "top_kl": traverse the n_latent_dims dimensions with highest mean KL. Sequence\[int\]: explicitly provided latent dimension indices. |
| `n_latent_dims`    | `int`                            | Number of dimensions to use when latent_dims="top_kl". Must be between 1 and the total number of latent dimensions.                                                    | `8`                                                                                                                                                                                                                       |
| `value_mode`       | `Literal['prior', 'percentile']` | How traversal values are chosen. Options: "prior": use evenly spaced absolute values over prior_range. "percentile": use empirical percentiles of mu_j across samples. | `'percentile'`                                                                                                                                                                                                            |
| `n_values`         | `int`                            | Number of traversal values per latent dimension.                                                                                                                       | `5`                                                                                                                                                                                                                       |
| `prior_range`      | `tuple[float, float]`            | Minimum and maximum absolute values for prior-based traversal.                                                                                                         | `(-2.0, 2.0)`                                                                                                                                                                                                             |
| `percentile_range` | `tuple[float, float]`            | Lower and upper percentile for data-adaptive traversal values.                                                                                                         | `(5.0, 95.0)`                                                                                                                                                                                                             |

Returns:

| Type     | Description                                                             |
| -------- | ----------------------------------------------------------------------- |
| `Result` | The modified Result object; the decoded volumes and settings are stored |
| `Result` | in result.sub_results["latent_traversal"].                              |

Raises:

| Type         | Description                                            |
| ------------ | ------------------------------------------------------ |
| `ValueError` | If n_values < 2 or no test posterior means are stored. |

## `XModalixEvaluator`

Bases: `GeneralEvaluator`

Evaluator for cross-modal (`XModalix`) results.

Extends `GeneralEvaluator` so that each modality's latent space is evaluated separately, and adds `pure_vae_comparison` to compare translation quality against a single-modality VAE.

(LLM-generated docstring, pending human review.)

### `pure_vae_comparison(xmodalix_result: Result, pure_vae_result: Result, to_key: str, param: Optional[str] = None) -> Tuple[Figure, pd.DataFrame]`

Compares the reconstruction performance of a pure VAE model and a cross-modal VAE (xmodalix) model using Mean Squared Error (MSE) on test samples.

For each sample in the test set, computes the MSE between the original and reconstructed images for:

- Pure VAE reconstructions ("imagix")
- xmodalix reference reconstructions ("xmodalix_reference")
- xmodalix translated reconstructions ("xmodalix_translated") The results are merged with sample metadata and returned in a long-format DataFrame suitable for plotting. Optionally, boxplots are generated grouped by a specified metadata parameter.

Parameters:

| Name              | Type            | Description                                                                          | Default    |
| ----------------- | --------------- | ------------------------------------------------------------------------------------ | ---------- |
| `xmodalix_result` | `Result`        | The result object containing xmodalix model outputs and test datasets.               | *required* |
| `pure_vae_result` | `Result`        | The result object containing pure VAE model outputs and test datasets.               | *required* |
| `to_key`          | `str`           | The key specifying the target modality in the xmodalix dataset.                      | *required* |
| `param`           | `Optional[str]` | Metadata column name to group boxplots by. If None, plots are grouped by model only. | `None`     |

Returns:

| Type        | Description                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| `Figure`    | The matplotlib/seaborn boxplot figure comparing MSE distributions.                                        |
| `DataFrame` | DataFrame: Long-format DataFrame containing MSE values and associated metadata for each sample and model. |
