# Visualize Module

## `GeneralVisualizer`

Bases: `BaseVisualizer`

Visualizer for tabular and single-cell pipelines.

Creates loss curves, model-weight heatmaps, latent-space plots (2D scatter, ridgeline, clustermap, coverage/correlation) and evaluation plots. Used by `Vanillix`, `Varix`, `Ontix`, `Disentanglix`, `Stackix`, `Maskix` and as the base of other visualizers.

Attributes:

| Name    | Type             | Description                                         |
| ------- | ---------------- | --------------------------------------------------- |
| `plots` | `Dict[str, Any]` | Nested dict of created figures, keyed by plot name. |

(LLM-generated docstring, pending human review.)

### `show_latent_space(result: Result, plot_type: Literal['2D-scatter', 'Ridgeline', 'Coverage-Correlation'] = '2D-scatter', labels: Optional[Union[list, pd.Series, None]] = None, focus_labels: Optional[Union[list, None]] = None, param: Optional[Union[list, str]] = None, epoch: Optional[Union[int, None]] = None, split: str = 'all', n_downsample: Optional[int] = 10000, **kwargs) -> None`

Visualizes the latent space of the given result using different types of plots.

Parameters:

| Name           | Type                                                         | Description                                                                                                                                                                                                                                      | Default        |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| `result`       | `Result`                                                     | The result object containing latent spaces and losses.                                                                                                                                                                                           | *required*     |
| `plot_type`    | `Literal['2D-scatter', 'Ridgeline', 'Coverage-Correlation']` | The type of plot to generate. Options are "2D-scatter", "Ridgeline", and "Coverage-Correlation". Default is "2D-scatter".                                                                                                                        | `'2D-scatter'` |
| `labels`       | `Optional[Union[list, Series, None]]`                        | List of labels for the data points in the latent space. Default is None.                                                                                                                                                                         | `None`         |
| `focus_labels` | `Optional[Union[list, None]]`                                | List of labels which should be considered for coloring. All other labels are set to 'other'. Defaults to None where all labels are considered.                                                                                                   | `None`         |
| `param`        | `Optional[Union[list, str]]`                                 | List of parameters provided and stored as metadata. Strings must match column names. If not a list, string "all" is expected for convenient way to make plots for all parameters available. Default is None where no colored labels are plotted. | `None`         |
| `epoch`        | `Optional[Union[int, None]]`                                 | The epoch number to visualize. If None, the last epoch is inferred from the losses. Default is None.                                                                                                                                             | `None`         |
| `split`        | `str`                                                        | The data split to visualize. Options are "train", "valid", "test", and "all". Default is "all".                                                                                                                                                  | `'all'`        |
| `n_downsample` | `Optional[int]`                                              | If provided, downsample the data to this number of samples for faster visualization. Default is 10000. Set to None to disable downsampling.                                                                                                      | `10000`        |
| `**kwargs`     |                                                              | additional arguments.                                                                                                                                                                                                                            | `{}`           |

### `show_weights() -> None`

Display the model weights plot if it exists in the plots dictionary.

### `visualize(result: Result, config: DefaultConfig) -> Result`

Create the standard plots (model weights, absolute and relative loss curves).

Plots are stored in `self.plots` and shown with the `show_*` methods.

Parameters:

| Name     | Type            | Description                                      | Default    |
| -------- | --------------- | ------------------------------------------------ | ---------- |
| `result` | `Result`        | Result object with losses and the trained model. | *required* |
| `config` | `DefaultConfig` | Configuration of the pipeline run.               | *required* |

Returns:

| Type     | Description                  |
| -------- | ---------------------------- |
| `Result` | The unchanged result object. |

Model weights are skipped for inputs with more than 3000 features; loss plots are skipped with a warning if no loss history is available (e.g. after loading a pipeline saved with `save_all=False`).

(LLM-generated docstring, pending human review.)

## `Imagix3DVisualizer`

Bases: `ImagixVisualizer`

Visualizer for 3D volume pipelines (`Imagix3D`).

Adds slice-based reconstruction grids, latent correlation, latent activity and latent traversal plots (see `Imagix3DEvaluator` for the underlying diagnostics).

(LLM-generated docstring, pending human review.)

### `show_image_recon_grid(result: Result, n_samples: int = 3, selection: str = 'random', metric: str = 'mse', chosen_split: str = 'test') -> None`

Plot original, reconstructed, and absolute-error slices for samples of the chosen split.

Parameters:

| Name           | Type     | Description                                                                 | Default    |
| -------------- | -------- | --------------------------------------------------------------------------- | ---------- |
| `result`       | `Result` | Result object.                                                              | *required* |
| `n_samples`    | `int`    | Number of random samples if selection="random". Ignored if selection="bmw". | `3`        |
| `selection`    | `str`    | "random" or "bmw". "bmw" selects best, median, and worst reconstruction.    | `'random'` |
| `metric`       | `str`    | Metric used for bmw selection. Options: "mse" or "bce".                     | `'mse'`    |
| `chosen_split` | `str`    | "test", "valid" or "train"                                                  | `'test'`   |

### `show_latent_activity(result: Result, final_epoch: Optional[int] = None, include_test: bool = True, show_table: bool = False, show_dim_table: bool = False, log_activity: bool = True) -> None`

Visualize latent-dimension activity statistics computed by Imagix3DEvaluator.

This method expects that the evaluator method

compute_latent_activity(...)

has already been called. The evaluator stores two tables in result.sub_results:

"latent_activity_summary" "latent_activity_by_dim"

The visualizer then creates three diagnostic plots:

1. Number of active latent dimensions across training epochs.
1. Activity per latent dimension at the final selected epoch.
1. Mean KL contribution per latent dimension at the final selected epoch, if KL values are available.

Parameters:

| Name             | Type            | Description                                                                                                                                  | Default    |
| ---------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `result`         | `Result`        | Result object containing latent-activity tables in result.sub_results.                                                                       | *required* |
| `final_epoch`    | `Optional[int]` | Stored epoch key to use for the per-dimension final-epoch plots. If None, the largest non-test epoch in the summary table is used.           | `None`     |
| `include_test`   | `bool`          | Whether to include the final test prediction rows in the final-epoch per-dimension plots, if they are available.                             | `True`     |
| `show_table`     | `bool`          | If True, display the summary table.                                                                                                          | `False`    |
| `show_dim_table` | `bool`          | If True, display the final-epoch per-dimension table.                                                                                        | `False`    |
| `log_activity`   | `bool`          | If True, use a log scale for the activity-per-dimension plot. This is often helpful because latent-dimension variances can be highly skewed. | `True`     |

### `show_latent_corr(result: Result, show_table: bool = False, show_corrmat: bool = False) -> None`

Show how correlated the latent dimensions are over training.

For each stored epoch and the train/valid splits, computes the mean and maximum absolute pairwise correlation between latent dimensions (near-zero variance dimensions are excluded) and plots them per epoch.

Parameters:

| Name           | Type     | Description                                                  | Default    |
| -------------- | -------- | ------------------------------------------------------------ | ---------- |
| `result`       | `Result` | Result object with stored latent spaces.                     | *required* |
| `show_table`   | `bool`   | If True, also display the summary table.                     | `False`    |
| `show_corrmat` | `bool`   | If True, also show the correlation matrix of the last epoch. | `False`    |

(LLM-generated docstring, pending human review.)

### `show_latent_traversal(result: Result, view: Literal['axial', 'coronal', 'sagittal'] = 'axial', channel: int = 0) -> None`

Visualize latent traversal volumes.

This method expects that the evaluator method compute_latent_traversal(...) has already been called.

The evaluator stores the decoded traversal volumes and metadata in

result.sub_results["latent_traversal"]

The figure layout is

rows = latent dimensions columns = traversal values

Parameters:

| Name      | Type                                      | Description                                                                                                                                               | Default    |
| --------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `result`  | `Result`                                  | Result object containing the latent traversal output in result.sub_results["latent_traversal"].                                                           | *required* |
| `view`    | `Literal['axial', 'coronal', 'sagittal']` | Anatomical plane to visualize. Options: "axial": slice along the depth axis "coronal": slice along the height axis "sagittal": slice along the width axis | `'axial'`  |
| `channel` | `int`                                     | Image channel to visualize. For single-channel 3D images, use 0.                                                                                          | `0`        |

## `ImagixVisualizer`

Bases: `GeneralVisualizer`

Visualizer for 2D image pipelines (`Imagix`).

Adds reconstruction grids (original versus reconstructed images) to the standard loss and latent space plots of `GeneralVisualizer`.

Attributes:

| Name    | Type             | Description                                         |
| ------- | ---------------- | --------------------------------------------------- |
| `plots` | `Dict[str, Any]` | Nested dict of created figures, keyed by plot name. |

(LLM-generated docstring, pending human review.)

### `show_image_recon_grid(result: Result, n_samples: int = 3) -> None`

Show original and reconstructed test images side by side.

Parameters:

| Name        | Type     | Description                                    | Default    |
| ----------- | -------- | ---------------------------------------------- | ---------- |
| `result`    | `Result` | Result object with test-split reconstructions. | *required* |
| `n_samples` | `int`    | Number of images to show.                      | `3`        |

(LLM-generated docstring, pending human review.)

### `show_weights() -> None`

Not implemented for image models.

Raises:

| Type                  | Description |
| --------------------- | ----------- |
| `NotImplementedError` | Always.     |

(LLM-generated docstring, pending human review.)

### `visualize(result: Result, config: DefaultConfig) -> Result`

Create the loss curves (absolute and relative) and store them in `self.plots`.

Parameters:

| Name     | Type            | Description                        | Default    |
| -------- | --------------- | ---------------------------------- | ---------- |
| `result` | `Result`        | Result object with losses.         | *required* |
| `config` | `DefaultConfig` | Configuration of the pipeline run. | *required* |

Returns:

| Type     | Description                  |
| -------- | ---------------------------- |
| `Result` | The unchanged result object. |

(LLM-generated docstring, pending human review.)

## `XModalVisualizer`

Bases: `BaseVisualizer`

Visualizer for cross-modal translation (`XModalix`).

Adds per-modality latent space plots, image translation grids and 2D embeddings of translated versus original data to the standard plots.

Attributes:

| Name    | Type             | Description                                         |
| ------- | ---------------- | --------------------------------------------------- |
| `plots` | `Dict[str, Any]` | Nested dict of created figures, keyed by plot name. |

(LLM-generated docstring, pending human review.)

### `show_2D_translation(result: Result, translated_modality: str, split: str = 'test', param: Optional[str] = None, reducer: str = 'UMAP') -> None`

Plot a 2D embedding of translated samples next to the original target data.

Reduces original and translated data of the target modality to 2D (UMAP, PCA or t-SNE) and shows both in one scatter plot.

Parameters:

| Name                  | Type            | Description                                          | Default    |
| --------------------- | --------------- | ---------------------------------------------------- | ---------- |
| `result`              | `Result`        | Result object with test-split translations.          | *required* |
| `translated_modality` | `str`           | Key of the target modality.                          | *required* |
| `split`               | `str`           | Data split; currently only "test" is supported.      | `'test'`   |
| `param`               | `Optional[str]` | Annotation column used for coloring.                 | `None`     |
| `reducer`             | `str`           | Dimension reduction method: "UMAP", "PCA" or "TSNE". | `'UMAP'`   |

Raises:

| Type                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `ValueError`          | If the split is unknown or no test data is available. |
| `NotImplementedError` | For splits other than "test".                         |

(LLM-generated docstring, pending human review.)

### `show_image_translation(result: Result, from_key: str, to_key: str, n_sample_per_class: int = 3, param: Optional[str] = None) -> None`

Visualizes image translation results for a given dataset.

Shows a grid of original, translated, and reference images, grouped by class values. Only the test split is supported.

Parameters:

| Name                 | Type            | Description                                                                             | Default    |
| -------------------- | --------------- | --------------------------------------------------------------------------------------- | ---------- |
| `result`             | `Result`        | The result object containing datasets and reconstructions.                              | *required* |
| `from_key`           | `str`           | The source modality key (not directly used in visualization, but relevant for context). | *required* |
| `to_key`             | `str`           | The target modality key. Must correspond to an image dataset (must contain "img").      | *required* |
| `n_sample_per_class` | `int`           | Number of samples to display per class value. Default is 3.                             | `3`        |
| `param`              | `Optional[str]` | The metadata column name used to group samples by class.                                | `None`     |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | If to_key does not correspond to an image dataset. |

### `show_latent_space(result: Result, plot_type: str = '2D-scatter', labels: Optional[Union[list, pd.Series, None]] = None, param: Optional[Union[list, str]] = None, epoch: Optional[Union[int, None]] = None, split: str = 'all') -> None`

Display the latent spaces of all modalities, colored by annotation.

Parameters:

| Name        | Type                                  | Description                                             | Default        |
| ----------- | ------------------------------------- | ------------------------------------------------------- | -------------- |
| `result`    | `Result`                              | Result object containing the latent spaces.             | *required*     |
| `plot_type` | `str`                                 | "2D-scatter", "Ridgeline" or "Coverage-Correlation".    | `'2D-scatter'` |
| `labels`    | `Optional[Union[list, Series, None]]` | Optional explicit labels per sample used for coloring.  | `None`         |
| `param`     | `Optional[Union[list, str]]`          | Annotation column(s) used for coloring, or "all".       | `None`         |
| `epoch`     | `Optional[Union[int, None]]`          | Epoch to plot; defaults to the last stored epoch.       | `None`         |
| `split`     | `str`                                 | Data split to plot ("train", "valid", "test" or "all"). | `'all'`        |

(LLM-generated docstring, pending human review.)

### `show_weights() -> None`

Not implemented for XModalix.

Raises:

| Type                  | Description |
| --------------------- | ----------- |
| `NotImplementedError` | Always.     |

(LLM-generated docstring, pending human review.)

### `visualize(result: Result, config: DefaultConfig) -> Result`

Create the loss curves (absolute and relative) and store them in `self.plots`.

Parameters:

| Name     | Type            | Description                        | Default    |
| -------- | --------------- | ---------------------------------- | ---------- |
| `result` | `Result`        | Result object with losses.         | *required* |
| `config` | `DefaultConfig` | Configuration of the pipeline run. | *required* |

Returns:

| Type     | Description                  |
| -------- | ---------------------------- |
| `Result` | The unchanged result object. |

(LLM-generated docstring, pending human review.)
