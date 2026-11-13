# XModalix Module

## `XModalix`

Bases: `BasePipeline`

XModalix specific version of the BasePipeline class.

Inherits preprocess, fit, evaluate, and visualize methods from BasePipeline. Overrides predict

This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to XModalixConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = XModalTrainer, dataset_type: Type[BaseDataset] = MultiModalDataset, model_type: Type[BaseAutoencoder] = VarixArchitecture, loss_type: Type[BaseLoss] = XModalLoss, preprocessor_type: Type[BasePreprocessor] = XModalPreprocessor, visualizer: Optional[Type[BaseVisualizer]] = XModalVisualizer, evaluator: Optional[Type[XModalixEvaluator]] = XModalixEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[DefaultConfig] = None, model_map: Dict[DataSetTypes, Type[BaseAutoencoder]] = {DataSetTypes.NUM: VarixArchitecture, DataSetTypes.IMG: ImageVAEArchitecture}) -> None`

See base class for full list of Args.

### `predict(data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]] = None, config: Optional[Union[None, DefaultConfig]] = None, from_key: Optional[str] = None, to_key: Optional[str] = None, predict_all: bool = False, **kwargs)`

Generates predictions using the trained model.

Uses the trained model to make predictions on test data or new data provided by the user. Processes the results and stores them in the result container.

Parameters:

| Name          | Type                                                              | Description                                                                                          | Default |
| ------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `data`        | `Optional[Union[DataPackage, DatasetContainer, AnnData, MuData]]` | Optional new data for predictions.                                                                   | `None`  |
| `config`      | `Optional[Union[None, DefaultConfig]]`                            | Optional custom configuration for prediction.                                                        | `None`  |
| `from_key`    | `Optional[str]`                                                   | string indicator of 'from' translation direction.                                                    | `None`  |
| `to_key`      | `Optional[str]`                                                   | string indicator of 'to' translation direction.                                                      | `None`  |
| `predict_all` | `bool`                                                            | If True, also predict the train and valid splits of the preprocessed data (not only the test split). | `False` |
| `**kwargs`    |                                                                   | Additional configuration parameters as keyword arguments.                                            | `{}`    |

Returns:

| Type | Description                |
| ---- | -------------------------- |
|      | The updated Result object. |

Raises:

| Type                  | Description                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `NotImplementedError` | If required components aren't initialized.                                                              |
| `ValueError`          | If no test data is available or data format is invalid, or predict_all is set without train/valid data. |

### `sample_latent_space(n_samples: int, split: str = 'test', epoch: int = -1) -> torch.Tensor`

Not supported for XModalix.

Parameters:

| Name        | Type  | Description                | Default    |
| ----------- | ----- | -------------------------- | ---------- |
| `n_samples` | `int` | Number of samples to draw. | *required* |
| `split`     | `str` | Data split to sample from. | `'test'`   |
| `epoch`     | `int` | Epoch to sample from.      | `-1`       |

Raises:

| Type                  | Description |
| --------------------- | ----------- |
| `NotImplementedError` | Always.     |

(LLM-generated docstring, pending human review.)

### `show_result()`

Displays key visualizations of model results.

This method generates the following visualizations:

1. Loss Curves: Displays the absolute loss curves to provide insights into the model's training and validation performance over epochs.
1. Latent Space Ridgeline Plot: Visualizes the distribution of the latent space representations across different dimensions, offering a high-level overview of the learned embeddings.
1. Latent Space 2D Scatter Plot: Projects the latent space into two dimensions for a detailed view of the clustering or separation of data points.

These visualizations help in understanding the model's performance and the structure of the latent space representations.
