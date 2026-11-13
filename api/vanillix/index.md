# Vanillix Module

## `Vanillix`

Bases: `BasePipeline`

Vanillix specific version of the BasePipeline class.

Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline. This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to VanillixConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = GeneralTrainer, dataset_type: Type[BaseDataset] = NumericDataset, model_type: Type[BaseAutoencoder] = VanillixArchitecture, loss_type: Type[BaseLoss] = VanillixLoss, preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor, visualizer: Type[BaseVisualizer] = GeneralVisualizer, evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[DefaultConfig] = None, ontologies: Optional[Union[List, Dict]] = None) -> None`

Initialize Vanillix pipeline with customizable components.

Some components are passed as types rather than instances because they require data that is only available after preprocessing.

See implementation of parent class for list of full Args.

### `sample_latent_space(n_samples: int, split: str = 'test', epoch: int = -1) -> torch.Tensor`

Samples latent space points from the empirical latent distribution.

This method draws new latent points by fitting a diagonal Gaussian distribution to the latent codes of the specified split and epoch, and sampling from it. This enables approximate generative sampling for autoencoders that do not model uncertainty explicitly.

Parameters:

| Name        | Type  | Description                                                        | Default    |
| ----------- | ----- | ------------------------------------------------------------------ | ---------- |
| `n_samples` | `int` | The number of latent points to sample. Must be a positive integer. | *required* |
| `split`     | `str` | The split to sample from (train, valid, test), default is test.    | `'test'`   |
| `epoch`     | `int` | The epoch to sample from, default is the last epoch (-1).          | `-1`       |

Returns:

| Name | Type     | Description                                     |
| ---- | -------- | ----------------------------------------------- |
| `z`  | `Tensor` | torch.Tensor - The sampled latent space points. |

Raises:

| Type         | Description                                                                                                     |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the model has not been trained, latent codes have not been computed, or n_samples is not a positive integer. |
| `TypeError`  | If the stored latent codes are not numpy arrays.                                                                |
