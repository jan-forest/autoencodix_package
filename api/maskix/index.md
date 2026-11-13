# Maskix Module

## `Maskix`

Bases: `BasePipeline`

Maskix specific version of the BasePipeline class.

Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to OntixConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = MaskixTrainer, dataset_type: Type[BaseDataset] = NumericDataset, model_type: Type[BaseAutoencoder] = MaskixArchitectureVanilla, loss_type: Type[BaseLoss] = MaskixLoss, preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor, visualizer: Type[BaseVisualizer] = GeneralVisualizer, evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[DefaultConfig] = None, masking_fn: Optional[Callable] = None, masking_fn_kwargs: Dict[str, Any] = {}, **kwargs: dict) -> None`

Initialize Maskix pipeline with customizable components.

Some components are passed as types rather than instances because they require data that is only available after preprocessing.

See parent class for full list of Arguments.

Raises:

| Type        | Description                            |
| ----------- | -------------------------------------- |
| `TypeError` | if ontologies are not a Tuple or List. |

### `impute(corrupted_tensor: torch.Tensor) -> ModelOutput`

Impute missing values in the corrupted tensor using the trained Maskix model.

Parameters:

| Name               | Type     | Description                                 | Default    |
| ------------------ | -------- | ------------------------------------------- | ---------- |
| `corrupted_tensor` | `Tensor` | A tensor with missing values to be imputed. | *required* |

Returns: model_output: A tensor with imputed values.
