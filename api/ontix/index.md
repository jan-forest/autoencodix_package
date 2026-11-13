# Ontix Module

## `Ontix`

Bases: `BasePipeline`

Ontix specific version of the BasePipeline class.

Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to OntixConfig here.

### `__init__(ontologies: Union[Tuple, List], sep: Optional[str] = '\t', data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = OntixTrainer, dataset_type: Type[BaseDataset] = NumericDataset, model_type: Type[BaseAutoencoder] = OntixArchitecture, loss_type: Type[BaseLoss] = VarixLoss, preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor, visualizer: Type[BaseVisualizer] = GeneralVisualizer, evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[DefaultConfig] = None) -> None`

Initialize Ontix pipeline with customizable components.

Some components are passed as types rather than instances because they require data that is only available after preprocessing.

See parent class for full list of Arguments.

Raises:

| Type        | Description                            |
| ----------- | -------------------------------------- |
| `TypeError` | if ontologies are not a Tuple or List. |
