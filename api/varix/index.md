# Varix Module

## `Varix`

Bases: `BasePipeline`

Varix specific version of the BasePipeline class.

Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline. This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to VarixConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = GeneralTrainer, dataset_type: Type[BaseDataset] = NumericDataset, model_type: Type[BaseAutoencoder] = VarixArchitecture, loss_type: Type[BaseLoss] = VarixLoss, preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor, visualizer: Type[BaseVisualizer] = GeneralVisualizer, evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, ontologies: Optional[Union[List, Dict]] = None, config: Optional[DefaultConfig] = None) -> None`

Initialize Varix pipeline with customizable components.

Some components are passed as types rather than instances because they require data that is only available after preprocessing.

See parent class for full list of Args.
