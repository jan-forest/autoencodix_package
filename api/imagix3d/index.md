# Imagix3D Module

## `Imagix3D`

Bases: `BasePipeline`

Imagix3D specific version of the BasePipeline class.

This class extends BasePipeline. See the parent class for a full list of attributes and methods.

Additional Attributes

\_default_config: Is set to DefaultConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = GeneralTrainer, dataset_type: Type[BaseDataset] = ImageDataset, model_type: Type[BaseAutoencoder] = VolumeVAEArchitecture, loss_type: Type[BaseLoss] = VarixLoss, preprocessor_type: Type[BasePreprocessor] = VolumePreprocessor, visualizer: Optional[Type[BaseVisualizer]] = Imagix3DVisualizer, evaluator: Optional[Type[Imagix3DEvaluator]] = Imagix3DEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[Imagix3DConfig] = None) -> None`

Initialize the Imagix3D pipeline with customizable components.

Some components are passed as types rather than instances because they require data that is only available after preprocessing.

See Parentclass for full list of Args.
