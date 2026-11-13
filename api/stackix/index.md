# Stackix Module

## `Stackix`

Bases: `BasePipeline`

Stackix pipeline for training multiple VAEs on different modalities and stacking their latent spaces.

This pipeline uses:

1. StackixPreprocessor to prepare data for multi-modality training
1. StackixTrainer to train individual VAEs, extract latent spaces, and train the final stacked model

Like other pipelines, it follows the standard BasePipeline interface and workflow.

Additional Attributes

\_default_config: Is set to StackixConfig here.

### `__init__(data: Optional[Union[DataPackage, DatasetContainer]] = None, trainer_type: Type[BaseTrainer] = StackixTrainer, dataset_type: Type[BaseDataset] = StackixDataset, model_type: Type[BaseAutoencoder] = VarixArchitecture, loss_type: Type[BaseLoss] = VarixLoss, preprocessor_type: Type[BasePreprocessor] = StackixPreprocessor, visualizer: Type[BaseVisualizer] = GeneralVisualizer, evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator, result: Optional[Result] = None, datasplitter_type: Type[DataSplitter] = DataSplitter, custom_splits: Optional[Dict[str, np.ndarray]] = None, config: Optional[DefaultConfig] = None, ontologies: Optional[Union[List, Dict]] = None) -> None`

Initialize the Stackix pipeline.

See parent class for full list of Args.
