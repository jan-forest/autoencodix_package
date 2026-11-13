# Trainers Module

## `GeneralTrainer`

Bases: `BaseTrainer`

Handles general training logic for autoencoder models.

Attributes:

| Name                     | Type            | Description                                                                        |
| ------------------------ | --------------- | ---------------------------------------------------------------------------------- |
| `_trainset`              |                 | The dataset used for training.                                                     |
| `_validset`              |                 | The dataset used for validation, if provided.                                      |
| `_result`                |                 | An object to store and manage training results.                                    |
| `_config`                |                 | Configuration object containing training hyperparameters and settings.             |
| `_model_type`            |                 | The autoencoder model class to be trained.                                         |
| `_loss_fn`               |                 | Instantiated loss function specific to the model.                                  |
| `_trainloader`           |                 | DataLoader for the training dataset.                                               |
| `_validloader`           |                 | DataLoader for the validation dataset, if provided.                                |
| `_model`                 |                 | The instantiated model architecture.                                               |
| `_optimizer`             |                 | The optimizer used for training.                                                   |
| `_fabric`                |                 | Lightning Fabric wrapper for device and precision management.                      |
| `n_train`                |                 | Number of training samples.                                                        |
| `n_valid`                |                 | Number of validation samples.                                                      |
| `n_test`                 | `Optional[int]` | Number of test samples (set during prediction).                                    |
| `n_features`             |                 | Number of input features.                                                          |
| `latent_dim`             |                 | Dimensionality of the latent space.                                                |
| `device`                 |                 | Device on which the model is located.                                              |
| `_n_cpus`                |                 | Number of CPU cores available.                                                     |
| `_reconstruction_buffer` |                 | Buffer to store reconstructions during training/validation/testing.                |
| `_latentspace_buffer`    |                 | Buffer to store latent representations during training/validation/testing.         |
| `_mu_buffer`             |                 | Buffer to store latent means (for VAE) during training/validation                  |
| `_sigma_buffer`          |                 | Buffer to store latent log-variances (for VAE) during training/validation/testing. |
| `_sample_ids_buffer`     |                 | Buffer to store sample IDs during training/validation/testing.                     |

### `__init__(trainset: Optional[BaseDataset], validset: Optional[BaseDataset], result: Result, config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], ontologies: Optional[Union[Tuple, List]] = None, **kwargs)`

Initializes the GeneralTrainer.

Parameters:

| Name         | Type                           | Description                                                            | Default    |
| ------------ | ------------------------------ | ---------------------------------------------------------------------- | ---------- |
| `trainset`   | `Optional[BaseDataset]`        | The dataset used for training.                                         | *required* |
| `validset`   | `Optional[BaseDataset]`        | The dataset used for validation, if provided.                          | *required* |
| `result`     | `Result`                       | An object to store and manage training results.                        | *required* |
| `config`     | `DefaultConfig`                | Configuration object containing training hyperparameters and settings. | *required* |
| `model_type` | `Type[BaseAutoencoder]`        | The autoencoder model class to be trained.                             | *required* |
| `loss_type`  | `Type[BaseLoss]`               | The loss function class specific to the model.                         | *required* |
| `ontologies` | `Optional[Union[Tuple, List]]` | Ontology information, if provided for Ontix.                           | `None`     |
| `**kwargs`   |                                | Additional keyword arguments passed to BaseTrainer.                    | `{}`       |

### `decode(x: torch.Tensor)`

Decodes the input tensor x using the trained model.

Parameters:

| Name | Type     | Description                 | Default    |
| ---- | -------- | --------------------------- | ---------- |
| `x`  | `Tensor` | Input tensor to be decoded. | *required* |

Returns: Decoded tensor.

### `get_model() -> torch.nn.Module`

Getter method for the trained model.

Returns:

```
The trained model as a torch.nn.Module.
```

### `maskix_hook(X: torch.Tensor)`

Only active when override from MaskixTrainer is used

### `predict(data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs) -> Result`

Decided to add predict method to the trainer class.

This violates SRP, but the trainer class has a lot of attributes and methods that are needed for prediction. So this way we don't need to write so much duplicate code

Parameters:

| Name       | Type               | Description                                 | Default    |
| ---------- | ------------------ | ------------------------------------------- | ---------- |
| `data`     | `BaseDataset`      | BaseDataset unseen data to run inference on | *required* |
| `model`    | `Optional[Module]` | torch.nn.Module model to run inference with | `None`     |
| `**kwargs` |                    | Additional arguments (not used here).       | `{}`       |

Returns:

| Type     | Description                                                   |
| -------- | ------------------------------------------------------------- |
| `Result` | self.\_result: Result object containing the inference results |

### `purge() -> None`

Cleans up any resources used during training, such as cached data or large attributes.

### `supervisix_capture_hook(model_output: ModelOutput, sample_ids: list[str], dataset_type: str, batch_class_means: dict[str, torch.Tensor])`

Only active when override from SupervisixTrainer is used

### `supervisix_get_hook(dataset_type: str)`

Only active when override from SupervisixTrainer is used

### `supervisix_update_hook(batch_class_means: dict[str, torch.Tensor], dataset_type: str)`

Only active when override from SupervisixTrainer is used

### `train(epochs_overwrite=None) -> Result`

Orchestrates training over multiple epochs, including training and validation phases.

Parameters:

| Name               | Type | Description                                                                                                                                | Default |
| ------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `epochs_overwrite` |      | If provided, overrides the number of epochs specified in the config. This is only there so we can use the train method for pretraining.Any | `None`  |

Returns: Result object containing training results and dynamics like latent spaces and reconstructions.

## `StackixOrchestrator`

StackixOrchestrator coordinates the training of multi-modality VAE stacking.

This orchestrator manages both parallel and sequential training of modality-specific autoencoders, followed by creating a concatenated latent space for the final stacked model training. It leverages Lightning Fabric's distribution strategies for efficient training.

Attributes: \_workdir: Directory for saving intermediate models and results modality_models: Dictionary of trained models for each modality modality_results: Dictionary of training results for each modality \_modality_latent_dims: Dictionary of latent dimensions for each modality concatenated_latent_spaces: Dictionary of concatenated latent spaces by split \_dataset_type: Class to use for creating datasets \_fabric: Lightning Fabric instance for distributed operations \_trainer_class: Class to use for training (dependency injection) trainset: Training dataset containing multiple modalities validset: Validation dataset containing multiple modalities testset: Test dataset containing multiple modalities loss_type: Type of loss function to use for training model_type: Type of autoencoder model to use for each modality config: Configuration parameters for training and model architecture stacked_model: The final stacked autoencoder model (initialized later) stacked_trainer: Trainer for the stacked model (initialized later) concat_idx: Dictionary tracking the start and end indices of each modality in the concatenated latent space dropped_indices_map: Dictionary tracking dropped sample indices for each modality during alignment reconstruction_shapes: Dictionary storing original shapes of latent spaces for reconstruction common_sample_ids: Common sample IDs across all modalities for alignment

### `__init__(trainset: Optional[MultiModalDataset], validset: Optional[MultiModalDataset], config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], testset: Optional[MultiModalDataset] = None, trainer_type: Type[GeneralTrainer] = GeneralTrainer, dataset_type: Type[BaseDataset] = NumericDataset, workdir: str = './stackix_work')`

Initialize the StackixOrchestrator with datasets and configuration.

Parameters:

| Name           | Type                          | Description                                                                     | Default            |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| `trainset`     | `Optional[MultiModalDataset]` | Training dataset containing multiple modalities                                 | *required*         |
| `validset`     | `Optional[MultiModalDataset]` | Validation dataset containing multiple modalities                               | *required*         |
| `config`       | `DefaultConfig`               | Configuration parameters for training and model architecture                    | *required*         |
| `model_type`   | `Type[BaseAutoencoder]`       | Type of autoencoder model to use for each modality                              | *required*         |
| `loss_type`    | `Type[BaseLoss]`              | Type of loss function to use for training                                       | *required*         |
| `testset`      | `Optional[MultiModalDataset]` | Dataset with test split                                                         | `None`             |
| `trainer_type` | `Type[GeneralTrainer]`        | Type to use for training (default is GeneralTrainer)                            | `GeneralTrainer`   |
| `dataset_type` | `Type[BaseDataset]`           | Type to use for creating datasets (default is NumericDataset)                   | `NumericDataset`   |
| `workdir`      | `str`                         | Directory to save intermediate models and results (default is "./stackix_work") | `'./stackix_work'` |

### `predict_modalities(data: MultiModalDataset) -> Dict[str, torch.Tensor]`

Predicts using the trained models for each modality.

Parameters:

| Name   | Type                | Description                                               | Default    |
| ------ | ------------------- | --------------------------------------------------------- | ---------- |
| `data` | `MultiModalDataset` | Input data for prediction, uses test data if not provided | *required* |

Returns:

| Type                | Description                                     |
| ------------------- | ----------------------------------------------- |
| `Dict[str, Tensor]` | Dictionary of reconstructed tensors by modality |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If model has not been trained yet or no data is available |

### `prepare_latent_datasets(split: str) -> NumericDataset`

Prepares a dataset with concatenated latent spaces for stacked model training.

This is the second phase of Stackix training where latent spaces from all modalities are extracted and concatenated.

Parameters:

| Name    | Type  | Description                                         | Default    |
| ------- | ----- | --------------------------------------------------- | ---------- |
| `split` | `str` | Data split to prepare ("train", "valid" or "test"). | *required* |

Returns:

| Type             | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `NumericDataset` | Dataset of concatenated per-modality latent spaces for the samples |
| `NumericDataset` | present in all modalities.                                         |

Raises:

| Type         | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `ValueError` | If no modality models have been trained or no latent spaces could be extracted |

### `reconstruct_from_stack(reconstructed_stack: torch.Tensor) -> Dict[str, torch.Tensor]`

Reconstructs the full data for each modality from the stacked latent reconstruction.

Parameters:

| Name                  | Type     | Description                                             | Default    |
| --------------------- | -------- | ------------------------------------------------------- | ---------- |
| `reconstructed_stack` | `Tensor` | Tensor with the reconstructed concatenated latent space | *required* |

Returns: Dictionary of reconstructed tensors by modality

### `set_testset(testset: MultiModalDataset) -> None`

Set the test dataset for the orchestrator.

Parameters:

| Name      | Type                | Description                                 | Default    |
| --------- | ------------------- | ------------------------------------------- | ---------- |
| `testset` | `MultiModalDataset` | Test dataset containing multiple modalities | *required* |

### `train_modalities() -> Tuple[Dict[str, BaseAutoencoder], Dict[str, Result]]`

Trains all modality-specific models.

This is the first phase of Stackix training where each modality is trained independently before their latent spaces are combined.

Returns:

| Type                         | Description                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `Dict[str, BaseAutoencoder]` | Dictionary of trained models for each modality and Dictionary of training results |
| `Dict[str, Result]`          | for each modality.                                                                |

Raises:

| Type         | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| `ValueError` | If trainset is not a MultiModalDataset or has no modalities |

## `StackixTrainer`

Bases: `GeneralTrainer`

StackixTrainer is a wrapper for StackixOrchestrator that conforms to the BaseTrainer interface.

This trainer maintains compatibility with the BasePipeline interface while leveraging the more modular and well-designed StackixOrchestrator classes for the actual implementation.

Attributes: \_workdir: Directory for saving intermediate models and results \_result: Result object to store training outcomes \_config: Configuration parameters for training and model architecture \_model_type: Type of autoencoder model to use for each modality \_loss_type: Type of loss function to use for training \_trainset: Training dataset containing multiple modalities \_validset: Validation dataset containing multiple modalities \_orchestrator_type: Type to use for orchestrating modality training (default is StackixOrchestrator) \_trainer_type: Type to use for training each modality model (default is GeneralTrainer) \_modality_trainers: Dictionary of trained models for each modality \_modality_results: Dictionary of training results for each modality \_trainer: Trainer for the stacked model \_fabric: Lightning Fabric wrapper for device and precision management \_train_latent_ds: Training dataset with concatenated latent spaces \_valid_latent_ds: Validation dataset with concatenated latent spaces concat_idx: Indices used for concatenating latent spaces \_model: The instantiated stacked model architecture \_optimizer: The optimizer used for training \_orchestrator: The orchestrator that manages modality model training and latent space preparation \_workdir: Directory for saving intermediate models and results

### `__init__(trainset: Optional[StackixDataset], validset: Optional[StackixDataset], result: Result, config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], orchestrator_type: Type[StackixOrchestrator] = StackixOrchestrator, trainer_type: Type[BaseTrainer] = GeneralTrainer, workdir: str = './stackix_work', ontologies: Optional[Tuple] = None, **kwargs) -> None`

Initialize the StackixTrainer with datasets and configuration.

Parameters:

| Name                | Type                        | Description                                                                      | Default               |
| ------------------- | --------------------------- | -------------------------------------------------------------------------------- | --------------------- |
| `trainset`          | `Optional[StackixDataset]`  | Training dataset containing multiple modalities                                  | *required*            |
| `validset`          | `Optional[StackixDataset]`  | Validation dataset containing multiple modalities                                | *required*            |
| `result`            | `Result`                    | Result object to store training outcomes                                         | *required*            |
| `config`            | `DefaultConfig`             | Configuration parameters for training and model architecture                     | *required*            |
| `model_type`        | `Type[BaseAutoencoder]`     | Type of autoencoder model to use for each modality                               | *required*            |
| `loss_type`         | `Type[BaseLoss]`            | Type of loss function to use for training                                        | *required*            |
| `orchestrator_type` | `Type[StackixOrchestrator]` | Type to use for orchestrating modality training (default is StackixOrchestrator) | `StackixOrchestrator` |
| `trainer_type`      | `Type[BaseTrainer]`         | Type to use for training each modality model (default is GeneralTrainer)         | `GeneralTrainer`      |
| `workdir`           | `str`                       | Directory to save intermediate models and results (default is "./stackix_work")  | `'./stackix_work'`    |
| `ontologies`        | `Optional[Tuple]`           | Ontology information, if provided for Ontix compatibility                        | `None`                |
| `**kwargs`          |                             | Additional keyword arguments passed to BaseTrainer.                              | `{}`                  |

### `get_model() -> torch.nn.Module`

Getter for the the trained model.

Returns:

| Type     | Description       |
| -------- | ----------------- |
| `Module` | The trained model |

### `predict(data: BaseDataset, model: Optional[torch.nn.Module] = None, **kwargs) -> Result`

Make predictions on the given dataset.

Parameters:

| Name       | Type               | Description                                                        | Default    |
| ---------- | ------------------ | ------------------------------------------------------------------ | ---------- |
| `data`     | `BaseDataset`      | The dataset to make predictions on.                                | *required* |
| `model`    | `Optional[Module]` | The model to use for predictions. If None, uses the trained model. | `None`     |
| `**kwargs` |                    | Additional keyword arguments.                                      | `{}`       |

Returns: Result: The prediction results including reconstructions and latent spaces.

### `train() -> Result`

Train the stacked model on the concatenated latent space.

Uses the standard BaseTrainer training process but with the stacked model.

Returns:

| Type     | Description                                                         |
| -------- | ------------------------------------------------------------------- |
| `Result` | Training results including losses, latent spaces, and other metrics |

## `SupervisixTrainer`

Bases: `GeneralTrainer`

Specialized trainer for Supervisix (supervised) autoencoders.

Handles retaining and passing on the mean loss per class in the data set for the previous epoch. Uses most of the functionality from the GeneralTrainer class and supervisix-specific functionality is added via a hook. Inherits all attributes from GeneralTrainer.

### `__init__(trainset: Optional[BaseDataset], validset: Optional[BaseDataset], result: Result, config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], **kwargs)`

Initializes the SupervisixTrainer with the given datasets, model, and configuration.

Parameters:

| Name         | Type                    | Description                                                            | Default    |
| ------------ | ----------------------- | ---------------------------------------------------------------------- | ---------- |
| `trainset`   | `Optional[BaseDataset]` | The dataset used for training.                                         | *required* |
| `validset`   | `Optional[BaseDataset]` | The dataset used for validation, if provided.                          | *required* |
| `result`     | `Result`                | An object to store and manage training results.                        | *required* |
| `config`     | `DefaultConfig`         | Configuration object containing training hyperparameters and settings. | *required* |
| `model_type` | `Type[BaseAutoencoder]` | The autoencoder model class to be trained.                             | *required* |
| `loss_type`  | `Type[BaseLoss]`        | The loss function class specific to the model.                         | *required* |
| `**kwargs`   |                         | Additional keyword arguments passed to BaseTrainer.                    | `{}`       |

### `supervisix_capture_hook(model_output: ModelOutput, sample_ids: List[str], dataset_type: str, batch_class_means: Dict[str, torch.Tensor])`

Capture latent representations for each class in the dataset per batch in the current epoch.

Parameters:

| Name                | Type                | Description                                                                               | Default    |
| ------------------- | ------------------- | ----------------------------------------------------------------------------------------- | ---------- |
| `model_output`      | `ModelOutput`       | The output from the model's forward pass.                                                 | *required* |
| `sample_ids`        | `List[str]`         | The identifiers for the samples in the current batch.                                     | *required* |
| `dataset_type`      | `str`               | A string indicating whether the data is from 'train' or 'valid'.                          | *required* |
| `batch_class_means` | `Dict[str, Tensor]` | A dictionary to store the mean latent representation for each class in the current batch. | *required* |

### `supervisix_get_hook(dataset_type: str) -> Dict[str, torch.Tensor]`

Get the last epoch class means for the current dataset type.

Parameters:

| Name           | Type  | Description                                                      | Default    |
| -------------- | ----- | ---------------------------------------------------------------- | ---------- |
| `dataset_type` | `str` | A string indicating whether the data is from 'train' or 'valid'. | *required* |

Returns:

| Type                | Description                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `Dict[str, Tensor]` | A dictionary containing the mean latent representations for each class from the last epoch.                  |
| `Dict[str, Tensor]` | During the first training epoch, this will return an emtpy dictionary, wich is handled by the loss function. |

### `supervisix_update_hook(batch_class_means: Dict[str, torch.Tensor], dataset_type: str)`

Update epoch class means at the end of each epoch.

Parameters:

| Name                | Type                | Description                                                                                  | Default    |
| ------------------- | ------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| `batch_class_means` | `Dict[str, Tensor]` | A dictionary containing the mean latent representations for each class in the current batch. | *required* |
| `dataset_type`      | `str`               | A string indicating whether the data is from 'train' or 'valid'.                             | *required* |

## `XModalTrainer`

Bases: `BaseTrainer`

Trainer for cross-modal autoencoders, implements multimodal training with adversarial component.

Attributes:

| Name                   | Type             | Description                                                                    |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `_trainset`            |                  | The dataset used for training, must be a MultiModalDataset.                    |
| `_n_train_modalities`  |                  | Number of modalities in the training dataset.                                  |
| `model_map`            |                  | Mapping from DataSetTypes to specific autoencoder architectures.               |
| `model_trainer_map`    |                  | Mapping from autoencoder architectures to their corresponding trainer classes. |
| `_n_cpus`              |                  | Number of CPU cores available for data loading.                                |
| `latent_dim`           |                  | Dimensionality of the shared latent space.                                     |
| `n_test`               | `Optional[int]`  | Number of samples in the test set, if provided.                                |
| `n_train`              |                  | Number of samples in the training set.                                         |
| `n_valid`              |                  | Number of samples in the validation set, if provided.                          |
| `n_features`           |                  | Total number of features across all modalities.                                |
| `_cur_epoch`           | `int`            | Current epoch number during training.                                          |
| `_is_checkpoint_epoch` | `Optional[bool]` | Flag indicating if the current epoch is a checkpoint epoch.                    |
| `_sub_loss_type`       |                  | Loss function type for individual modality autoencoders.                       |
| `sub_loss`             |                  | Instantiated loss function for individual modality autoencoders.               |
| `_clf_epoch_loss`      | `float`          | Cumulative classifier loss for the current epoch.                              |
| `_epoch_loss`          | `float`          | Cumulative total loss for the current epoch.                                   |
| `_epoch_loss_valid`    | `float`          | Cumulative total validation loss for the current epoch.                        |
| `_modality_dynamics`   | `float`          | Dictionary holding model, optimizer, and training state for each modality.     |
| `_latent_clf`          | `float`          | Classifier model for adversarial training on latent spaces.                    |
| `_clf_optim`           | `float`          | Optimizer for the classifier model.                                            |
| `_clf_loss_fn`         | `float`          | Loss function for the classifier.                                              |
| `_trainloader`         | `float`          | DataLoader for the training dataset.                                           |
| `_validloader`         | `float`          | DataLoader for the validation dataset, if provided.                            |
| `_model`               | `float`          | The instantiated stacked model architecture.                                   |
| `_validset`            |                  | The dataset used for validation, if provided, must be a MultiModalDataset.     |
| `_fabric`              |                  | Lightning Fabric wrapper for device and precision management.                  |

### `__init__(trainset: MultiModalDataset, validset: MultiModalDataset, result: Result, config: DefaultConfig, model_type: Type[BaseAutoencoder], loss_type: Type[BaseLoss], model_map: Dict[DataSetTypes, Type[BaseAutoencoder]] = {DataSetTypes.NUM: VarixArchitecture, DataSetTypes.IMG: ImageVAEArchitecture}, sub_loss_type: Type[BaseLoss] = VarixLoss, ontologies: Optional[Union[Tuple, List]] = None, **kwargs)`

Initializes the XModalTrainer with datasets and configuration.

Parameters:

| Name            | Type                                        | Description                                                            | Default                                               |
| --------------- | ------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------- |
| `trainset`      | `MultiModalDataset`                         | Training dataset containing multiple modalities                        | *required*                                            |
| `validset`      | `MultiModalDataset`                         | Validation dataset containing multiple modalities                      | *required*                                            |
| `result`        | `Result`                                    | Result object to store training outcomes                               | *required*                                            |
| `config`        | `DefaultConfig`                             | Configuration parameters for training and model architecture           | *required*                                            |
| `model_type`    | `Type[BaseAutoencoder]`                     | Type of autoencoder model to use for each modality (not used directly) | *required*                                            |
| `loss_type`     | `Type[BaseLoss]`                            | Type of loss function to use for training the stacked model            | *required*                                            |
| `model_map`     | `Dict[DataSetTypes, Type[BaseAutoencoder]]` | Mapping from DataSetTypes to specific autoencoder architectures        | `{NUM: VarixArchitecture, IMG: ImageVAEArchitecture}` |
| `sub_loss_type` | `Type[BaseLoss]`                            | Type of loss function to use for individual modality autoencoders      | `VarixLoss`                                           |
| `ontologies`    | `Optional[Union[Tuple, List]]`              | Ontology information, for compatibility with Ontix                     | `None`                                                |
| `**kwargs`      |                                             | Additional keyword arguments passed to BaseTrainer.                    | `{}`                                                  |

### `decode(x: torch.Tensor)`

Decodes input latent representations Args: x: Latent representations to decode, shape (n_samples, latent_dim)

### `predict(data: BaseDataset, model: Optional[Dict[str, torch.nn.Module]] = None, **kwargs) -> Result`

Performs cross-modal prediction from a specified 'from' modality to a 'to' modality.

The direction is determined by the 'translate_direction' attribute in the config. Results are stored in the Result object under split='test' and epoch=-1.

Parameters:

| Name       | Type                          | Description                                                                                                   | Default    |
| ---------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| `data`     | `BaseDataset`                 | A MultiModalDataset containing the input data for the 'from' modality.                                        | *required* |
| `model`    | `Optional[Dict[str, Module]]` | Trained models per modality (e.g. after loading a saved pipeline); the trainer's own models are used if None. | `None`     |
| `**kwargs` |                               | Ignored; accepted for interface compatibility.                                                                | `{}`       |

Returns:

| Type     | Description                                          |
| -------- | ---------------------------------------------------- |
| `Result` | The Result object populated with prediction results. |

### `purge() -> None`

Cleans up all instantiated resources used during training, including all modality-specific models/optimizers and the adversarial classifier.

### `train()`

Orchestrates the full training process for the cross-modal autoencoder.
