# Utils Module

## `AnnDataConverter`

Utility class for converting datasets into AnnData or multimodal AnnData dictionaries.

### `dataset_to_adata(datasetcontainer: DatasetContainer, split: Literal['train', 'valid', 'test'] = 'train') -> Optional[Dict[str, ad.AnnData]]`

Convert a DatasetContainer split to an AnnData or multimodal AnnData dictionary.

Parameters:

| Name               | Type                                | Description                                        | Default    |
| ------------------ | ----------------------------------- | -------------------------------------------------- | ---------- |
| `datasetcontainer` | `DatasetContainer`                  | Container holding train/valid/test datasets.       | *required* |
| `split`            | `Literal['train', 'valid', 'test']` | The dataset split to convert. Defaults to "train". | `'train'`  |

Returns:

| Type                           | Description                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `Optional[Dict[str, AnnData]]` | A single AnnData object (for NumericDataset) or a dictionary of AnnData objects (for MultiModalDataset). |

Raises:

| Type                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `ValueError`          | If the specified split does not exist in the DatasetContainer. |
| `NotImplementedError` | If the dataset type is not supported.                          |

## `BulkDataReader`

Reads bulk data from files based on configuration.

Supports both paired and unpaired data reading strategies.

Attributes:

| Name     | Type | Description          |
| -------- | ---- | -------------------- |
| `config` |      | Configuration object |

### `__init__(config: DefaultConfig)`

Initialize the BulkDataReader with a configuration.

Parameters:

| Name     | Type            | Description                                                    | Default    |
| -------- | --------------- | -------------------------------------------------------------- | ---------- |
| `config` | `DefaultConfig` | Configuration object containing data paths and specifications. | *required* |

### `read_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]`

Read all data according to the configuration.

Returns:

| Type                                                | Description                                                 |
| --------------------------------------------------- | ----------------------------------------------------------- |
| `Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]` | A tuple containing (bulk_dataframes, annotation_dataframes) |

### `read_paired_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]`

Reads numeric paired data

Returns:

| Type                                                | Description                                                                                                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]` | Tuple containing two Dicts: 1. with name of the data as key and pandas DataFrame as value 2. with str 'paired' as key and a common annotaion/metadata as DataFrame |

### `read_unpaired_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]`

Read data without enforcing sample alignment across modalities.

Returns:

| Type                                                | Description                                                 |
| --------------------------------------------------- | ----------------------------------------------------------- |
| `Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]` | A tuple containing (bulk_dataframes, annotation_dataframes) |

## `ImageDataReader`

Reads and processes image data.

Reads all images from the specified directory, processes them, and returns a list of ImgData objects.

### `parse_image_to_tensor(image_path: Union[str, Path], to_h: Optional[int] = None, to_w: Optional[int] = None) -> np.ndarray`

Reads an image from the given path, optionally resizes it, and converts it to a tensor.

Parameters:

| Name         | Type               | Description                                               | Default    |
| ------------ | ------------------ | --------------------------------------------------------- | ---------- |
| `image_path` | `Union[str, Path]` | The path to the image file.                               | *required* |
| `to_h`       | `Optional[int]`    | The desired height of the output tensor, by default None. | `None`     |
| `to_w`       | `Optional[int]`    | The desired width of the output tensor, by default None.  | `None`     |

Returns:

| Type      | Description                      |
| --------- | -------------------------------- |
| `ndarray` | The processed image as a tensor. |

Raises:

| Type                   | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `FileNotFoundError`    | If the image path is invalid or the image cannot be read.                           |
| `ImageProcessingError` | If the image format is unsupported or an unexpected error occurs during processing. |

### `read_all_images_from_dir(img_dir: str, to_h: Optional[int], to_w: Optional[int], annotation_df: pd.DataFrame, is_paired: Union[bool, None] = None) -> List[ImgData]`

Reads all images from a specified directory, processes them, returns list of ImgData objects.

Parameters:

| Name            | Type                | Description                                     | Default    |
| --------------- | ------------------- | ----------------------------------------------- | ---------- |
| `img_dir`       | `str`               | The directory containing the images.            | *required* |
| `to_h`          | `Optional[int]`     | The desired height of the output tensors.       | *required* |
| `to_w`          | `Optional[int]`     | The desired width of the output tensors.        | *required* |
| `annotation_df` | `DataFrame`         | DataFrame containing image annotations.         | *required* |
| `is_paired`     | `Union[bool, None]` | Whether the images are paired with annotations. | `None`     |

Returns:

| Type            | Description                           |
| --------------- | ------------------------------------- |
| `List[ImgData]` | List of processed image data objects. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the annotation DataFrame is missing required columns. |

### `read_annotation_file(data_info: DataInfo) -> pd.DataFrame`

Reads annotation file and returns DataFrame with file contents Args: data_info: specific part of the Configuration object for input data Returns: DataFrame with annotation data.

### `read_data(config: DefaultConfig) -> Tuple[Dict[str, List[ImgData]], Dict[str, pd.DataFrame]]`

Read image data from the specified directory based on configuration.

Parameters:

| Name     | Type            | Description                                                 | Default    |
| -------- | --------------- | ----------------------------------------------------------- | ---------- |
| `config` | `DefaultConfig` | The configuration object containing the data configuration. | *required* |

Returns:

| Type                                                    | Description                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `Dict[str, List[ImgData]]`                              | A Tuple of Dicts:                                                              |
| `Dict[str, DataFrame]`                                  | Dict with type of image data as key and actual List of ImgData as value.       |
| `Tuple[Dict[str, List[ImgData]], Dict[str, DataFrame]]` | Dict with type of image data as key and DataFrame of annotation data as value. |

Raises:

| Type        | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| `Exception` | If no image data is found in the configuration or other validation errors occur. |

### `validate_image_path(image_path: Union[str, Path]) -> bool`

Checks if file extension is allowed:

Allowed are (independent of capitalization):

- jpg
- jpeg
- png
- tif
- tiff

Parameters:

| Name         | Type               | Description                  | Default    |
| ------------ | ------------------ | ---------------------------- | ---------- |
| `image_path` | `Union[str, Path]` | path or str of image to read | *required* |

## `ModelOutput`

A structured output dataclass for autoencoder models.

This class is used to encapsulate the outputs of autoencoder models in a consistent format, allowing for flexibility in the type of outputs returned by different architectures.

Attributes:

| Name              | Type               | Description                                                                         |
| ----------------- | ------------------ | ----------------------------------------------------------------------------------- |
| `reconstruction`  | `Tensor`           | The reconstructed input data.                                                       |
| `latent_mean`     | `Optional[Tensor]` | The mean of the latent space distribution, applicable for models like VAEs.         |
| `latent_logvar`   | `Optional[Tensor]` | The log variance of the latent space distribution, applicable for models like VAEs. |
| `additional_info` | `Optional[dict]`   | A dictionary to store any additional information or intermediate outputs.           |

## `Result`

A dataclass to store results from the pipeline with predefined keys.

Attributes:

| Name                   | Type                                   | Description                                                                                                          |
| ---------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `latentspaces`         | `TrainingDynamics`                     | TrainingDynamics object storing latent space representations for 'train', 'valid', and 'test' splits.                |
| `sample_ids`           | `TrainingDynamics`                     | TrainingDynamics object storing sample identifiers for 'train', 'valid', and 'test' splits.                          |
| `reconstructions`      | `TrainingDynamics`                     | TrainingDynamics object storing reconstructed outputs for 'train', 'valid', and 'test' splits.                       |
| `mus`                  | `TrainingDynamics`                     | TrainingDynamics object storing mean values of latent distributions for 'train', 'valid', and 'test' splits.         |
| `sigmas`               | `TrainingDynamics`                     | TrainingDynamics object storing standard deviations of latent distributions for 'train', 'valid', and 'test' splits. |
| `losses`               | `TrainingDynamics`                     | TrainingDynamics object storing the total loss for different epochs and splits ('train', 'valid', 'test').           |
| `sub_losses`           | `LossRegistry`                         | LossRegistry object (extendable) for all sublosses.                                                                  |
| `preprocessed_data`    | `Tensor`                               | torch.Tensor containing data after preprocessing.                                                                    |
| `model`                | `Union[Dict[str, Module], Module]`     | final trained torch.nn.Module model.                                                                                 |
| `model_checkpoints`    | `TrainingDynamics`                     | TrainingDynamics object storing model state at each checkpoint.                                                      |
| `datasets`             | `Optional[DatasetContainer]`           | Optional[DatasetContainer] containing train, valid, and test datasets.                                               |
| `new_datasets`         | `Optional[DatasetContainer]`           | Optional[DatasetContainer] containing new train, valid, and test datasets.                                           |
| `adata_latent`         | `Optional[AnnData]`                    | Optional[AnnData] containing latent representations as AnnData.                                                      |
| `final_reconstruction` | `Optional[Union[DataPackage, MuData]]` | Optional\[Union[DataPackage, MuData]\] containing final reconstruction results.                                      |
| `sub_results`          | `Optional[Dict[str, Any]]`             | Optional\[Dict[str, Any]\] containing sub-results for multi-task or multi-modal models.                              |
| `sub_reconstructions`  | `Optional[Dict[str, Any]]`             | Optional\[Dict[str, Any]\] containing sub-reconstructions for multi-task or multi-modal models.                      |
| `embedding_evaluation` | `DataFrame`                            | pd.DataFrame containing embedding evaluation results.                                                                |
| `class_means`          | `TrainingDynamics`                     | Optional\[Dict[str, Any]\] containing the class means of the last epoch.                                             |

### `__getitem__(key: str) -> Any`

Retrieve the value associated with a specific key.

Parameters:

| Name  | Type  | Description                            | Default    |
| ----- | ----- | -------------------------------------- | ---------- |
| `key` | `str` | The name of the attribute to retrieve. | *required* |

Returns: The value of the specified attribute. Raises: KeyError - If the key is not a valid attribute of the Results class.

### `__repr__() -> str`

Return the same representation as **str** for consistency.

### `__setitem__(key: str, value: Any) -> None`

Assign a value to a specific attribute.

Parameters:

| Name    | Type  | Description                           | Default    |
| ------- | ----- | ------------------------------------- | ---------- |
| `key`   | `str` | The name of the attribute to set.     | *required* |
| `value` | `Any` | The value to assign to the attribute. | *required* |

Raises: KeyError: If the key is not a valid attribute of the Results class.

### `__str__() -> str`

Provide a readable string representation of the Result object's public attributes.

Returns:

| Type  | Description                                                     |
| ----- | --------------------------------------------------------------- |
| `str` | Formatted string showing all public attributes and their values |

### `get_latent_df(epoch: int, split: str, modality: Optional[str] = None) -> pd.DataFrame`

Return latent representations as a DataFrame.

Retrieves latent vectors and their corresponding sample IDs for a given epoch and data split. If a specific modality is provided, the results are restricted to that modality. Column names are inferred from model ontologies if available; otherwise, generic latent dimension labels are used.

Parameters:

| Name       | Type            | Description                                                  | Default    |
| ---------- | --------------- | ------------------------------------------------------------ | ---------- |
| `epoch`    | `int`           | The epoch number to retrieve latents from.                   | *required* |
| `split`    | `str`           | The dataset split to query (e.g., "train", "valid", "test"). | *required* |
| `modality` | `Optional[str]` | Optional modality name to filter the latents and sample IDs. | `None`     |

Returns:

| Type        | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `DataFrame` | A DataFrame where rows correspond to samples, columns represent latent |
| `DataFrame` | dimensions, and the index contains sample IDs.                         |

### `get_reconstructions_df(epoch: int, split: str, modality: Optional[str] = None) -> pd.DataFrame`

Return reconstructions as a DataFrame.

Retrieves reconstructed features and their corresponding sample IDs for a given epoch and data split. If a specific modality is provided, the results are restricted to that modality. Column names are based on the dataset's feature identifiers.

Parameters:

| Name       | Type            | Description                                                          | Default    |
| ---------- | --------------- | -------------------------------------------------------------------- | ---------- |
| `epoch`    | `int`           | The epoch number to retrieve reconstructions from.                   | *required* |
| `split`    | `str`           | The dataset split to query (e.g., "train", "valid", "test").         | *required* |
| `modality` | `Optional[str]` | Optional modality name to filter the reconstructions and sample IDs. | `None`     |

Returns:

| Type        | Description                                                     |
| ----------- | --------------------------------------------------------------- |
| `DataFrame` | A DataFrame where rows correspond to samples, columns represent |
| `DataFrame` | reconstructed features, and the index contains sample IDs.      |

### `update(other: Result) -> None`

Update the current Result object with values from another Result object.

For TrainingDynamics, merges the data across epochs and splits and overwrites if already exists. For all other attributes, replaces the current value with the other value.

Parameters:

| Name    | Type     | Description                       | Default    |
| ------- | -------- | --------------------------------- | ---------- |
| `other` | `Result` | The Result object to update from. | *required* |

Raises: TypeError: If the input object is not a Result instance

## `SingleCellDataReader`

Reader for multi-modal single-cell data.

### `read_data(config: DefaultConfig) -> Dict[str, MuData]`

Read multiple single-cell modalities into MuData object(s).

Args: config: Configuration object containing data paths and parameters.

Returns:

| Type                | Description                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, MuData]` | For non-paired translation: Dict of Dicts with {'multi_sc': DataDict} as outer dict and with modalty keys and mudata obj as inner dict. |
| `Dict[str, MuData]` | For paired translation and non translation cases: dict with "multi_sc" as key and mudata as value                                       |

## `VolumeDataReader`

Reads and processes 3D image data.

Reads all 3D images (volumes) from the specified directory, processes them, and returns a list of ImgData objects.

### `parse_image_to_array(image_path: Union[str, Path]) -> np.ndarray`

Reads a 3D image from the given path, optionally crops or pads it, and converts it to a numpy array.

Parameters:

| Name         | Type               | Description                 | Default    |
| ------------ | ------------------ | --------------------------- | ---------- |
| `image_path` | `Union[str, Path]` | The path to the image file. | *required* |

Returns:

| Type      | Description                      |
| --------- | -------------------------------- |
| `ndarray` | The processed image as a tensor. |

Raises:

| Type                   | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `FileNotFoundError`    | If the image path is invalid or the image cannot be read.                           |
| `ImageProcessingError` | If the image format is unsupported or an unexpected error occurs during processing. |

### `read_all_images_from_dir(img_dir: str, annotation_df: pd.DataFrame, is_paired: Union[bool, None] = None) -> List[ImgData]`

Reads all images from a specified directory, processes them, returns list of ImgData objects.

Parameters:

| Name            | Type                | Description                                     | Default    |
| --------------- | ------------------- | ----------------------------------------------- | ---------- |
| `img_dir`       | `str`               | The directory containing the images.            | *required* |
| `annotation_df` | `DataFrame`         | DataFrame containing image annotations.         | *required* |
| `is_paired`     | `Union[bool, None]` | Whether the images are paired with annotations. | `None`     |

Returns:

| Type            | Description                           |
| --------------- | ------------------------------------- |
| `List[ImgData]` | List of processed image data objects. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the annotation DataFrame is missing required columns. |

### `read_annotation_file(data_info: DataInfo) -> pd.DataFrame`

Reads annotation file and returns DataFrame with file contents Args: data_info: specific part of the Configuration object for input data Returns: DataFrame with annotation data.

### `read_data(config: Imagix3DConfig) -> Tuple[Dict[str, List[ImgData]], Dict[str, pd.DataFrame]]`

Read 3D image data from the specified directory based on configuration.

Parameters:

| Name     | Type             | Description                                                 | Default    |
| -------- | ---------------- | ----------------------------------------------------------- | ---------- |
| `config` | `Imagix3DConfig` | The configuration object containing the data configuration. | *required* |

Returns:

| Type                                                    | Description                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `Dict[str, List[ImgData]]`                              | A Tuple of Dicts:                                                              |
| `Dict[str, DataFrame]`                                  | Dict with type of image data as key and actual List of ImgData as value.       |
| `Tuple[Dict[str, List[ImgData]], Dict[str, DataFrame]]` | Dict with type of image data as key and DataFrame of annotation data as value. |

Raises:

| Type        | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| `Exception` | If no image data is found in the configuration or other validation errors occur. |

### `validate_image_path(image_path: Union[str, Path]) -> bool`

Checks if file extension is allowed:

Allowed are NIfTI files (independent of capitalization):

- nii
- nii.gz

Parameters:

| Name         | Type               | Description                  | Default    |
| ------------ | ------------------ | ---------------------------- | ---------- |
| `image_path` | `Union[str, Path]` | path or str of image to read | *required* |
