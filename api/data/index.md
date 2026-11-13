# Data Module

## `BalancedBatchSampler`

Bases: `Sampler[List[int]]`

A custom PyTorch Sampler that avoids creating a final batch of size 1.

This sampler behaves like a standard `BatchSampler` but with a key difference in handling the last batch. If the last batch would normally have a size of 1, this sampler redistributes the last two batches to be of roughly equal size. For example, if a dataset of 129 samples is used with a batch size of 128, instead of yielding batches of [128, 1], it will yield two balanced batches, such as [65, 64].

This is particularly useful for avoiding issues with layers like BatchNorm, which require batch sizes greater than 1, without having to drop data (`drop_last=True`).

Parameters:

| Name          | Type    | Description                                                           | Default    |
| ------------- | ------- | --------------------------------------------------------------------- | ---------- |
| `data_source` | `Sized` | The dataset to sample from.                                           | *required* |
| `batch_size`  | `int`   | The target number of samples in each batch.                           | *required* |
| `shuffle`     | `bool`  | If True, the sampler will shuffle the indices at start of each epoch. | `True`     |

### `__init__(data_source: Sized, batch_size: int, shuffle: bool = True)`

Initializes the BalancedBatchSampler. Args: data_source: The dataset to sample from. batch_size: The target number of samples in each batch. shuffle: If True, the sampler will shuffle the indices at start of each epoch.

### `__iter__() -> Iterator[List[int]]`

Returns an iterator over batches of indices.

### `__len__() -> int`

Returns the total number of batches in an epoch.

## `DataFilter`

Preprocesses dataframes, including filtering and scaling.

This class separates the filtering logic that needs to be applied consistently across train, validation, and test sets from the scaling logic that is typically fitted on the training data and then applied to the other sets.

Attributes:

| Name                | Type                 | Description                                                                   |
| ------------------- | -------------------- | ----------------------------------------------------------------------------- |
| `data_info`         |                      | Configuration object containing preprocessing parameters.                     |
| `filtered_features` | `Optional[Set[str]]` | Set of features to keep after filtering on the training data. None initially. |
| `_scaler`           |                      | The fitted scaler object. None initially.                                     |
| `ontologies`        |                      | Ontology information, if provided for Ontix.                                  |
| `config`            |                      | Configuration object containing default parameters.                           |

### `available_methods: List[str]`

Lists all available filtering methods.

Returns:

| Type        | Description                               |
| ----------- | ----------------------------------------- |
| `List[str]` | List of available filtering method names. |

### `__init__(data_info: DataInfo, config: DefaultConfig, ontologies: Optional[tuple] = None)`

Initializes the DataFilter with a configuration.

Parameters:

| Name         | Type              | Description                                               | Default    |
| ------------ | ----------------- | --------------------------------------------------------- | ---------- |
| `data_info`  | `DataInfo`        | Configuration object containing preprocessing parameters. | *required* |
| `config`     | `DefaultConfig`   | Configuration object containing default parameters.       | *required* |
| `ontologies` | `Optional[tuple]` | Ontology information, if provided for Ontix.              | `None`     |

### `filter(df: pd.DataFrame, genes_to_keep: Optional[List] = None) -> Tuple[Union[pd.Series, pd.DataFrame], List[str]]`

Applies the configured filtering method to the dataframe.

This method is intended to be called on the training data to determine which features to keep. The `filtered_features` attribute will be set based on the result.

Parameters:

| Name            | Type             | Description                                                                                    | Default    |
| --------------- | ---------------- | ---------------------------------------------------------------------------------------------- | ---------- |
| `df`            | `DataFrame`      | Input dataframe to be filtered (typically the training set).                                   | *required* |
| `genes_to_keep` | `Optional[List]` | A list of gene names to explicitly keep. If provided, other filtering methods will be ignored. | `None`     |

Returns:

| Type                                         | Description                                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `Tuple[Union[Series, DataFrame], List[str]]` | A tuple containing: - The filtered dataframe. - A list of column names (features) that were kept. |

Raises:

| Type       | Description                                                      |
| ---------- | ---------------------------------------------------------------- |
| `KeyError` | If some genes in genes_to_keep are not present in the dataframe. |

### `fit_scaler(df: Union[pd.Series, pd.DataFrame]) -> Any`

Fits the scaler to the input dataframe (typically the training set).

Parameters:

| Name | Type                       | Description                           | Default    |
| ---- | -------------------------- | ------------------------------------- | ---------- |
| `df` | `Union[Series, DataFrame]` | Input dataframe to fit the scaler on. | *required* |

Returns:

| Type  | Description               |
| ----- | ------------------------- |
| `Any` | The fitted scaler object. |

### `scale(df: Union[pd.Series, pd.DataFrame], scaler: Any) -> Union[pd.Series, pd.DataFrame]`

Applies the fitted scaler to the input dataframe.

Parameters:

| Name     | Type                       | Description                   | Default    |
| -------- | -------------------------- | ----------------------------- | ---------- |
| `df`     | `Union[Series, DataFrame]` | Input dataframe to be scaled. | *required* |
| `scaler` | `Any`                      | The fitted scaler object.     | *required* |

Returns:

| Type                       | Description       |
| -------------------------- | ----------------- |
| `Union[Series, DataFrame]` | Scaled dataframe. |

## `DataPackage`

Container for all input data of a pipeline, grouped by data type.

Every attribute is a dict that maps a modality name (your choice, e.g. "rna") to the data of that modality. One or more modalities per attribute are allowed. Samples are matched across modalities by their IDs (DataFrame index, AnnData `obs_names` or `ImgData.sample_id`).

Example

> > > DataPackage( ... multi_bulk={"rna": rna_df, "protein": protein_df}, ... annotation={"paired": clinical_df}, ... )

Attributes:

| Name            | Type                                                                    | Description                                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `multi_sc`      | `Optional[Dict[str, MuData]]`                                           | Single-cell data as MuData objects (one AnnData per modality inside).                                                                                           |
| `multi_bulk`    | `Optional[Dict[str, DataFrame]]`                                        | Tabular data (samples x features) as DataFrames; use this for any non-single-cell numeric data.                                                                 |
| `annotation`    | `Optional[Dict[str, Union[DataFrame, None]]]`                           | Sample metadata as DataFrames. Use the key "paired" for one table shared by all modalities, or the same keys as the data modalities for one table per modality. |
| `img`           | `Optional[Dict[str, List[ImgData]]]`                                    | Images as lists of ImgData.                                                                                                                                     |
| `from_modality` | `Optional[Dict[str, Union[DataFrame, List[ImgData], MuData, AnnData]]]` | Source modality for cross-modal translation; filled automatically for translation data cases.                                                                   |
| `to_modality`   | `Optional[Dict[str, Union[DataFrame, List[ImgData], MuData, AnnData]]]` | Target modality for cross-modal translation; filled automatically for translation data cases.                                                                   |

(LLM-generated docstring, pending human review.)

### `__getitem__(key: str) -> Any`

Allow dictionary-like access to top-level attributes.

### `__iter__() -> Iterator[Tuple[str, Any]]`

Make DataPackage iterable, yielding (key, value) pairs.

For dictionary attributes, yields nested items as (parent_key.child_key, value).

### `__setitem__(key: str, value: Any) -> None`

Allow dictionary-like item assignment to top-level attributes.

### `format_shapes() -> str`

Format the shape dictionary in a clean, readable way.

### `get_common_ids() -> List[str]`

Get the common sample IDs across modalities that have data.

Returns:

| Type        | Description                                                     |
| ----------- | --------------------------------------------------------------- |
| `List[str]` | List of sample IDs that are present in all modalities with data |

### `get_modality_key(direction: str) -> Optional[str]`

Get the first key for a specific direction's modality.

Parameters:

| Name        | Type  | Description           | Default    |
| ----------- | ----- | --------------------- | ---------- |
| `direction` | `str` | Either 'from' or 'to' | *required* |

Returns:

| Type            | Description                                           |
| --------------- | ----------------------------------------------------- |
| `Optional[str]` | First key of the modality dictionary or None if empty |

### `get_n_samples() -> Dict[str, Dict[str, int]]`

Get the number of samples for each data type in nested dictionary format.

Returns:

| Type                        | Description                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `Dict[str, Dict[str, int]]` | Dictionary with nested structure: {modality_type: {sub_key: count}} |

### `is_empty() -> bool`

Check if the data package is empty.

### `shape() -> Dict[str, Dict[str, Any]]`

Get the shape of the data for each data type in nested dictionary format.

Returns:

| Type                        | Description                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `Dict[str, Dict[str, Any]]` | Dictionary with nested structure: {modality_type: {sub_key: shape}} |

## `DataPackageSplitter`

Splits DataPackage objects into training, validation, and testing sets.

Supports paired and unpaired (translation) splitting.

Attributes:

| Name           | Type | Description                                           |
| -------------- | ---- | ----------------------------------------------------- |
| `data_package` |      | The original DataPackage to split.                    |
| `config`       |      | The configuration settings for the splitting process. |
| `indices`      |      | The indices for each split (train/val/test).          |

### `__init__(data_package: DataPackage, config: DefaultConfig, indices: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> None`

Store the DataPackage, config and precomputed split indices.

Parameters:

| Name           | Type                                       | Description                                                                                      | Default    |
| -------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------ | ---------- |
| `data_package` | `DataPackage`                              | The DataPackage to split.                                                                        | *required* |
| `config`       | `DefaultConfig`                            | Pipeline configuration.                                                                          | *required* |
| `indices`      | `Dict[str, Dict[str, Dict[str, ndarray]]]` | Sample indices per data attribute, modality and split, i.e. indices[attribute][modality][split]. | *required* |

Raises:

| Type        | Description                           |
| ----------- | ------------------------------------- |
| `TypeError` | If data_package is not a DataPackage. |

(LLM-generated docstring, pending human review.)

### `split() -> Dict[str, Optional[Dict[str, Any]]]`

Splits the underlying DataPackage into train, valid, and test subsets. Returns: A dictionary containing the split data packages for "train", "valid", and "test". Each entry contains a "data" key with the DataPackage and an "indices" key with the corresponding indices. Raises: ValueError: If no data package is available for splitting. TypeError: If indices are not provided for unpaired translation case.

## `DataSplitter`

Splits data into train, validation, and test sets. And validates the splits.

Also allows for custom splits to be provided. Here we allow empty splits (e.g. test_ratio=0), this might raise an error later in the pipeline, when this split is expected to be non-empty. However, this allows are more flexible usage of the pipeline (e.g. when the user only wants to run the fit step).

Constraints:

1. Split ratios must sum to 1
1. Each non-empty split must have at least min_samples_per_split samples
1. Any split ratio must be \<= 1.0
1. Custom splits must contain 'train', 'valid', and 'test' keys and non-overlapping indices

Attributes:

| Name             | Type | Description                                  |
| ---------------- | ---- | -------------------------------------------- |
| `_config`        |      | Configuration object containing split ratios |
| `_custom_splits` |      | Optional pre-defined split indices           |
| `_test_ratio`    |      | Fraction of samples in the test split.       |
| `_valid_ratio`   |      | Fraction of samples in the validation split. |

### `__init__(config: DefaultConfig, custom_splits: Optional[Dict[str, np.ndarray]] = None)`

Initialize DataSplitter with configuration and optional custom splits.

Parameters:

| Name            | Type                           | Description                                  | Default    |
| --------------- | ------------------------------ | -------------------------------------------- | ---------- |
| `config`        | `DefaultConfig`                | Configuration object containing split ratios | *required* |
| `custom_splits` | `Optional[Dict[str, ndarray]]` | Pre-defined split indices                    | `None`     |

### `split(n_samples: int) -> Dict[str, np.ndarray]`

Split data into train, validation, and test sets.

Parameters:

| Name        | Type  | Description                            | Default    |
| ----------- | ----- | -------------------------------------- | ---------- |
| `n_samples` | `int` | Total number of samples in the dataset | *required* |

Returns:

| Type                 | Description                                                                             |
| -------------------- | --------------------------------------------------------------------------------------- |
| `Dict[str, ndarray]` | Dictionary containing indices for each split, with empty arrays for splits with ratio=0 |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | If resulting splits would violate size constraints |

## `DatasetContainer`

A container for datasets used in training, validation, and testing.

train : Dataset The training dataset. valid : Dataset The validation dataset. test : Dataset The testing dataset.

### `__getitem__(key: str) -> BaseDataset`

Allows dictionary-like access to datasets.

### `__setitem__(key: str, value: BaseDataset)`

Allows dictionary-like assignment of datasets.

## `GeneralPreprocessor`

Bases: `BasePreprocessor`

Preprocessor for handling multi-modal data.

Attributes:

| Name                          | Type                                                | Description                                                 |
| ----------------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| `_datapackage_dict`           | `Optional[Dict[str, Any]]`                          | Dictionary holding DataPackage objects for each data split. |
| `_dataset_container`          | `Optional[DatasetContainer]`                        | Container holding processed datasets for each split.        |
| `_reverse_mapping_multi_bulk` | `Dict[str, Dict[str, Tuple[List[int], List[str]]]]` | Reverse mapping for multi-bulk data reconstruction.         |
| `_reverse_mapping_multi_sc`   | `Dict[str, Dict[str, Tuple[List[int], List[str]]]]` | Reverse mapping for multi-single-cell data reconstruction.  |

### `__init__(config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None) -> None`

Initialize the preprocessor and empty reverse mappings per split.

Parameters:

| Name         | Type                           | Description                                               | Default    |
| ------------ | ------------------------------ | --------------------------------------------------------- | ---------- |
| `config`     | `DefaultConfig`                | Pipeline configuration.                                   | *required* |
| `ontologies` | `Optional[Union[Tuple, Dict]]` | Ontology definitions (Ontix), used for feature filtering. | `None`     |

(LLM-generated docstring, pending human review.)

### `format_reconstruction(reconstruction: torch.Tensor, result: Optional[Result] = None) -> DataPackage`

Split a combined reconstruction back into the original modalities.

Parameters:

| Name             | Type               | Description                                       | Default    |
| ---------------- | ------------------ | ------------------------------------------------- | ---------- |
| `reconstruction` | `Tensor`           | Reconstructed data (samples x combined features). | *required* |
| `result`         | `Optional[Result]` | Unused; kept for interface compatibility.         | `None`     |

Returns:

| Type          | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `DataPackage` | DataPackage with DataFrames (multi-bulk) or a MuData (single-cell) |
| `DataPackage` | per modality, plus the sample metadata.                            |

Raises:

| Type                  | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `NotImplementedError` | For data cases other than multi-bulk and multi-single-cell. |

(LLM-generated docstring, pending human review.)

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data: bool = False) -> DatasetContainer`

Run the full preprocessing and return one dataset per split.

Reads, filters, scales and splits the data (see `BasePreprocessor._general_preprocess`) and combines all modalities of a split into one `NumericDataset`.

Parameters:

| Name               | Type                    | Description                                                                                                | Default |
| ------------------ | ----------------------- | ---------------------------------------------------------------------------------------------------------- | ------- |
| `raw_user_data`    | `Optional[DataPackage]` | Input data. If None, data is read as defined in the config.                                                | `None`  |
| `predict_new_data` | `bool`                  | If True, treat all data as test data and reuse the fitted scalers and feature selection (used by predict). | `False` |

Returns:

| Type               | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `DatasetContainer` | Container with "train", "valid" and "test" datasets (None for |
| `DatasetContainer` | empty splits).                                                |

Raises:

| Type        | Description                        |
| ----------- | ---------------------------------- |
| `TypeError` | If preprocessing produced no data. |

(LLM-generated docstring, pending human review.)

## `GlobalVolumeNormalizer`

Normalizes (scales) 3D images using stats fitted on the training set.

This class implements train-global normalization for volumetric image data. Stats are computed once across all training images and then reused for transforming training, validation, and test data.

Supported normalization methods are

- STANDARD: subtract channel-wise mean and divide by channel-wise standard deviation
- MINMAX: subtract channel-wise minimum and divide by channel-wise range
- ROBUST: subtract channel-wise median and divide by channel-wise interquartile range
- NONE: no normalization

Attributes:

| Name                     | Type                                          | Description                                                          |
| ------------------------ | --------------------------------------------- | -------------------------------------------------------------------- |
| `method`                 | `ScalingMethod`                               | The normalization method to apply.                                   |
| `normalize_nonzero_only` | `bool`                                        | If True, compute statistics only on nonzero voxels                   |
| `stats`                  | `Dict[str, ndarray]`                          | Dictionary containing fitted channel-wise normalization statistics.  |
| `The`                    | `exact entries depend on the selected method` | MINMAX: "min", "max" STANDARD: "mean", "std" ROBUST: "median", "iqr" |

### `fit(images: list, method: ScalingMethod, normalize_nonzero_only: bool) -> GlobalVolumeNormalizer`

Fits a GlobalVolumeNormalizer on a list of training images.

This class method computes normalization statistics across all training images and returns a fitted normalizer object.

Parameters:

| Name                     | Type            | Description                                     | Default    |
| ------------------------ | --------------- | ----------------------------------------------- | ---------- |
| `images`                 | `list`          | List of image data objects.                     | *required* |
| `method`                 | `ScalingMethod` | The normalization method to fit.                | *required* |
| `normalize_nonzero_only` | `bool`          | If True, fit statistics only on nonzero voxels. | *required* |

Returns:

| Type                     | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| `GlobalVolumeNormalizer` | A fitted GlobalVolumeNormalizer instance containing the selected method |
| `GlobalVolumeNormalizer` | and the corresponding channel-wise statistics.                          |

Raises:

| Type         | Description                                         |
| ------------ | --------------------------------------------------- |
| `ValueError` | If an unsupported normalization method is provided. |

### `transform_volume(image: np.ndarray) -> np.ndarray`

Applies the fitted normalization transform to a single 3D image volume. The transformation uses the statistics previously computed by `fit()`.

Parameters:

| Name    | Type      | Description                          | Default    |
| ------- | --------- | ------------------------------------ | ---------- |
| `image` | `ndarray` | Input image volume as a NumPy array. | *required* |

Returns:

| Type      | Description                                   |
| --------- | --------------------------------------------- |
| `ndarray` | The normalized image volume as a NumPy array. |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | If the stored normalization method is unsupported. |

## `ImageDataset`

Bases: `TensorAwareDataset`

A custom PyTorch dataset that handles image data with proper dtype conversion.

Attributes:

| Name            | Type | Description                                                             |
| --------------- | ---- | ----------------------------------------------------------------------- |
| `raw_data`      |      | List of ImgData objects containing original image data and metadata.    |
| `config`        |      | Configuration object for dataset settings.                              |
| `mytype`        |      | Enum indicating the dataset type (set to DataSetTypes.IMG).             |
| `data`          |      | List of image tensors converted to the appropriate dtype.               |
| `sample_ids`    |      | List of identifiers for each sample.                                    |
| `split_indices` |      | Optional numpy array of indices for splitting the dataset.              |
| `feature_ids`   |      | Optional list of identifiers for each feature (set to None for images). |
| `metadata`      |      | Optional pandas DataFrame containing additional metadata.               |

### `__getitem__(idx)`

Get item at index - data is already converted to proper dtype Returns: Tuple of (index, image tensor, sample_id)

### `__init__(data: List[ImgData], config: DefaultConfig, split_indices: Optional[Dict[str, Any]] = None, metadata: Optional[pd.DataFrame] = None)`

Initialize the dataset and convert all images to tensors once.

Parameters:

| Name            | Type                       | Description                                     | Default    |
| --------------- | -------------------------- | ----------------------------------------------- | ---------- |
| `data`          | `List[ImgData]`            | List of image data objects.                     | *required* |
| `config`        | `DefaultConfig`            | Configuration object; defines the tensor dtype. | *required* |
| `split_indices` | `Optional[Dict[str, Any]]` | Indices of the samples belonging to this split. | `None`     |
| `metadata`      | `Optional[DataFrame]`      | Sample metadata (annotation) for the images.    | `None`     |

Raises:

| Type         | Description        |
| ------------ | ------------------ |
| `ValueError` | If config is None. |

### `get_input_dim() -> Tuple[int, ...]`

Gets the input dimension of the dataset's feature space.

Returns:

| Type              | Description                                        |
| ----------------- | -------------------------------------------------- |
| `Tuple[int, ...]` | The input dimension of the dataset's feature space |

## `ImagePreprocessor`

Bases: `GeneralPreprocessor`

Preprocessor for a single image modality (`Imagix`).

Splits and normalizes the images of a DataPackage and wraps each split in an `ImageDataset`. Only one image modality is supported; additional ones are ignored with a warning.

Attributes:

| Name            | Type | Description                                                          |
| --------------- | ---- | -------------------------------------------------------------------- |
| `data_config`   |      | Configuration specific to data handling and preprocessing.           |
| `dataset_dicts` |      | Dictionary holding datasets for different splits (train/test/valid). |

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data: bool = False) -> DatasetContainer`

Preprocess the data according to the configuration.

Parameters:

| Name               | Type                    | Description                                     | Default |
| ------------------ | ----------------------- | ----------------------------------------------- | ------- |
| `raw_user_data`    | `Optional[DataPackage]` | The raw data package provided by the user.      | `None`  |
| `predict_new_data` | `bool`                  | Flag indicating if new data is being predicted. | `False` |

Returns: A DatasetContainer with processed training, validation, and test datasets.

## `ImgData`

Stores image data along with its associated metadata.

Attributes:

| Name         | Type                       | Description                                                          |
| ------------ | -------------------------- | -------------------------------------------------------------------- |
| `img`        |                            | The image data as a NumPy array.                                     |
| `sample_id`  | `str`                      | A unique identifier for the image sample.                            |
| `annotation` | `Union[Series, DataFrame]` | A DataFrame containing annotations or metadata related to the image. |

## `MultiModalDataset`

Bases: `BaseDataset`, `Dataset`

Handles multiple datasets of different modalities.

Attributes:

| Name                   | Type        | Description                                                   |
| ---------------------- | ----------- | ------------------------------------------------------------- |
| `datasets`             |             | Dictionary of datasets for each modality.                     |
| `n_modalities`         |             | Number of modalities.                                         |
| `sample_to_modalities` |             | Mapping from sample IDs to available modalities.              |
| `sample_ids`           | `List[Any]` | List of all unique sample IDs across modalities.              |
| `config`               |             | Configuration object.                                         |
| `data`                 |             | Data from the first modality (for compatibility).             |
| `feature_ids`          |             | Feature IDs (currently None, to be implemented).              |
| `_id_to_idx`           |             | Reverse lookup tables for sample IDs to indices per modality. |
| `paired_sample_ids`    |             | List of sample IDs that have data in all modalities.          |
| `unpaired_sample_ids`  |             | List of sample IDs that do not have data in all modalities.   |

### `is_fully_paired: bool`

Returns True if all samples are fully paired across all modalities (no unpaired samples).

### `__getitem__(idx: Union[int, str])`

Return the data of one sample across all modalities.

Parameters:

| Name  | Type              | Description                                                   | Default    |
| ----- | ----------------- | ------------------------------------------------------------- | ---------- |
| `idx` | `Union[int, str]` | Position in the paired sample IDs (int) or a sample ID (str). | *required* |

Returns:

| Name   | Type | Description                                             |
| ------ | ---- | ------------------------------------------------------- |
| `dict` |      | {"sample_id": id, <modality>: data or None if missing}. |

(LLM-generated docstring, pending human review.)

### `__init__(datasets: Dict[str, BaseDataset], config: DefaultConfig)`

Initialize the MultiModalDataset.

Parameters:

| Name       | Type                     | Description                               | Default    |
| ---------- | ------------------------ | ----------------------------------------- | ---------- |
| `datasets` | `Dict[str, BaseDataset]` | Dictionary of datasets for each modality. | *required* |
| `config`   | `DefaultConfig`          | Configuration object.                     | *required* |

## `NaNRemover`

Removes NaN values from multi-modal datasets.

This object identifies and removes NaN values from various data structures commonly used in single-cell and multi-modal omics, including AnnData, MuData, and Pandas DataFrames. It supports processing of X matrices, layers, and observation annotations within AnnData objects, as well as handling bulk and annotation data within a DataPackage.

Attributes:

| Name            | Type | Description                                                   |
| --------------- | ---- | ------------------------------------------------------------- |
| `config`        |      | Configuration object containing settings for data processing. |
| `relevant_cols` |      | List of columns in metadata to check for NaNs.                |

### `__init__(config: DefaultConfig)`

Initialize the NaNRemover with configuration settings. Args: config: Configuration object containing settings for data processing.

### `remove_nan(data: DataPackage) -> DataPackage`

Removes NaN values from all applicable DataPackage components.

Iterates through the bulk data, annotation data, and multi-modal single-cell data (MuData and AnnData objects) within the provided DataPackage and removes rows/columns/entries containing NaN values.

Parameters:

| Name   | Type          | Description                                         | Default    |
| ------ | ------------- | --------------------------------------------------- | ---------- |
| `data` | `DataPackage` | The DataPackage object containing multi-modal data. | *required* |

Returns:

| Type          | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| `DataPackage` | The DataPackage object with NaN values removed from its components. |

## `NumericDataset`

Bases: `TensorAwareDataset`

A custom PyTorch dataset that handles tensors.

Attributes:

| Name            | Type | Description                                                   |
| --------------- | ---- | ------------------------------------------------------------- |
| `data`          |      | The input features as a torch.Tensor.                         |
| `config`        |      | Configuration object containing settings for data processing. |
| `sample_ids`    |      | Optional list of sample identifiers.                          |
| `feature_ids`   |      | Optional list of feature identifiers.                         |
| `metadata`      |      | Optional pandas DataFrame containing metadata.                |
| `split_indices` |      | Optional numpy array for data splitting.                      |
| `mytype`        |      | Enum indicating the dataset type (set to DataSetTypes.NUM).   |

### `__getitem__(index: int) -> Union[Tuple[Union[torch.Tensor, int], Union[torch.Tensor, 'ImgData'], Any], Dict[str, Tuple[Any, torch.Tensor, Any]]]`

Retrieves a single sample and its corresponding label.

Parameters:

| Name    | Type  | Description                      | Default    |
| ------- | ----- | -------------------------------- | ---------- |
| `index` | `int` | Index of the sample to retrieve. | *required* |

Returns:

| Type                                                                                                  | Description                                                                        |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Union[Tuple[Union[Tensor, int], Union[Tensor, 'ImgData'], Any], Dict[str, Tuple[Any, Tensor, Any]]]` | A tuple containing the index, the data sample and its label, or a dictionary       |
| `Union[Tuple[Union[Tensor, int], Union[Tensor, 'ImgData'], Any], Dict[str, Tuple[Any, Tensor, Any]]]` | mapping keys to such tuples in case we have multiple uncombined data at this step. |

### `__init__(data: Union[torch.Tensor, np.ndarray, sp.sparse.spmatrix], config: DefaultConfig, sample_ids: Union[None, List[Any]] = None, feature_ids: Union[None, List[Any]] = None, metadata: Optional[Union[pd.Series, pd.DataFrame]] = None, split_indices: Optional[Union[Dict[str, Any], List[Any], np.ndarray]] = None)`

Initialize the dataset

Parameters:

| Name            | Type                                                  | Description                  | Default    |
| --------------- | ----------------------------------------------------- | ---------------------------- | ---------- |
| `data`          | `Union[Tensor, ndarray, spmatrix]`                    | Input features               | *required* |
| `config`        | `DefaultConfig`                                       | Configuration object         | *required* |
| `sample_ids`    | `Union[None, List[Any]]`                              | Optional sample identifiers  | `None`     |
| `feature_ids`   | `Union[None, List[Any]]`                              | Optional feature identifiers | `None`     |
| `metadata`      | `Optional[Union[Series, DataFrame]]`                  | Optional metadata            | `None`     |
| `split_indices` | `Optional[Union[Dict[str, Any], List[Any], ndarray]]` | Optional split indices       | `None`     |

### `__len__() -> int`

Returns the number of samples (rows) in the dataset

## `SingleCellFilter`

Filter and scale single-cell data, returning a MuData object with synchronized metadata.AnnData

Attributes:

| Name                 | Type | Description                                                                                            |
| -------------------- | ---- | ------------------------------------------------------------------------------------------------------ |
| `data_info`          |      | Configuration for filtering and scaling (can be a single DataInfo or a dict of DataInfo per modality). |
| `total_features`     |      | Total number of features to keep across all modalities.                                                |
| `config`             |      | Configuration object containing settings for data processing.                                          |
| `_is_data_info_dict` |      | Internal flag indicating if data_info is a dictionary.                                                 |

### `__init__(data_info: Union[Dict[str, DataInfo], DataInfo], config: DefaultConfig)`

Initialize single-cell filter. Args: data_info: Either a single data_info object for all modalities or a dictionary of data_info objects for each modality. config: Configuration object containing settings for data processing.

### `distribute_features_across_modalities(mudata: MuData, total_features: Optional[int]) -> Dict[str, int]`

Distributes a total number of features across modalities evenly.

Parameters:

| Name             | Type            | Description                                                  | Default    |
| ---------------- | --------------- | ------------------------------------------------------------ | ---------- |
| `mudata`         | `MuData`        | Multi-modal data object                                      | *required* |
| `total_features` | `Optional[int]` | Total number of features to distribute across all modalities | *required* |

Returns:

| Type             | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `Dict[str, int]` | Dictionary mapping modality keys to number of features to keep |

### `general_postsplit_processing(mudata: MuData, gene_map: Optional[Dict[str, List]], scaler_map: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[MuData, Dict[str, List], Dict[str, Dict[str, Any]]]`

Process single-cell data with proper MuData handling Args: mudata: Input multi-modal data container gene_map: Optional override of genes to keep per modality scaler_map: Optional pre-fitted scalers per modality and layer Returns: Processed MuData with filtered and scaled modalities,

### `presplit_processing(multi_sc: Union[MuData, Dict[str, MuData]]) -> Dict[str, MuData]`

Process each modality independently to filter cells based on min_genes.

Parameters:

| Name       | Type                               | Description                                                      | Default    |
| ---------- | ---------------------------------- | ---------------------------------------------------------------- | ---------- |
| `multi_sc` | `Union[MuData, Dict[str, MuData]]` | Either a single MuData object or a dictionary of MuData objects. | *required* |

Returns: A dictionary mapping modality keys to processed MuData objects.

### `sc_postsplit_processing(mudata: MuData, gene_map: Optional[Dict[str, List[str]]] = None) -> Tuple[MuData, Dict[str, List[str]]]`

Process each modality independently to filter genes based on X layer, then consistently apply the same filtering to all layers.

Args: mudata : Input multi-modal data container gene_map : Optional override of genes to keep per modality

Returns:

| Type                   | Description                               |
| ---------------------- | ----------------------------------------- |
| `MuData`               | Processed MuData with filtered modalities |
| `Dict[str, List[str]]` | Mapping of modality to kept gene names    |

## `StackixDataset`

Bases: `NumericDataset`

Dataset for handling multiple modalities in Stackix models.

This dataset holds individual BaseDataset objects for multiple data modalities and provides a consistent interface for accessing them during training. It's designed to work specifically with StackixTrainer.

Attributes:

| Name            | Type | Description                                          |
| --------------- | ---- | ---------------------------------------------------- |
| `dataset_dict`  |      | Dictionary mapping modality names to dataset objects |
| `modality_keys` |      | List of modality names                               |

### `__getitem__(index: int) -> Union[Tuple[torch.Tensor, Any], Dict[str, Tuple[torch.Tensor, Any]]]`

Get a single sample and its label from the dataset.

Returns the data from the first modality to maintain compatibility with the BaseDataset interface, while still supporting multi-modality access through dataset_dict. Args: index: Index of the sample to retrieve

Returns:

| Type                                                       | Description                                                |
| ---------------------------------------------------------- | ---------------------------------------------------------- |
| `Union[Tuple[Tensor, Any], Dict[str, Tuple[Tensor, Any]]]` | Dictionary of (data tensor, label) pairs for each modality |

### `__init__(dataset_dict: Dict[str, BaseDataset], config: DefaultConfig)`

Initialize a StackixDataset instance.

Parameters:

| Name           | Type                     | Description                                          | Default    |
| -------------- | ------------------------ | ---------------------------------------------------- | ---------- |
| `dataset_dict` | `Dict[str, BaseDataset]` | Dictionary mapping modality names to dataset objects | *required* |
| `config`       | `DefaultConfig`          | Configuration object                                 | *required* |

Raises:

| Type                  | Description                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `ValueError`          | If the datasets dictionary is empty or if modality datasets have different numbers of samples |
| `NotImplementedError` | If the datasets have incompatible shapes for concatenation                                    |

### `__len__() -> int`

Return the number of samples in the dataset.

### `get_modality_item(modality: str, index: int) -> Tuple[torch.Tensor, Any]`

Get a sample for a specific modality. Args: modality: The modality name to retrieve data from index: Index of the sample to retrieve

Returns:

| Type                 | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `Tuple[Tensor, Any]` | Tuple of (data tensor, label) for the specified modality and sample index |

Raises:

| Type       | Description                                            |
| ---------- | ------------------------------------------------------ |
| `KeyError` | If the requested modality doesn't exist in the dataset |

## `StackixPreprocessor`

Bases: `BasePreprocessor`

Preprocessor for Stackix architecture, which handles multiple modalities separately.

Unlike GeneralPreprocessor which combines all modalities, StackixPreprocessor keeps modalities separate for individual VAE training in the Stackix architecture.

Attributes: config: Configuration parameters for preprocessing and model architecture \_datapackage: Dictionary storing processed data splits \_dataset_container:Container for processed datasets by split

### `__init__(config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None) -> None`

Initialize the StackixPreprocessor with the given configuration.

Parameters:

| Name         | Type                           | Description                                                | Default    |
| ------------ | ------------------------------ | ---------------------------------------------------------- | ---------- |
| `config`     | `DefaultConfig`                | Configuration parameters for preprocessing.                | *required* |
| `ontologies` | `Optional[Union[Tuple, Dict]]` | Accepted for interface compatibility; not used by Stackix. | `None`     |

### `format_reconstruction(reconstruction: Any, result: Optional[Result] = None) -> DataPackage`

Takes the reconstructed tensor and from which modality it comes and uses the dataset_dict to obtain the format of the original datapackage, but instead of the .data attribute we populate this attribute with the reconstructed tensor (as pd.DataFrame or MuData object)

Parameters:

| Name             | Type               | Description                                        | Default    |
| ---------------- | ------------------ | -------------------------------------------------- | ---------- |
| `reconstruction` | `Any`              | The reconstructed tensor                           | *required* |
| `result`         | `Optional[Result]` | Optional[Result] containing additional information | `None`     |

Returns: DataPackage with reconstructed data in original format

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data=False) -> DatasetContainer`

Execute preprocessing steps for Stackix architecture.

Parameters:

| Name               | Type                    | Description                                                                                | Default |
| ------------------ | ----------------------- | ------------------------------------------------------------------------------------------ | ------- |
| `raw_user_data`    | `Optional[DataPackage]` | Raw user data to preprocess, or None to read the data defined in the config.               | `None`  |
| `predict_new_data` |                         | If True, treat all data as test data and reuse the fitted preprocessing (used by predict). | `False` |

Returns:

| Type               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `DatasetContainer` | Container with MultiModalDataset for each split. |

Raises:

| Type        | Description                                 |
| ----------- | ------------------------------------------- |
| `TypeError` | If datapackage is None after preprocessing. |

## `TensorAwareDataset`

Bases: `BaseDataset`

Handles dtype mapping and tensor conversion logic.

## `VolumePreprocessor`

Bases: `ImagePreprocessor`

Preprocessor for 3D image data

Attributes:

| Name            | Type              | Description                                                                                                                 |
| --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `data_config`   |                   | Configuration specific to data handling and preprocessing.                                                                  |
| `config`        | `Imagix3DConfig`  | Complete 3D image-specific configuration                                                                                    |
| `dataset_dicts` | `Imagix3DConfig`  | Dictionary holding datasets for different splits (train/test/valid).                                                        |
| `data_readers`  | `Dict[Enum, Any]` | A dictionary mapping DataCase enum values to data reader instances for different modalities. Overwritten here for 3D images |

### `__init__(config: Imagix3DConfig, ontologies: Optional[Union[Tuple, Dict]] = None)`

Initialize the preprocessor and register 3D volume readers.

Parameters:

| Name         | Type                           | Description                                               | Default    |
| ------------ | ------------------------------ | --------------------------------------------------------- | ---------- |
| `config`     | `Imagix3DConfig`               | Imagix3D configuration (scaling strategy, target shapes). | *required* |
| `ontologies` | `Optional[Union[Tuple, Dict]]` | Accepted for interface compatibility; not used.           | `None`     |

(LLM-generated docstring, pending human review.)

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data: bool = False) -> DatasetContainer`

Preprocess the data according to the configuration.

Parameters:

| Name               | Type                    | Description                                     | Default |
| ------------------ | ----------------------- | ----------------------------------------------- | ------- |
| `raw_user_data`    | `Optional[DataPackage]` | The raw data package provided by the user.      | `None`  |
| `predict_new_data` | `bool`                  | Flag indicating if new data is being predicted. | `False` |

Returns: A DatasetContainer with processed training, validation, and test datasets.

## `XModalPreprocessor`

Bases: `GeneralPreprocessor`

Preprocessor for cross-modal data, handling multiple data types and their transformations.

Attributes:

| Name            | Type | Description                                                            |
| --------------- | ---- | ---------------------------------------------------------------------- |
| `data_config`   |      | Configuration specific to data handling.                               |
| `dataset_dicts` |      | Dictionary holding datasets for different splits (train, test, valid). |

### `__init__(config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None)`

Initializes the XModalPreprocessor Args: config: Configuration object for the preprocessor. ontologies: Optional ontologies for data processing.

### `format_reconstruction(reconstruction, result=None)`

Not used for XModalix; reconstructions stay in tensor form.

(LLM-generated docstring, pending human review.)

### `preprocess(raw_user_data: Optional[DataPackage] = None, predict_new_data: bool = False) -> DatasetContainer`

Preprocess the data according to the configuration. Args: raw_user_data: Optional raw data provided by the user. predict_new_data: Flag indicating if new data is being predicted.
