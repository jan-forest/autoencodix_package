# Configs Module

## `DataCase`

Bases: `str`, `Enum`

Supported combinations of input data, set automatically from the data.

`MULTI_BULK` and `MULTI_SINGLE_CELL` cover one or more tabular or single-cell modalities that are integrated into one latent space. The `X_TO_Y` cases are cross-modal translation setups (`XModalix`, `Imagix`), e.g. `IMG_TO_BULK` translates between images and tabular data.

(LLM-generated docstring, pending human review.)

## `DataConfig`

Bases: `BaseModel`, `SchemaPrinterMixin`

Data settings for all modalities of a pipeline run.

Usually filled automatically from the `DataPackage`; set it explicitly to customize per-modality preprocessing.

Attributes:

| Name                   | Type                  | Description                                                                                                                       |
| ---------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `data_info`            | `Dict[str, DataInfo]` | Settings per modality, keyed by modality name.                                                                                    |
| `require_common_cells` | `Optional[bool]`      | For single-cell data, keep only cells present in all modalities.                                                                  |
| `annotation_columns`   | `Optional[List[str]]` | Deprecated; set annotation_columns in the main config (DefaultConfig) instead. Values given here are copied there with a warning. |

(LLM-generated docstring, pending human review.)

## `DataInfo`

Bases: `BaseModel`, `SchemaPrinterMixin`

Per-modality data settings, stored in `DataConfig.data_info[<modality>]`.

Controls how one data modality is read, filtered, scaled and (for images) resized. Settings here override the corresponding global `DefaultConfig` values for this modality. Call `DataInfo.print_schema()` for all fields with defaults and descriptions.

Attributes:

| Name                  | Type                                                               | Description                                                                                                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_type`           | `Literal['NUMERIC', 'CATEGORICAL', 'IMG', 'ANNOTATION']`           | Kind of data: "NUMERIC" (features to model), "CATEGORICAL", "IMG" (images) or "ANNOTATION" (sample metadata, not modeled).                                                                                                                      |
| `filtering`           | `Literal['VAR', 'MAD', 'CORR', 'VARCORR', 'NOFILT', 'NONZEROVAR']` | Feature selection method applied before training: "VAR" (highest variance), "MAD" (median absolute deviation), "CORR" (correlation-based), "VARCORR" (variance, then correlation), "NONZEROVAR" (drop zero-variance features only) or "NOFILT". |
| `sep`                 | `Union[str, None]`                                                 | Column separator used when reading text files with pandas.                                                                                                                                                                                      |
| `extra_anno_file`     | `Union[str, None]`                                                 | Optional path to an additional annotation file.                                                                                                                                                                                                 |
| `is_single_cell`      | `bool`                                                             | Whether this modality is single-cell data.                                                                                                                                                                                                      |
| `selected_layers`     | `List[str]`                                                        | AnnData layers to use; must include "X".                                                                                                                                                                                                        |
| `img_width_resize`    | `Union[int, None]`                                                 | Target image width in pixels (must equal the height).                                                                                                                                                                                           |
| `img_height_resize`   | `Union[int, None]`                                                 | Target image height in pixels (must equal the width).                                                                                                                                                                                           |
| `translate_direction` | `Union[Literal['from', 'to'], None]`                               | "from" or "to" for the source and target modality in cross-modal translation; None otherwise.                                                                                                                                                   |

(LLM-generated docstring, pending human review.)

### `validate_image_resize(v, info: ValidationInfo)`

Ensure image resize dimensions are positive and square.

Raises:

| Type         | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `ValueError` | If a dimension is not positive or width and height differ. |

(LLM-generated docstring, pending human review.)

### `validate_selected_layers(v)`

Ensure that the AnnData layer "X" is always among the selected layers.

Raises:

| Type         | Description        |
| ------------ | ------------------ |
| `ValueError` | If "X" is missing. |

(LLM-generated docstring, pending human review.)

## `DefaultConfig`

Bases: `BaseModel`, `SchemaPrinterMixin`

Complete configuration for model, training, hardware, and data handling.

### `determine_case() -> DefaultConfig`

Assign the correct DataCase after model validation.

### `get_params() -> Dict[str, Dict[str, Any]]`

Get detailed information about all config fields including types and default values.

Returns:

| Type                        | Description                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `Dict[str, Dict[str, Any]]` | Dictionary containing field name, type, default value, and description if available |

### `handle_backward_compatibility() -> DefaultConfig`

Handle migration of annotation_columns from DataConfig to DefaultConfig.

### `print_schema(filter_params: Optional[None] = None) -> None`

Print a human-readable schema of all config parameters.

### `validate_data_config(data_config: DataConfig)`

Main validation logic for dataset consistency and translation.

Checks that at least one NUMERIC or IMG dataset exists, that numeric datasets are either all single-cell or all bulk, and that translation directions ("from"/"to") are set consistently.

Parameters:

| Name          | Type         | Description                         | Default    |
| ------------- | ------------ | ----------------------------------- | ---------- |
| `data_config` | `DataConfig` | The data configuration to validate. | *required* |

Returns:

| Type | Description                       |
| ---- | --------------------------------- |
|      | The validated data configuration. |

Raises:

| Type                    | Description                                 |
| ----------------------- | ------------------------------------------- |
| `ConfigValidationError` | If the dataset combination is inconsistent. |

### `validate_float_precision(v, values)`

Validate float precision based on device type.

Raises:

| Type         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | If the device is "mps" and the precision is not "32". |

### `validate_gpu_strategy(v, values)`

Ensure that the MPS backend is only used with the "auto" GPU strategy.

Raises:

| Type         | Description                                            |
| ------------ | ------------------------------------------------------ |
| `ValueError` | If the device is "mps" and the strategy is not "auto". |

(LLM-generated docstring, pending human review.)

### `validate_k_filter_with_nonzero_var()`

Forbid combining a global `k_filter` with "NONZEROVAR" filtering.

Returns:

| Type | Description           |
| ---- | --------------------- |
|      | The validated config. |

Raises:

| Type         | Description                                            |
| ------------ | ------------------------------------------------------ |
| `ValueError` | If k_filter is set and any modality uses "NONZEROVAR". |

(LLM-generated docstring, pending human review.)

### `validate_ratios(v, values)`

Ensure that train, valid and test ratios sum to at most 1.

Raises:

| Type         | Description                       |
| ------------ | --------------------------------- |
| `ValueError` | If the ratios sum to more than 1. |

(LLM-generated docstring, pending human review.)

## `DisentanglixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig.

## `Imagix3DConfig`

Bases: `DefaultConfig`

A specialized configuration for Imagix3D, inheriting from DefaultConfig.

### `check_target_shape_3d() -> Imagix3DConfig`

Ensures that the fixed 3D target image dimensions (if provided) are compatible with the number of CNN layers.

### `set_target_multiple() -> Imagix3DConfig`

Ensures that targeted 3D image dimensions are compatible with the number of CNN layers.

### `validate_spatial_shape_settings()`

Check that `target_shape_3d` matches the chosen `spatial_shape_policy`.

`target_shape_3d` must be None for the "\*\_to_multiple" policies and must be set for "crop_or_pad_to_shape".

Returns:

| Type | Description           |
| ---- | --------------------- |
|      | The validated config. |

Raises:

| Type         | Description                                         |
| ------------ | --------------------------------------------------- |
| `ValueError` | If target_shape_3d and the policy are inconsistent. |

(LLM-generated docstring, pending human review.)

## `MaskixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig.

## `OntixConfig`

Bases: `DefaultConfig`

A specialized configuration for Ontix that only allows scaling methods guaranteeing non-negative outputs.

### `validate_nested_scaling() -> OntixConfig`

Ensures that any scaling method set within DataInfo is also a valid positive-value scaler.

## `StackixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig. For Stackix, `save_memory` is always False (feature not supported).

## `SupervisixConfig`

Bases: `VarixConfig`

A specialized configuration for supervisix, inheriting from DefaultConfig.

## `VanillixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig.

## `VarixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig.

## `XModalixConfig`

Bases: `DefaultConfig`

A specialized configuration inheriting from DefaultConfig.

This class overrides specific training parameters like pretrain_epochs and beta for the XModalix model, while inheriting all other settings.
