from enum import Enum
from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator


class DataCase(str, Enum):
    MULTI_SINGLE_CELL = "Multi Single Cell"
    MULTI_BULK = "Multi Bulk"
    BULK_TO_BULK = "Bulk<->Bulk"
    IMG_TO_BULK = "IMG<->Bulk"
    SINGLE_CELL_TO_SINGLE_CELL = "Single Cell<->Single Cell"
    SINGLE_CELL_TO_IMG = "Single Cell<->IMG"
    IMG_TO_IMG = "IMG<->IMG"


class ConfigValidationError(Exception):
    pass


class DataInfo(BaseModel):
    # general -------------------------------------
    file_path: str
    data_type: Literal["NUMERIC", "CATEGORICAL", "IMG", "ANNOTATION"] = Field(
        default="NUMERIC"
    )
    scaling: Literal["STANDARD", "MINMAX", "ROBUST", "NONE"] = Field(default="STANDARD")
    filtering: Literal["VAR", "MAD", "CORR", "VARCORR", "NOFILT", "NONZEROVAR"] = Field(
        default="VAR"
    )
    k_filter: Optional[int] = Field(
        default=None,
        description="Number of features (columns) to keep, default keeps all",
    )
    n_filter: Optional[int] = Field(
        default=None, description="Number of samples (rows) to keep, default: keeps all"
    )
    sep: Optional[str] = Field(default=None)  # for pandas read_csv
    extra_anno_file: Optional[str] = Field(default=None)

    # single cell specific -------------------------
    is_single_cell: bool = Field(default=False)
    
    selected_layers: Optional[List[str]] = Field(
        default=None
    )  # if None, only X is used
    is_X: Optional[bool] = Field(default=None)  # only for single cell data
    # image specific ------------------------------
    img_root: Optional[str] = Field(default=None)
    img_width_resize: Optional[int] = Field(default=None)
    img_height_resize: Optional[int] = Field(default=None)
    # annotation specific -------------------------
    # xmodalix specific -------------------------
    translate_direction: Optional[Literal["from", "to"]] = Field(default=None)


class DataConfig(BaseModel):
    data_info: Dict[str, DataInfo]
    require_common_cells: Optional[bool] = Field(default=False)
    annotation_columns: Optional[List[str]] = Field(default=None)


# internal check done
# write tests: done
class DefaultConfig(BaseModel):
    """
    Complete configuration for model, training, hardware, and data handling.

    Attributes
    ----------
    all configuration parameters (change over time of development)

    Methods
    -------
    update(**kwargs)
        Update configuration with support for nested attributes.
    get_params()
        Get detailed information about all config fields including types and default values.
    print_schema()
        Print a human-readable schema of all config parameters.

    """

    # Datasets configuration --------------------------------------------------
    data_config: DataConfig = DataConfig(data_info={})
    paired_translation: bool = Field(
        True,
        description="Indicator if the samples for the xmodalix are paired, based on some sample id",
    )

    data_case: Optional[DataCase] = Field(
        None, description="Data case for the model, will be determined automatically"
    )
    # Model configuration -----------------------------------------------------
    latent_dim: int = Field(
        default=16, ge=1, description="Dimension of the latent space"
    )
    n_layers: int = Field(
        default=3,
        ge=0,
        description="Number of layers in encoder/decoder, without latent layer. If 0, is only the latent layer.",
    )
    enc_factor: int = Field(
        default=4, ge=1, description="Scaling factor for encoder dimensions"
    )
    input_dim: int = Field(default=10000, ge=1, description="Input dimension")
    drop_p: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )

    # Training configuration --------------------------------------------------
    learning_rate: float = Field(
        default=0.001, gt=0, description="Learning rate for optimization"
    )
    batch_size: int = Field(default=32, ge=1, description="Number of samples per batch")
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    weight_decay: float = Field(
        default=0.01, ge=0, description="L2 regularization factor"
    )
    reconstruction_loss: Literal["mse", "bce"] = Field(
        default="mse", description="Type of reconstruction loss"
    )
    default_vae_loss: Literal["kl", "mmd"] = Field(
        default="kl", description="Type of VAE loss"
    )
    loss_reduction: Literal["sum", "mean"] = Field(
        default="mean",
        description="Loss reduction in PyTorch i.e in torch.nn.functional.binary_cross_entropy_with_logits(reduction=loss_reduction)",
    )
    beta: float = Field(
        default=0.1, ge=0, description="Beta weighting factor for VAE loss"
    )
    min_samples_per_split: int = Field(
        default=1, ge=1, description="Minimum number of samples per split"
    )

    # Hardware configuration --------------------------------------------------
    device: Literal["cpu", "cuda", "gpu", "tpu", "mps", "auto"] = Field(
        default="auto", description="Device to use"
    )
    # 0 uses cpu and not gpu
    n_gpus: int = Field(default=1, ge=1, description="Number of GPUs to use")
    n_workers: int = Field(
        default=2, ge=0, description="Number of data loading workers"
    )
    checkpoint_interval: int = Field(
        default=1, ge=1, description="Interval for saving checkpoints"
    )
    float_precision: Literal[
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        "64",
        "32",
        "16",
        "bf16",
    ] = Field(default="32", description="Floating point precision")
    gpu_strategy: Literal[
        "auto",
        "dp",
        "ddp",
        "ddp_spawn",
        "ddp_find_unused_parameters_true",
        "xla",
        "deepspeed",
        "fsdp",
    ] = Field(default="auto", description="GPU parallelization strategy")

    # Data handling configuration ---------------------------------------------
    train_ratio: float = Field(
        default=0.7, gt=0, lt=1, description="Ratio of data for training"
    )
    test_ratio: float = Field(
        default=0.2, gt=0, lt=1, description="Ratio of data for testing"
    )
    valid_ratio: float = Field(
        default=0.1, gt=0, lt=1, description="Ratio of data for validation"
    )

    # General configuration ---------------------------------------------------
    reproducible: bool = Field(
        default=True, description="Whether to ensure reproducibility"
    )
    global_seed: int = Field(default=1, ge=0, description="Global random seed")

    ##### VALIDATION ##### -----------------------------------------------------
    ##### ----------------- -----------------------------------------------------
    @field_validator("data_config")
    @classmethod
    def validate_data_config(cls, data_config: DataConfig):
        """Main validation logic for dataset consistency and translation."""
        data_info = data_config.data_info

        numeric_count = sum(
            1 for info in data_info.values() if info.data_type == "NUMERIC"
        )
        img_count = sum(1 for info in data_info.values() if info.data_type == "IMG")

        if numeric_count == 0 and img_count == 0:
            raise ConfigValidationError("At least one NUMERIC dataset is required.")

        numeric_datasets = [
            info for info in data_info.values() if info.data_type == "NUMERIC"
        ]
        if numeric_datasets:
            is_single_cell = numeric_datasets[0].is_single_cell
            if any(info.is_single_cell != is_single_cell for info in numeric_datasets):
                raise ConfigValidationError(
                    "All numeric datasets must be either single cell or bulk."
                )

        from_dataset = next(
            (
                (name, info)
                for name, info in data_info.items()
                if info.translate_direction == "from"
            ),
            None,
        )
        to_dataset = next(
            (
                (name, info)
                for name, info in data_info.items()
                if info.translate_direction == "to"
            ),
            None,
        )

        if bool(from_dataset) != bool(to_dataset):
            raise ConfigValidationError(
                "Translation requires exactly one 'from' and one 'to' dataset."
            )

        if from_dataset and to_dataset:
            from_info, to_info = from_dataset[1], to_dataset[1]
            if from_info.data_type == "NUMERIC" and to_info.data_type == "NUMERIC":
                if from_info.is_single_cell != to_info.is_single_cell:
                    raise ConfigValidationError(
                        "Cannot translate between single cell and bulk data."
                    )

        return data_config

    @model_validator(mode="after")
    def determine_case(self) -> "DefaultConfig":
        """Assign the correct DataCase after model validation."""
        data_info = self.data_config.data_info

        # Handle empty data_info case
        if not data_info:
            return self

        # Find 'from' and 'to' datasets
        from_dataset = next(
            (
                (name, info)
                for name, info in data_info.items()
                if info.translate_direction == "from"
            ),
            None,
        )
        to_dataset = next(
            (
                (name, info)
                for name, info in data_info.items()
                if info.translate_direction == "to"
            ),
            None,
        )

        try:
            if from_dataset and to_dataset:
                from_info, to_info = from_dataset[1], to_dataset[1]
                if from_info.data_type == "NUMERIC" and to_info.data_type == "NUMERIC":
                    self.data_case = (
                        DataCase.SINGLE_CELL_TO_SINGLE_CELL
                        if from_info.is_single_cell
                        else DataCase.BULK_TO_BULK
                    )
                elif "IMG" in {from_info.data_type, to_info.data_type}:
                    numeric_dataset = (
                        from_info if from_info.data_type == "NUMERIC" else to_info
                    )
                    # check for IMG_IMG
                    if from_info.data_type == "IMG" and to_info.data_type == "IMG":
                        self.data_case = DataCase.IMG_TO_IMG
                    else:
                        self.data_case = (
                            DataCase.SINGLE_CELL_TO_IMG
                            if numeric_dataset.is_single_cell
                            else DataCase.IMG_TO_BULK
                        )
            else:
                # Find any numeric dataset
                numeric_datasets = [
                    info for info in data_info.values() if info.data_type == "NUMERIC"
                ]

                if not numeric_datasets:
                    raise ValueError("No numeric datasets found in data_info")

                numeric_dataset = numeric_datasets[0]
                self.data_case = (
                    DataCase.MULTI_SINGLE_CELL
                    if numeric_dataset.is_single_cell
                    else DataCase.MULTI_BULK
                )
        except Exception as e:
            # Log the error but don't fail validation
            import warnings

            warnings.warn(f"Could not determine data_case: {str(e)}")

        return self

    @field_validator("test_ratio", "valid_ratio")
    def validate_ratios(cls, v, values):
        total = (
            sum(
                values.data.get(key, 0)
                for key in ["train_ratio", "test_ratio", "valid_ratio"]
            )
            + v
        )
        if total > 1.0:
            raise ValueError(f"Data split ratios must sum to 1.0 or less (got {total})")
        return v

    # model specific validation
    paired_translation: bool = True

    # TODO test if other float precisions work with MPS
    @field_validator("float_precision")
    def validate_float_precision(cls, v, values):
        """Validate float precision based on device type."""
        device = values.data["device"]
        if device == "mps" and v != "32":
            raise ValueError("MPS backend only supports float precision '32'")
        return v

    # gpu strategy needs to be auto for mps # TODO test if other strategies work
    @field_validator("gpu_strategy")
    def validate_gpu_strategy(cls, v, values):
        device = values.data.get("device")
        if device == "mps" and v != "auto":
            raise ValueError("MPS backend only supports GPU strategy 'auto'")

    #### END VALIDATION #### --------------------------------------------------

    #### READIBILITY #### ------------------------------------------------------
    #### ------------ #### ------------------------------------------------------
    @classmethod
    def get_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all config fields including types and default values.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing field name, type, default value, and description if available
        """
        fields_info = {}
        for name, field in cls.model_fields.items():
            fields_info[name] = {
                "type": str(field.annotation),
                "default": field.default,
                "description": field.description or "No description available",
            }
        return fields_info

    @classmethod
    def print_schema(cls, filter_params: Optional[None] = None) -> None:
        """
        Print a human-readable schema of all config parameters.
        """
        if filter_params:
            filter_params = list(filter_params)
            print("Valid Keyword Arguments:")
            print("-" * 50)
        else:
            print(f"\n{cls.__name__} Configuration Parameters:")
            print("-" * 50)

        for name, info in cls.get_params().items():
            if filter_params and name not in filter_params:
                continue
            print(f"\n{name}:")
            print(f"  Type: {info['type']}")
            print(f"  Default: {info['default']}")
            print(f"  Description: {info['description']}")
