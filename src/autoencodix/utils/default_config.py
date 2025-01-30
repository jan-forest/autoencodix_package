from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict, Any, Optional


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

    # Model configuration -----------------------------------------------------
    latent_dim: int = Field(
        default=16, ge=1, description="Dimension of the latent space"
    )
    n_layers: int = Field(
        default=3, ge=1, description="Number of layers in encoder/decoder"
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
