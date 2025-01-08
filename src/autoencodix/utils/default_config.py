from pydantic import BaseModel, Field, field_validator
from typing import Literal


class DefaultConfig(BaseModel):
    """Complete configuration for model, training, hardware, and data handling."""

    # Model configuration -----------------------------------------------------
    latent_dim: int = Field(16, ge=1, description="Dimension of the latent space")
    n_layers: int = Field(3, ge=1, description="Number of layers in encoder/decoder")
    enc_factor: int = Field(
        4, ge=1, description="Scaling factor for encoder dimensions"
    )
    input_dim: int = Field(10000, ge=1, description="Input dimension")
    drop_p: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability")

    # Training configuration --------------------------------------------------
    learning_rate: float = Field(
        0.001, gt=0, description="Learning rate for optimization"
    )
    batch_size: int = Field(32, ge=1, description="Number of samples per batch")
    epochs: int = Field(100, ge=1, description="Number of training epochs")
    weight_decay: float = Field(0.01, ge=0, description="L2 regularization factor")
    reconstruction_loss: Literal["mse", "bce"] = Field(
        "mse", description="Type of reconstruction loss"
    )
    default_vae_loss: Literal["kl"] = Field("kl", description="Type of VAE loss")
    min_samples_per_split: int = Field(1, ge=1, description="Minimum number of samples per split")

    # Hardware configuration --------------------------------------------------
    use_gpu: bool = Field(False, description="Whether to use GPU acceleration")
    n_gpus: int = Field(1, ge=0, description="Number of GPUs to use")
    n_devices: int = Field(1, ge=1, description="Number of devices for computation")
    n_workers: int = Field(2, ge=0, description="Number of data loading workers")
    checkpoint_interval: int = Field(10, ge=1, description="Interval for saving checkpoints")
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
    ] = Field("32", description="Floating point precision")
    print("float_precision", float_precision)
    gpu_strategy: Literal[
        "auto",
        "dp",
        "ddp",
        "ddp_spawn",
        "ddp_find_unused_parameters_true",
        "xla",
        "deepspeed",
        "fsdp",
    ] = Field("auto", description="GPU parallelization strategy")

    # Data handling configuration ---------------------------------------------
    train_ratio: float = Field(
        0.7, gt=0, lt=1, description="Ratio of data for training"
    )
    test_ratio: float = Field(0.2, gt=0, lt=1, description="Ratio of data for testing")
    valid_ratio: float = Field(
        0.1, gt=0, lt=1, description="Ratio of data for validation"
    )

    @field_validator("test_ratio", "valid_ratio")
    def validate_ratios(cls, v, values):
        total = (
            sum(
                values.get(key, 0)
                for key in ["train_ratio", "test_ratio", "valid_ratio"]
            )
            + v
        )
        if total > 1.0:
            raise ValueError(f"Data split ratios must sum to 1.0 or less (got {total})")
        return v

    # General configuration
    reproducible: bool = Field(True, description="Whether to ensure reproducibility")
    global_seed: int = Field(1, ge=0, description="Global random seed")

    def update(self, **kwargs):
        """Update configuration with support for nested attributes."""
        for key, value in kwargs.items():
            parts = key.split(".")
            if len(parts) > 1:
                section = parts[0]
                param = parts[1]
                if not hasattr(self, section):
                    raise ValueError(f"Unknown configuration section: {section}")
                section_value = getattr(self, section)
                if not hasattr(section_value, param):
                    raise ValueError(
                        f"Unknown parameter '{param}' in section '{section}'"
                    )
                setattr(section_value, param, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
