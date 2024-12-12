from dataclasses import dataclass


@dataclass
class DefaultConfig:
    """
    A dataclass for storing default configuration parameters.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the model (default: 0.001).
    batch_size : int
        The batch size for training (default: 32).
    epochs : int
        The number of epochs for training (default: 100).
    """

    # Model architecture and training parameters
    # TODO use pydantic to vlidate the configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    latent_dim: int = 16
    n_layers: int = 3
    enc_factor: int = 4
    input_dim: int = 10000
    drop_p: float = 0.1
    weight_decay: float = 0.01
    use_gpu: bool = False
    n_devices: int = 1
    # ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16')
    float_precision: str = "bf16-mixed"
    gpu_strategy: str = "auto" #"FSDP"
    checkpoint_interval: int = 10
    reconstruction_loss: str = "mse"
    default_vae_loss: str = "kl"

    # scientific options
    reproducible: bool = True
    global_seed: int = 1

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs of configuration parameters to update.

        Raises
        ------
        ValueError
            If an unknown configuration parameter is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
