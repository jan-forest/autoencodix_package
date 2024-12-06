from typing import Optional, Union
from autoencodix.src.utils._result import Result
from autoencodix.src.utils.default_config import DefaultConfig, config_method
import torch


class Trainer:
    """
    Handles model training process
    """
    def __init__(self):
        pass
    def train(
        self,
        model: Optional[torch.nn.Module],
        train: torch.utils.data.Dataset,
        valid: torch.utils.data.Dataset,
        result: Result,
        config: Optional[Union[DefaultConfig, None]] = None,
    ) -> Result:
        """
        Train the model
        """
        return result
