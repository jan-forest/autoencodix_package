from typing import Optional
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._result import Result


class Predictor:
    def __init__(self):
        pass

    def predict(self, data: Optional[BaseDataset], model: torch.nn.Module) -> Result:
        
        return Result()
