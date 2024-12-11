from typing import Optional

from src.autoencodix.base._base_dataset import BaseDataset
from src.autoencodix.utils._result import Result


class Predictor:
    def __init__(self):
        pass

    def predict(self, data: Optional[BaseDataset]) -> Result:
        return Result()
