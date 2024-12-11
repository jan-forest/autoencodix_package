from typing import Optional

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._result import Result


class Predictor:
    def __init__(self):
        pass

    def predict(self, data: Optional[BaseDataset]) -> Result:
        return Result()
