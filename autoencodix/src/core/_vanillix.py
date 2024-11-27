from autoencodix.core._model_interface import ModelInterface
from autoencodix.preprocessing import Preprocessor
from typing import Dict, Any
import numpy as np
import torch

class Vanillix(ModelInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.preprocessor = Preprocessor()

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        return self.preprocessor.mock_preprocess(data)
    