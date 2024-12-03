import torch
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data

        Args:
            data (np.ndarray): Input data to preprocess

        Returns:
            np.ndarray: Preprocessed data
        """
        raise NotImplementedError("Preprocessor not implemented")

    def mock_preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Mock preprocess input data

        Args:
            data (np.ndarray): Input data to preprocess

        Returns:
            torch.Tensor: Preprocessed data
        """
        tensor = torch.from_numpy(data)
        return tensor
