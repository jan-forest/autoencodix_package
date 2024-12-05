import torch
from typing import Union
import numpy as np
import pandas as pd
from anndata import AnnData


class Preprocessor:
    def __init__(self):
        self._preprocessed = False

    def preprocess(
        self, data: Union[np.ndarray, pd.DataFrame, AnnData]
    ) -> torch.Tensor:
        """
        Mock preprocess input data

        Args:
            data (np.ndarray): Input data to preprocess

        Returns:
            torch.Tensor: Preprocessed data
        """
        tensor = torch.from_numpy(data)
        self.preprocessed = True
        return tensor
