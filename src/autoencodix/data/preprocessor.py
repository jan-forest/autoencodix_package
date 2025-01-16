import torch
from typing import Union
import numpy as np
import pandas as pd
from anndata import AnnData #type: ignore
from autoencodix.base._base_preprocessor import BasePreprocessor

# internal check done
# write tests: TODO
class Preprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()
