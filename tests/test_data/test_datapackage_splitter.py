import pytest


import numpy as np
import pandas as pd

from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.default_config import (
    DefaultConfig,
    DataInfo,
    DataConfig,
    DataCase,
)
import anndata as ad  # type: ignore
import mudata as md  # type: ignore

"""
change for new PairedUnpairedSplitter
"""
