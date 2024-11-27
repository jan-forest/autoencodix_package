# SPDX-FileCopyrightText: 2024-present joas <joas@informatik.uni-leipzig.de>
# SPDX-License-Identifier: MIT
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("autoencodix")
except PackageNotFoundError:
    __version__ = "unknown"

from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.core._model_interface import ModelInterface
from autoencodix.src.models._vanillix_model import VanillixModel

__all__ = [
    'Preprocessor', 
    'ModelInterface', 
    'VanillixModel',
    '__version__'
]