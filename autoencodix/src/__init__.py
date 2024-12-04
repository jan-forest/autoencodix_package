# SPDX-FileCopyrightText: 2024-present joas <joas@informatik.uni-leipzig.de>
# SPDX-License-Identifier: MIT
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("autoencodix")
except PackageNotFoundError:
    __version__ = "unknown"

from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.core._base_pipeline import BasePipeline
from autoencodix.src.models.vanillix import VanillixModel

__all__ = ["Preprocessor", "BasePipeline", "VanillixModel", "__version__"]
