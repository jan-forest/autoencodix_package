# SPDX-FileCopyrightText: 2024-present joas <joas@informatik.uni-leipzig.de>
# SPDX-License-Identifier: MIT
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("autoencodix")
except PackageNotFoundError:
    __version__ = "unknown"
