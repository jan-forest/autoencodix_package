"""Backward-compatible re-exports of loss classes.

Each loss implementation has been moved to its own module under
``autoencodix.utils``. This module keeps the historical import path
``autoencodix.utils._losses`` working during the refactor.
"""

from autoencodix.losses.vanillix_loss import VanillixLoss
from autoencodix.losses.varix_loss import VarixLoss
from autoencodix.losses.xmodal_loss import XModalLoss
from autoencodix.losses.disentanglix_loss import DisentanglixLoss
from autoencodix.losses.maskix_loss import MaskixLoss

__all__ = [
    "VanillixLoss",
    "VarixLoss",
    "XModalLoss",
    "DisentanglixLoss",
    "MaskixLoss",
]
