"""Shared pytest configuration."""

import os

# Deterministic CuBLAS operations (needed by the tests that train with
# ``reproducible=True`` on CUDA) require this variable to be set before torch
# initializes CUDA. See the README FAQ.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
