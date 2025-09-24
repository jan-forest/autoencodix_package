# Add your package to sys.path
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # adjust to reach src/

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Parse Google/NumPy style docstrings
]

# Theme
html_theme = "sphinx_rtd_theme"
