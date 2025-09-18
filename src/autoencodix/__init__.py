from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("autoencodix")
except PackageNotFoundError:
    __version__ = "unknown"

# Import key classes to make them directly accessible
from .vanillix import Vanillix
from .varix import Varix
from .stackix import Stackix
from .ontix import Ontix
from .disentanglix import Disentanglix
from .xmodalix import XModalix
from .imagix import Imagix

__all__ = [
    "Vanillix",
    "Varix",
    "Stackix",
    "Ontix",
    "XModalix",
    "Imagix",
    "Disentanglix",
]
