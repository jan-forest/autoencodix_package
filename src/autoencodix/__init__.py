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
from .imagix3d import Imagix3D
from .maskix import Maskix
<<<<<<< HEAD
from .xontix import XOntix
=======
from .tuning import propose_initial_config

>>>>>>> 1bd54d5d7412907f8539ac8e3e4f7a7f5104316a
__all__ = [
    "Vanillix",
    "Varix",
    "Stackix",
    "Ontix",
    "XModalix",
    "Imagix",
    "Imagix3D",
    "Disentanglix",
    "Maskix",
<<<<<<< HEAD
    "XOntix",
=======
    "propose_initial_config",
>>>>>>> 1bd54d5d7412907f8539ac8e3e4f7a7f5104316a
]
