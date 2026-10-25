"""HyperDiscover: joint discovery of higher-order network structure and symbolic equations."""

from . import benchmarks
from .api import HyperDiscover
from .systems.base import Structure

__version__ = "0.1.0.dev0"
__all__ = ["HyperDiscover", "Structure", "benchmarks", "__version__"]
