"""Expose the public ENTRAP estimator, configuration, and result classes."""

from .estimator import ENTRAP
from .config import EntrapConfig
from .results import EntrapResults
__version__ = "1.1.0.dev1"
__all__ = ["ENTRAP", "EntrapConfig", "EntrapResults"]
