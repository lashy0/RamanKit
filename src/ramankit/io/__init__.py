"""Expose generic I/O extension points and built-in NPZ persistence."""

from ramankit.io import bwtek, npz
from ramankit.io.base import BaseLoader, BaseSaver

__all__ = ["BaseLoader", "BaseSaver", "bwtek", "npz"]
