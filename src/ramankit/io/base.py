from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.spectrum import Spectrum


class BaseLoader[SpectralContainerT: (Spectrum, SpectrumCollection, RamanImage)](ABC):
    """Define the generic contract for loading one spectral container from a path."""

    @abstractmethod
    def load(self, path: str | Path) -> SpectralContainerT:
        """Load one spectral container from a file or directory path."""


class BaseSaver[SpectralContainerT: (Spectrum, SpectrumCollection, RamanImage)](ABC):
    """Define the generic contract for saving one spectral container to a path."""

    @abstractmethod
    def save(self, data: SpectralContainerT, path: str | Path) -> None:
        """Save one spectral container to a file or directory path."""
