from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.spectrum import Spectrum


class BaseLoader[SpectralContainerT: (Spectrum, SpectrumCollection, RamanImage)](ABC):
    """Define the generic contract for loading one spectral container from a path.

    Concrete subclasses are responsible for mapping a file or directory path to
    one validated RamanKit data container.
    """

    @abstractmethod
    def load(self, path: str | Path) -> SpectralContainerT:
        """Load one spectral container from a file or directory path.

        Args:
            path: The input location for one container.

        Returns:
            A validated RamanKit spectral container.
        """


class BaseSaver[SpectralContainerT: (Spectrum, SpectrumCollection, RamanImage)](ABC):
    """Define the generic contract for saving one spectral container to a path."""

    @abstractmethod
    def save(self, data: SpectralContainerT, path: str | Path) -> None:
        """Save one spectral container to a file or directory path.

        Args:
            data: The container instance to persist.
            path: The output location for the serialized container.
        """
