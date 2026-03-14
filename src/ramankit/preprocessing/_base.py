from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ramankit.preprocessing._types import Array1D, SpectralDataT
from ramankit.preprocessing._utils import apply_spectral_transform


class PreprocessingStep:
    """Base class for configured preprocessing steps."""

    function_name: ClassVar[str]
    method_name: ClassVar[str]

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply this preprocessing step to spectral data."""

        return apply_spectral_transform(
            data,
            transform=self._transform,
            function_name=self.function_name,
            method=self.method_name,
            parameters=self.parameters(),
        )

    def parameters(self) -> dict[str, object]:
        """Return structured parameters for provenance recording."""

        return {}

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        raise NotImplementedError


class Pipeline:
    """Apply a sequence of preprocessing steps to spectral data."""

    def __init__(self, steps: Sequence[PreprocessingStep]) -> None:
        self.steps = tuple(steps)

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply all configured preprocessing steps in order."""

        result = data
        for step in self.steps:
            result = step.apply(result)
        return result
