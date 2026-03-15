from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ramankit.preprocessing._types import Array1D, SpectralDataT
from ramankit.preprocessing._utils import apply_spectral_transform


class PreprocessingStep:
    """Define one configured preprocessing transform.

    Subclasses provide a concrete 1D spectral transform and metadata describing
    the preprocessing family and method name used for provenance recording.
    """

    function_name: ClassVar[str]
    method_name: ClassVar[str]

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply this step to spectral data and return a new container.

        Args:
            data: A supported spectral container.

        Returns:
            A new container of the same type with updated intensity values and
            one appended provenance step.
        """

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
    """Apply a sequence of preprocessing steps in order."""

    def __init__(self, steps: Sequence[PreprocessingStep]) -> None:
        """Create a preprocessing pipeline from configured steps."""

        self.steps = tuple(steps)

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply all configured preprocessing steps in order.

        Args:
            data: A supported spectral container.

        Returns:
            A new container of the same type after all steps have been applied.
        """

        result = data
        for step in self.steps:
            result = step.apply(result)
        return result
