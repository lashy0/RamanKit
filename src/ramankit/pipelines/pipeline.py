from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ramankit.preprocessing._types import Array1D, Array2D, SpectralDataT
from ramankit.preprocessing._utils import apply_axis_transform, apply_spectral_transform


class _BaseStep:
    """Define shared metadata and provenance behavior for preprocessing steps."""

    function_name: ClassVar[str]
    method_name: ClassVar[str]

    def parameters(self) -> dict[str, object]:
        """Return structured parameters for provenance recording."""

        return {}


class PreprocessingStep(_BaseStep):
    """Define one configured axis-preserving preprocessing transform.

    Subclasses provide a concrete 1D intensity transform and metadata
    describing the preprocessing family and method name used for provenance
    recording. Implementations must preserve the spectral axis exactly.
    """

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
            batch_transform=self._transform_batch,
            function_name=self.function_name,
            method=self.method_name,
            parameters=self.parameters(),
        )

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        raise NotImplementedError

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        return None


class AxisTransformStep(_BaseStep):
    """Define one configured preprocessing transform that may change the axis.

    Subclasses provide a concrete 1D transform that returns both a new spectral
    axis and matching intensity values. Implementations may preserve or change
    the axis, but any axis change is explicit in the type of step used.
    """

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply this axis-aware step and return a new container."""

        return apply_axis_transform(
            data,
            transform=self._transform_with_axis,
            batch_transform=self._transform_batch_with_axis,
            function_name=self.function_name,
            method=self.method_name,
            parameters=self.parameters(),
        )

    def _transform_with_axis(
        self,
        intensity: Array1D,
        axis: Array1D,
    ) -> tuple[Array1D, Array1D]:
        raise NotImplementedError

    def _transform_batch_with_axis(
        self,
        intensity: Array2D,
        axis: Array1D,
    ) -> tuple[Array1D, Array2D] | None:
        return None


type PipelineStep = PreprocessingStep | AxisTransformStep


class Pipeline:
    """Apply a sequence of preprocessing steps in order."""

    def __init__(self, steps: Sequence[PipelineStep]) -> None:
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
