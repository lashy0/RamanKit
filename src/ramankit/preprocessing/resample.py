from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ramankit.core._validation import NumericArray, coerce_axis
from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D, SpectralDataT
from ramankit.preprocessing._utils import apply_axis_transform


@dataclass(frozen=True, slots=True)
class Linear(PreprocessingStep):
    """Linear interpolation onto an explicitly provided spectral axis."""

    function_name = "resample"
    method_name = "linear"

    target_axis: npt.ArrayLike

    def __post_init__(self) -> None:
        axis, _ = coerce_axis(self.target_axis)
        object.__setattr__(self, "target_axis", axis)

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply linear resampling to spectral data."""

        target_axis = self.target_axis_array
        source_min = float(np.min(data.axis))
        source_max = float(np.max(data.axis))
        target_min = float(np.min(target_axis))
        target_max = float(np.max(target_axis))
        if target_min < source_min or target_max > source_max:
            raise ValueError("Expected target_axis to stay within the source axis range.")

        return apply_axis_transform(
            data,
            transform=self._transform_with_axis,
            batch_transform=self._transform_batch_with_axis,
            function_name=self.function_name,
            method=self.method_name,
            parameters={"target_axis": target_axis},
        )

    @property
    def target_axis_array(self) -> NumericArray:
        """Return the validated target axis as a NumPy array."""

        return np.array(self.target_axis, dtype=np.float64, copy=True)

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        raise NotImplementedError("Linear resampling uses _transform_with_axis instead.")

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        source_axis = axis
        source_intensity = intensity
        if source_axis[0] > source_axis[-1]:
            source_axis = source_axis[::-1]
            source_intensity = source_intensity[::-1]

        target_axis = self.target_axis_array
        target_for_interp = target_axis if target_axis[0] < target_axis[-1] else target_axis[::-1]
        resampled = np.interp(target_for_interp, source_axis, source_intensity)
        if target_axis[0] > target_axis[-1]:
            resampled = resampled[::-1]
        return target_axis, np.asarray(resampled, dtype=np.float64)
