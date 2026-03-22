from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramankit.core._validation import validate_axis_compatibility
from ramankit.core.spectrum import Spectrum
from ramankit.pipelines.pipeline import AxisTransformStep, PreprocessingStep
from ramankit.preprocessing._types import Array1D, Array2D, SpectralDataT


@dataclass(frozen=True, slots=True)
class Cropper(AxisTransformStep):
    """Crop spectra to an explicit inclusive axis range."""

    function_name = "crop"
    method_name = "axis_range"

    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("Expected crop bounds to be finite values.")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("Expected lower_bound to be smaller than upper_bound.")

    def parameters(self) -> dict[str, object]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        source_min = float(np.min(axis))
        source_max = float(np.max(axis))
        if self.lower_bound < source_min or self.upper_bound > source_max:
            raise ValueError("Expected crop bounds to stay within the source axis range.")

        mask = (axis >= self.lower_bound) & (axis <= self.upper_bound)
        if not np.any(mask):
            raise ValueError("Expected crop bounds to retain at least one spectral point.")
        return axis[mask], intensity[mask]

    def _transform_batch_with_axis(
        self, intensity: Array2D, axis: Array1D,
    ) -> tuple[Array1D, Array2D] | None:
        source_min = float(np.min(axis))
        source_max = float(np.max(axis))
        if self.lower_bound < source_min or self.upper_bound > source_max:
            raise ValueError("Expected crop bounds to stay within the source axis range.")
        mask = (axis >= self.lower_bound) & (axis <= self.upper_bound)
        if not np.any(mask):
            raise ValueError("Expected crop bounds to retain at least one spectral point.")
        return axis[mask], intensity[:, mask]


@dataclass(frozen=True, slots=True)
class IndexCropper(AxisTransformStep):
    """Crop spectra by index positions using Python-style slicing."""

    function_name = "crop"
    method_name = "index_range"

    start_index: int | None = None
    stop_index: int | None = None

    def __post_init__(self) -> None:
        if self.start_index is not None and not isinstance(self.start_index, int):
            raise TypeError("Expected start_index to be an integer.")
        if self.stop_index is not None and not isinstance(self.stop_index, int):
            raise TypeError("Expected stop_index to be an integer.")
        if self.start_index is None and self.stop_index is None:
            raise ValueError("Expected at least one of start_index or stop_index.")
        if (
            self.start_index is not None
            and self.stop_index is not None
            and self.start_index >= self.stop_index
        ):
            raise ValueError("Expected start_index to be smaller than stop_index.")

    def parameters(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "stop_index": self.stop_index,
        }

    def _transform_with_axis(
        self, intensity: Array1D, axis: Array1D,
    ) -> tuple[Array1D, Array1D]:
        start = self.start_index if self.start_index is not None else 0
        stop = self.stop_index if self.stop_index is not None else len(axis)
        cropped_axis = axis[start:stop]
        if len(cropped_axis) == 0:
            raise ValueError(
                "Expected index crop to retain at least one spectral point."
            )
        return cropped_axis, intensity[start:stop]

    def _transform_batch_with_axis(
        self, intensity: Array2D, axis: Array1D,
    ) -> tuple[Array1D, Array2D] | None:
        start = self.start_index if self.start_index is not None else 0
        stop = self.stop_index if self.stop_index is not None else len(axis)
        cropped_axis = axis[start:stop]
        if len(cropped_axis) == 0:
            raise ValueError(
                "Expected index crop to retain at least one spectral point."
            )
        return cropped_axis, intensity[:, start:stop]


@dataclass(frozen=True, slots=True)
class BackgroundSubtractor(PreprocessingStep):
    """Subtract a reference background spectrum from spectral data."""

    function_name = "background_subtract"
    method_name = "reference_spectrum"

    background: Spectrum

    def parameters(self) -> dict[str, object]:
        parameters: dict[str, object] = {
            "background_points": self.background.n_points,
        }
        if self.background.metadata.sample is not None:
            parameters["background_sample"] = self.background.metadata.sample
        if self.background.provenance.source is not None:
            parameters["background_source"] = self.background.provenance.source
        return parameters

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Subtract the configured background spectrum from spectral data."""

        validate_axis_compatibility(
            data.axis,
            self.background.axis,
            left_name=data.spectral_axis_name,
            right_name=self.background.spectral_axis_name,
            left_unit=data.spectral_unit,
            right_unit=self.background.spectral_unit,
        )
        return PreprocessingStep.apply(self, data)

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        return intensity - self.background.intensity
