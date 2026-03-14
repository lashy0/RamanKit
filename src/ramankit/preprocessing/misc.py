from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramankit.core._validation import validate_axis_compatibility
from ramankit.core.spectrum import Spectrum
from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D, SpectralDataT
from ramankit.preprocessing._utils import apply_axis_transform, apply_spectral_transform


@dataclass(frozen=True, slots=True)
class Cropper(PreprocessingStep):
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

    def apply(self, data: SpectralDataT) -> SpectralDataT:
        """Apply spectral cropping to data containers."""

        source_min = float(np.min(data.axis))
        source_max = float(np.max(data.axis))
        if self.lower_bound < source_min or self.upper_bound > source_max:
            raise ValueError("Expected crop bounds to stay within the source axis range.")

        return apply_axis_transform(
            data,
            transform=self._transform_with_axis,
            function_name=self.function_name,
            method=self.method_name,
            parameters=self.parameters(),
        )

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        raise NotImplementedError("Cropper uses _transform_with_axis instead.")

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        mask = (axis >= self.lower_bound) & (axis <= self.upper_bound)
        if not np.any(mask):
            raise ValueError("Expected crop bounds to retain at least one spectral point.")
        return axis[mask], intensity[mask]


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
        return apply_spectral_transform(
            data,
            transform=self._transform,
            function_name=self.function_name,
            method=self.method_name,
            parameters=self.parameters(),
        )

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        return intensity - self.background.intensity
