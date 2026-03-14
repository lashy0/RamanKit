from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from ramankit.core._validation import (
    AxisDirection,
    NumericArray,
    coerce_axis,
    coerce_intensity,
    validate_axis_length,
)
from ramankit.core.collection import SpectrumCollection
from ramankit.core.metadata import Metadata, Provenance, ensure_metadata, ensure_provenance
from ramankit.core.spectrum import Spectrum


@dataclass(frozen=True, slots=True, init=False)
class RamanImage:
    """Represent a hyperspectral Raman image with a shared spectral axis."""

    axis: NumericArray
    intensity: NumericArray
    metadata: Metadata
    provenance: Provenance
    spectral_axis_name: str | None
    spectral_unit: str | None
    axis_direction: AxisDirection

    def __init__(
        self,
        axis: npt.ArrayLike,
        intensity: npt.ArrayLike,
        *,
        metadata: Metadata | None = None,
        provenance: Provenance | None = None,
        spectral_axis_name: str | None = None,
        spectral_unit: str | None = None,
    ) -> None:
        """Create a validated Raman image."""

        axis_array, axis_direction = coerce_axis(axis)
        intensity_array = coerce_intensity(intensity, ndim=3, label="RamanImage intensity")
        validate_axis_length(axis_array, intensity_array.shape[-1], label="RamanImage intensity")

        object.__setattr__(self, "axis", axis_array)
        object.__setattr__(self, "intensity", intensity_array)
        object.__setattr__(self, "metadata", ensure_metadata(metadata))
        object.__setattr__(self, "provenance", ensure_provenance(provenance))
        object.__setattr__(self, "spectral_axis_name", spectral_axis_name)
        object.__setattr__(self, "spectral_unit", spectral_unit)
        object.__setattr__(self, "axis_direction", axis_direction)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the image spatial shape as ``(height, width)``."""

        return int(self.intensity.shape[0]), int(self.intensity.shape[1])

    @property
    def n_points(self) -> int:
        """Return the number of spectral points per pixel."""

        return int(self.intensity.shape[2])

    @property
    def n_pixels(self) -> int:
        """Return the total number of spatial pixels."""

        height, width = self.spatial_shape
        return height * width

    def pixel(self, row: int, column: int) -> Spectrum:
        """Return the spectrum stored at one image pixel."""

        return Spectrum(
            axis=self.axis,
            intensity=self.intensity[row, column],
            metadata=self.metadata,
            provenance=self.provenance,
            spectral_axis_name=self.spectral_axis_name,
            spectral_unit=self.spectral_unit,
        )

    def copy(self) -> RamanImage:
        """Return a detached copy of the Raman image."""

        return RamanImage(
            axis=self.axis,
            intensity=self.intensity,
            metadata=self.metadata,
            provenance=self.provenance,
            spectral_axis_name=self.spectral_axis_name,
            spectral_unit=self.spectral_unit,
        )

    def flatten(self) -> SpectrumCollection:
        """Flatten the image into a spectrum collection."""

        from ramankit.core.operations import flatten_image

        return flatten_image(self)

    def add(self, other: RamanImage | float | int) -> RamanImage:
        """Return the elementwise sum of this image and an operand."""

        from ramankit.core.operations import add

        return add(self, other)

    def subtract(self, other: RamanImage | float | int) -> RamanImage:
        """Return the elementwise difference of this image and an operand."""

        from ramankit.core.operations import subtract

        return subtract(self, other)

    def multiply(self, other: RamanImage | float | int) -> RamanImage:
        """Return the elementwise product of this image and an operand."""

        from ramankit.core.operations import multiply

        return multiply(self, other)

    def divide(self, other: RamanImage | float | int) -> RamanImage:
        """Return the elementwise quotient of this image and an operand."""

        from ramankit.core.operations import divide

        return divide(self, other)

    def mean(self) -> Spectrum:
        """Return the mean spectrum of the image."""

        from ramankit.core.operations import mean

        return mean(self)

    def sum(self) -> Spectrum:
        """Return the summed spectrum of the image."""

        from ramankit.core.operations import sum

        return sum(self)

    def std(self) -> Spectrum:
        """Return the standard-deviation spectrum of the image."""

        from ramankit.core.operations import std

        return std(self)

    def __add__(self, other: RamanImage | float | int) -> RamanImage:
        return self.add(other)

    def __sub__(self, other: RamanImage | float | int) -> RamanImage:
        return self.subtract(other)

    def __mul__(self, other: RamanImage | float | int) -> RamanImage:
        return self.multiply(other)

    def __truediv__(self, other: RamanImage | float | int) -> RamanImage:
        return self.divide(other)

