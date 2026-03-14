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
        """Create a validated Raman image.

        Args:
            axis: Shared one-dimensional spectral axis values.
            intensity: Three-dimensional array with shape ``(height, width, n_points)``.
            metadata: Scientific metadata attached to the image.
            provenance: Provenance describing how the image was created.
            spectral_axis_name: Explicit semantic label for the spectral axis.
            spectral_unit: Explicit unit for the spectral axis values.

        Raises:
            ValueError: If the axis or intensity arrays are invalid or incompatible.
        """

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
