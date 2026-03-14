from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from ramankit.core._validation import AxisDirection, NumericArray, coerce_axis, coerce_intensity
from ramankit.core.metadata import Metadata, Provenance, ensure_metadata, ensure_provenance


@dataclass(frozen=True, slots=True, init=False)
class Spectrum:
    """Represent one spectrum with explicit spectral-axis semantics."""

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
        """Create a validated spectrum instance.

        Args:
            axis: One-dimensional spectral axis values.
            intensity: One-dimensional intensity values aligned with ``axis``.
            metadata: Scientific metadata attached to the spectrum.
            provenance: Provenance describing how the spectrum was created.
            spectral_axis_name: Explicit semantic label for the spectral axis.
            spectral_unit: Explicit unit for the spectral axis values.

        Raises:
            ValueError: If the axis or intensity arrays are invalid or incompatible.
        """

        axis_array, axis_direction = coerce_axis(axis)
        intensity_array = coerce_intensity(intensity, ndim=1, label="Spectrum intensity")
        if axis_array.shape != intensity_array.shape:
            raise ValueError(
                "Expected Spectrum axis and intensity to have the same shape; "
                f"got {axis_array.shape} and {intensity_array.shape}."
            )

        object.__setattr__(self, "axis", axis_array)
        object.__setattr__(self, "intensity", intensity_array)
        object.__setattr__(self, "metadata", ensure_metadata(metadata))
        object.__setattr__(self, "provenance", ensure_provenance(provenance))
        object.__setattr__(self, "spectral_axis_name", spectral_axis_name)
        object.__setattr__(self, "spectral_unit", spectral_unit)
        object.__setattr__(self, "axis_direction", axis_direction)

    @property
    def n_points(self) -> int:
        """Return the number of spectral points."""

        return int(self.intensity.shape[0])

    def copy(self) -> Spectrum:
        """Return a detached copy of the spectrum."""

        return Spectrum(
            axis=self.axis,
            intensity=self.intensity,
            metadata=self.metadata,
            provenance=self.provenance,
            spectral_axis_name=self.spectral_axis_name,
            spectral_unit=self.spectral_unit,
        )
