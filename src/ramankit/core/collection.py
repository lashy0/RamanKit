from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class SpectrumCollection:
    """Represent a batch of spectra that share one spectral axis."""

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
        """Create a validated spectrum collection.

        Args:
            axis: One-dimensional spectral axis shared by every spectrum.
            intensity: Two-dimensional intensity array of shape
                ``(n_spectra, n_points)``.
            metadata: Scientific metadata attached to the collection.
            provenance: Provenance describing how the collection was created.
            spectral_axis_name: Explicit semantic label for the spectral axis.
            spectral_unit: Explicit unit for the spectral axis values.

        Raises:
            ValueError: If the axis or intensity arrays are invalid or incompatible.
        """

        axis_array, axis_direction = coerce_axis(axis)
        intensity_array = coerce_intensity(intensity, ndim=2, label="SpectrumCollection intensity")
        validate_axis_length(
            axis_array,
            intensity_array.shape[-1],
            label="SpectrumCollection intensity",
        )

        object.__setattr__(self, "axis", axis_array)
        object.__setattr__(self, "intensity", intensity_array)
        object.__setattr__(self, "metadata", ensure_metadata(metadata))
        object.__setattr__(self, "provenance", ensure_provenance(provenance))
        object.__setattr__(self, "spectral_axis_name", spectral_axis_name)
        object.__setattr__(self, "spectral_unit", spectral_unit)
        object.__setattr__(self, "axis_direction", axis_direction)

    @classmethod
    def from_spectra(
        cls,
        spectra: list[Spectrum] | tuple[Spectrum, ...],
    ) -> SpectrumCollection:
        """Build a collection from spectra sharing the same spectral axis."""

        from ramankit.core.operations import stack_spectra

        return stack_spectra(spectra)

    @classmethod
    def load(cls, path: str | Path) -> SpectrumCollection:
        """Load one collection from the built-in NPZ format."""

        from ramankit.io.npz import NPZLoader

        loaded = NPZLoader().load(path)
        if not isinstance(loaded, cls):
            raise ValueError(f"Expected {cls.__name__} in NPZ file; got {type(loaded).__name__}.")
        return loaded

    @property
    def n_spectra(self) -> int:
        """Return the number of spectra in the collection."""

        return int(self.intensity.shape[0])

    @property
    def n_points(self) -> int:
        """Return the number of spectral points per spectrum."""

        return int(self.intensity.shape[1])

    def __len__(self) -> int:
        """Return the number of spectra in the collection."""

        return self.n_spectra

    def __getitem__(self, item: int | slice) -> Spectrum | SpectrumCollection:
        """Return one spectrum or a sliced sub-collection.

        Integer indexing returns a `Spectrum`. Slice indexing returns a new
        `SpectrumCollection` that keeps the shared axis and metadata.
        """

        subset = self.intensity[item]
        if subset.ndim == 1:
            return Spectrum(
                axis=self.axis,
                intensity=subset,
                metadata=self.metadata,
                provenance=self.provenance,
                spectral_axis_name=self.spectral_axis_name,
                spectral_unit=self.spectral_unit,
            )
        return SpectrumCollection(
            axis=self.axis,
            intensity=subset,
            metadata=self.metadata,
            provenance=self.provenance,
            spectral_axis_name=self.spectral_axis_name,
            spectral_unit=self.spectral_unit,
        )

    def copy(self) -> SpectrumCollection:
        """Return a detached copy of the collection."""

        return SpectrumCollection(
            axis=self.axis,
            intensity=self.intensity,
            metadata=self.metadata,
            provenance=self.provenance,
            spectral_axis_name=self.spectral_axis_name,
            spectral_unit=self.spectral_unit,
        )

    def add(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        """Return the elementwise sum of this collection and an operand."""

        from ramankit.core.operations import add

        return add(self, other)

    def subtract(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        """Return the elementwise difference of this collection and an operand."""

        from ramankit.core.operations import subtract

        return subtract(self, other)

    def multiply(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        """Return the elementwise product of this collection and an operand."""

        from ramankit.core.operations import multiply

        return multiply(self, other)

    def divide(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        """Return the elementwise quotient of this collection and an operand."""

        from ramankit.core.operations import divide

        return divide(self, other)

    def save(self, path: str | Path) -> None:
        """Persist this collection in the built-in NPZ format."""

        from ramankit.io.npz import NPZSaver

        NPZSaver().save(self, path)

    def mean(self) -> Spectrum:
        """Return the mean spectrum of the collection."""

        from ramankit.core.operations import mean

        return mean(self)

    def sum(self) -> Spectrum:
        """Return the summed spectrum of the collection."""

        from ramankit.core.operations import sum

        return sum(self)

    def std(self) -> Spectrum:
        """Return the standard-deviation spectrum of the collection."""

        from ramankit.core.operations import std

        return std(self)

    def __add__(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        return self.add(other)

    def __sub__(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        return self.subtract(other)

    def __mul__(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        return self.multiply(other)

    def __truediv__(self, other: SpectrumCollection | float | int) -> SpectrumCollection:
        return self.divide(other)
