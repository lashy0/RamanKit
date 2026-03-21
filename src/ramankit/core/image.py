from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt

from ramankit.core._nd import (
    assign_spectral_nd,
    build_spectrum_from,
    coerce_spectral_nd,
    rebuild_like,
)
from ramankit.core._validation import AxisDirection, NumericArray
from ramankit.core.collection import SpectrumCollection
from ramankit.core.metadata import Metadata, Provenance
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
            axis: One-dimensional spectral axis shared by every pixel.
            intensity: Three-dimensional intensity array of shape
                ``(height, width, n_points)``.
            metadata: Scientific metadata attached to the image.
            provenance: Provenance describing how the image was created.
            spectral_axis_name: Explicit semantic label for the spectral axis.
            spectral_unit: Explicit unit for the spectral axis values.

        Raises:
            ValueError: If the axis or intensity arrays are invalid or incompatible.
        """

        validated = coerce_spectral_nd(
            axis,
            intensity,
            ndim=3,
            container_name="RamanImage",
            metadata=metadata,
            provenance=provenance,
            spectral_axis_name=spectral_axis_name,
            spectral_unit=spectral_unit,
        )
        assign_spectral_nd(self, validated)

    @classmethod
    def load(cls, path: str | Path) -> RamanImage:
        """Load one image from the built-in NPZ format."""

        from ramankit.io.npz import NPZLoader

        loaded = NPZLoader().load(path)
        if not isinstance(loaded, cls):
            raise ValueError(f"Expected {cls.__name__} in NPZ file; got {type(loaded).__name__}.")
        return loaded

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

        return build_spectrum_from(
            self,
            intensity=self.intensity[row, column],
            provenance=self.provenance,
        )

    def copy(self) -> RamanImage:
        """Return a detached copy of the Raman image."""

        return rebuild_like(self, intensity=self.intensity, provenance=self.provenance)

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

    def save(self, path: str | Path) -> None:
        """Persist this image in the built-in NPZ format."""

        from ramankit.io.npz import NPZSaver

        NPZSaver().save(self, path)

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
