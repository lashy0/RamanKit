from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload

import numpy.typing as npt

from ramankit.core._validation import (
    AxisDirection,
    NumericArray,
    coerce_axis,
    coerce_intensity,
    validate_axis_compatibility,
    validate_axis_length,
)
from ramankit.core.metadata import Metadata, Provenance, ensure_metadata, ensure_provenance

if TYPE_CHECKING:
    from ramankit.core.collection import SpectrumCollection
    from ramankit.core.image import RamanImage
    from ramankit.core.spectrum import Spectrum


class _SpectralContainer(Protocol):
    @property
    def axis(self) -> NumericArray: ...

    @property
    def intensity(self) -> NumericArray: ...

    @property
    def metadata(self) -> Metadata: ...

    @property
    def provenance(self) -> Provenance: ...

    @property
    def spectral_axis_name(self) -> str | None: ...

    @property
    def spectral_unit(self) -> str | None: ...

    @property
    def axis_direction(self) -> AxisDirection: ...


@dataclass(frozen=True, slots=True)
class _SpectralND:
    axis: NumericArray
    intensity: NumericArray
    metadata: Metadata
    provenance: Provenance
    spectral_axis_name: str | None
    spectral_unit: str | None
    axis_direction: AxisDirection


@dataclass(frozen=True, slots=True)
class _SpectralBatch:
    rows: NumericArray
    leading_shape: tuple[int, ...]


def coerce_spectral_nd(
    axis: npt.ArrayLike,
    intensity: npt.ArrayLike,
    *,
    ndim: int,
    container_name: str,
    metadata: Metadata | None = None,
    provenance: Provenance | None = None,
    spectral_axis_name: str | None = None,
    spectral_unit: str | None = None,
) -> _SpectralND:
    """Return validated spectral container data with shared ND invariants."""

    axis_array, axis_direction = coerce_axis(axis)
    intensity_label = f"{container_name} intensity"
    intensity_array = coerce_intensity(intensity, ndim=ndim, label=intensity_label)

    if ndim == 1:
        if axis_array.shape != intensity_array.shape:
            raise ValueError(
                "Expected Spectrum axis and intensity to have the same shape; "
                f"got {axis_array.shape} and {intensity_array.shape}."
            )
    else:
        validate_axis_length(axis_array, intensity_array.shape[-1], label=intensity_label)

    return _SpectralND(
        axis=axis_array,
        intensity=intensity_array,
        metadata=ensure_metadata(metadata),
        provenance=ensure_provenance(provenance),
        spectral_axis_name=spectral_axis_name,
        spectral_unit=spectral_unit,
        axis_direction=axis_direction,
    )


def assign_spectral_nd(instance: object, values: _SpectralND) -> None:
    """Populate a frozen spectral container from validated ND data."""

    object.__setattr__(instance, "axis", values.axis)
    object.__setattr__(instance, "intensity", values.intensity)
    object.__setattr__(instance, "metadata", values.metadata)
    object.__setattr__(instance, "provenance", values.provenance)
    object.__setattr__(instance, "spectral_axis_name", values.spectral_axis_name)
    object.__setattr__(instance, "spectral_unit", values.spectral_unit)
    object.__setattr__(instance, "axis_direction", values.axis_direction)


def ensure_compatible_spectral_data(left: _SpectralContainer, right: _SpectralContainer) -> None:
    """Validate that two spectral containers can participate in elementwise arithmetic."""

    if type(left) is not type(right):
        raise ValueError("Spectral operands must have the same container type.")
    validate_axis_compatibility(
        left.axis,
        right.axis,
        left_name=left.spectral_axis_name,
        right_name=right.spectral_axis_name,
        left_unit=left.spectral_unit,
        right_unit=right.spectral_unit,
    )
    if left.intensity.shape != right.intensity.shape:
        raise ValueError(
            "Expected spectral operand shapes to match; "
            f"got {left.intensity.shape} and {right.intensity.shape}."
        )


def flatten_spectral_rows(data: _SpectralContainer) -> _SpectralBatch:
    """Flatten all non-spectral dimensions into batch rows."""

    return _SpectralBatch(
        rows=data.intensity.reshape(-1, data.intensity.shape[-1]),
        leading_shape=tuple(data.intensity.shape[:-1]),
    )


def restore_spectral_rows(rows: NumericArray, *, leading_shape: tuple[int, ...]) -> NumericArray:
    """Restore flattened spectral rows to the original non-spectral shape."""

    if leading_shape:
        return rows.reshape(*leading_shape, rows.shape[-1])
    return rows.reshape(rows.shape[-1])


def build_spectrum_from(
    data: _SpectralContainer,
    *,
    intensity: npt.ArrayLike,
    provenance: Provenance,
) -> Spectrum:
    """Build a Spectrum that inherits spectral semantics from another container."""

    from ramankit.core.spectrum import Spectrum

    return Spectrum(
        axis=data.axis,
        intensity=intensity,
        metadata=data.metadata,
        provenance=provenance,
        spectral_axis_name=data.spectral_axis_name,
        spectral_unit=data.spectral_unit,
    )


@overload
def rebuild_like(
    data: Spectrum,
    *,
    axis: npt.ArrayLike | None = None,
    intensity: npt.ArrayLike,
    provenance: Provenance,
) -> Spectrum: ...


@overload
def rebuild_like(
    data: SpectrumCollection,
    *,
    axis: npt.ArrayLike | None = None,
    intensity: npt.ArrayLike,
    provenance: Provenance,
) -> SpectrumCollection: ...


@overload
def rebuild_like(
    data: RamanImage,
    *,
    axis: npt.ArrayLike | None = None,
    intensity: npt.ArrayLike,
    provenance: Provenance,
) -> RamanImage: ...


def rebuild_like(
    data: _SpectralContainer,
    *,
    axis: npt.ArrayLike | None = None,
    intensity: npt.ArrayLike,
    provenance: Provenance,
) -> Spectrum | SpectrumCollection | RamanImage:
    """Rebuild a spectral container with updated axis or intensity values."""

    from ramankit.core.collection import SpectrumCollection
    from ramankit.core.image import RamanImage
    from ramankit.core.spectrum import Spectrum

    axis_values = data.axis if axis is None else axis
    if isinstance(data, Spectrum):
        return Spectrum(
            axis=axis_values,
            intensity=intensity,
            metadata=data.metadata,
            provenance=provenance,
            spectral_axis_name=data.spectral_axis_name,
            spectral_unit=data.spectral_unit,
        )
    if isinstance(data, SpectrumCollection):
        return SpectrumCollection(
            axis=axis_values,
            intensity=intensity,
            metadata=data.metadata,
            provenance=provenance,
            spectral_axis_name=data.spectral_axis_name,
            spectral_unit=data.spectral_unit,
        )

    return RamanImage(
        axis=axis_values,
        intensity=intensity,
        metadata=data.metadata,
        provenance=provenance,
        spectral_axis_name=data.spectral_axis_name,
        spectral_unit=data.spectral_unit,
    )
