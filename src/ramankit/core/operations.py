from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ramankit.core._nd import (
    build_spectrum_from,
    ensure_compatible_spectral_data,
    rebuild_like,
)
from ramankit.core._validation import NumericArray, validate_axis_compatibility
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import ProvenanceStep
from ramankit.core.spectrum import Spectrum

SpectralData = Spectrum | SpectrumCollection | RamanImage
Reducer = Callable[..., NumericArray]


def stack_spectra(spectra: list[Spectrum] | tuple[Spectrum, ...]) -> SpectrumCollection:
    """Stack spectra with identical spectral axes into one collection."""

    if not spectra:
        raise ValueError("stack_spectra requires at least one spectrum.")

    reference = spectra[0]
    for spectrum in spectra[1:]:
        validate_axis_compatibility(
            reference.axis,
            spectrum.axis,
            left_name=reference.spectral_axis_name,
            right_name=spectrum.spectral_axis_name,
            left_unit=reference.spectral_unit,
            right_unit=spectrum.spectral_unit,
        )

    intensity = np.stack([spectrum.intensity for spectrum in spectra], axis=0)
    provenance = reference.provenance.append(
        ProvenanceStep(name="stack_spectra", parameters={"count": len(spectra)})
    )
    return SpectrumCollection(
        axis=reference.axis,
        intensity=intensity,
        metadata=reference.metadata,
        provenance=provenance,
        spectral_axis_name=reference.spectral_axis_name,
        spectral_unit=reference.spectral_unit,
    )


def flatten_image(image: RamanImage) -> SpectrumCollection:
    """Flatten a Raman image into a spectrum collection."""

    height, width = image.spatial_shape
    flattened = image.intensity.reshape(height * width, image.n_points)
    provenance = image.provenance.append(
        ProvenanceStep(
            name="flatten_image",
            parameters={"height": height, "width": width},
        )
    )
    return SpectrumCollection(
        axis=image.axis,
        intensity=flattened,
        metadata=image.metadata,
        provenance=provenance,
        spectral_axis_name=image.spectral_axis_name,
        spectral_unit=image.spectral_unit,
    )


def add[T: (Spectrum, SpectrumCollection, RamanImage)](
    left: T,
    right: T | float | int,
) -> T:
    """Return the elementwise sum of spectral data and an operand."""

    return _binary_operation(left, right, operator=np.add, operation_name="add")


def subtract[T: (Spectrum, SpectrumCollection, RamanImage)](
    left: T,
    right: T | float | int,
) -> T:
    """Return the elementwise difference of spectral data and an operand."""

    return _binary_operation(left, right, operator=np.subtract, operation_name="subtract")


def multiply[T: (Spectrum, SpectrumCollection, RamanImage)](
    left: T,
    right: T | float | int,
) -> T:
    """Return the elementwise product of spectral data and an operand."""

    return _binary_operation(left, right, operator=np.multiply, operation_name="multiply")


def divide[T: (Spectrum, SpectrumCollection, RamanImage)](
    left: T,
    right: T | float | int,
) -> T:
    """Return the elementwise quotient of spectral data and an operand."""

    return _binary_operation(left, right, operator=np.divide, operation_name="divide")


def mean(data: SpectrumCollection | RamanImage) -> Spectrum:
    """Reduce a collection or image to its mean spectrum."""

    return _reduce_to_spectrum(data, reducer=np.mean, operation_name="mean")


def sum(data: SpectrumCollection | RamanImage) -> Spectrum:
    """Reduce a collection or image to its summed spectrum."""

    return _reduce_to_spectrum(data, reducer=np.sum, operation_name="sum")


def std(data: SpectrumCollection | RamanImage) -> Spectrum:
    """Reduce a collection or image to its standard-deviation spectrum."""

    return _reduce_to_spectrum(data, reducer=np.std, operation_name="std")


def _binary_operation[T: (Spectrum, SpectrumCollection, RamanImage)](
    left: T,
    right: T | float | int,
    *,
    operator: Callable[[NumericArray, NumericArray | float | int], NumericArray],
    operation_name: str,
) -> T:
    if isinstance(right, (Spectrum, SpectrumCollection, RamanImage)):
        ensure_compatible_spectral_data(left, right)
        right_values: NumericArray | float | int = right.intensity
        parameter: object = type(right).__name__
    else:
        right_values = right
        parameter = right

    intensity = operator(left.intensity, right_values)
    provenance = left.provenance.append(
        ProvenanceStep(name=operation_name, parameters={"operand": parameter})
    )
    return rebuild_like(left, intensity=intensity, provenance=provenance)


def _reduce_to_spectrum(
    data: SpectrumCollection | RamanImage,
    *,
    reducer: Reducer,
    operation_name: str,
) -> Spectrum:
    axes = tuple(range(data.intensity.ndim - 1))
    intensity = reducer(data.intensity, axis=axes)
    provenance = data.provenance.append(
        ProvenanceStep(name=operation_name, parameters={"source": type(data).__name__})
    )
    return build_spectrum_from(data, intensity=intensity, provenance=provenance)
