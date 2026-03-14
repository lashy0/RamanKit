from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import Provenance, ProvenanceStep
from ramankit.core.spectrum import Spectrum
from ramankit.preprocessing._types import FloatArray, SpectralDataT, Transform1D


def apply_spectral_transform(
    data: SpectralDataT,
    *,
    transform: Transform1D,
    function_name: str,
    method: str,
    parameters: Mapping[str, object],
) -> SpectralDataT:
    """Apply a 1D transform along the spectral axis of a spectral container."""

    if isinstance(data, Spectrum):
        intensity = _transform_row(
            data.intensity,
            data.axis,
            transform=transform,
            function_name=function_name,
        )
    else:
        flattened = data.intensity.reshape(-1, data.n_points)
        intensity = np.stack(
            [
                _transform_row(
                    row,
                    data.axis,
                    transform=transform,
                    function_name=function_name,
                )
                for row in flattened
            ],
            axis=0,
        ).reshape(data.intensity.shape)

    provenance = data.provenance.append(
        ProvenanceStep(
            name=function_name,
            parameters={"method": method, **_serialize_parameters(parameters)},
        )
    )
    return rebuild_like(data, intensity=intensity, provenance=provenance)


def rebuild_like(
    data: SpectralDataT,
    *,
    intensity: FloatArray,
    provenance: Provenance,
) -> SpectralDataT:
    """Rebuild a spectral container with new intensity data."""

    if isinstance(data, Spectrum):
        return Spectrum(
            axis=data.axis,
            intensity=intensity,
            metadata=data.metadata,
            provenance=provenance,
            spectral_axis_name=data.spectral_axis_name,
            spectral_unit=data.spectral_unit,
        )
    if isinstance(data, SpectrumCollection):
        return SpectrumCollection(
            axis=data.axis,
            intensity=intensity,
            metadata=data.metadata,
            provenance=provenance,
            spectral_axis_name=data.spectral_axis_name,
            spectral_unit=data.spectral_unit,
        )
    return RamanImage(
        axis=data.axis,
        intensity=intensity,
        metadata=data.metadata,
        provenance=provenance,
        spectral_axis_name=data.spectral_axis_name,
        spectral_unit=data.spectral_unit,
    )


def ensure_supported_method(method: str, *, allowed: tuple[str, ...], label: str) -> None:
    """Validate that a preprocessing method is supported."""

    if method not in allowed:
        allowed_methods = ", ".join(allowed)
        raise ValueError(f"Unsupported {label} '{method}'. Allowed methods: {allowed_methods}.")


def _transform_row(
    intensity: FloatArray,
    axis: FloatArray,
    *,
    transform: Transform1D,
    function_name: str,
) -> FloatArray:
    transformed = np.array(transform(intensity.copy(), axis), dtype=np.float64, copy=True)
    if transformed.ndim != 1:
        raise ValueError(f"{function_name} must return 1D intensity data.")
    if transformed.shape != intensity.shape:
        raise ValueError(
            f"{function_name} must preserve the input intensity shape; "
            f"got {transformed.shape} instead of {intensity.shape}."
        )
    if not np.all(np.isfinite(transformed)):
        raise ValueError(f"{function_name} must return only finite intensity values.")
    return transformed


def _serialize_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in parameters.items():
        if isinstance(value, np.generic):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized
