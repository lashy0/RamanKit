from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ramankit.core._validation import coerce_axis
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import Provenance, ProvenanceStep
from ramankit.core.spectrum import Spectrum
from ramankit.preprocessing._types import (
    AxisTransform1D,
    AxisTransform2D,
    FloatArray,
    SpectralDataT,
    Transform1D,
    Transform2D,
)


def apply_spectral_transform(
    data: SpectralDataT,
    *,
    transform: Transform1D,
    batch_transform: Transform2D | None = None,
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
        transformed_batch = None
        if batch_transform is not None:
            transformed_batch = batch_transform(flattened, data.axis)

        if transformed_batch is None:
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
        else:
            intensity = _validate_batch_transform(
                transformed_batch,
                expected_shape=flattened.shape,
                function_name=function_name,
            ).reshape(data.intensity.shape)

    provenance = data.provenance.append(
        build_provenance_step(function_name=function_name, method=method, parameters=parameters)
    )
    return rebuild_like(data, intensity=intensity, provenance=provenance)


def apply_axis_transform(
    data: SpectralDataT,
    *,
    transform: AxisTransform1D,
    batch_transform: AxisTransform2D | None = None,
    function_name: str,
    method: str,
    parameters: Mapping[str, object],
) -> SpectralDataT:
    """Apply a 1D transform that returns a new axis and intensity array."""

    if isinstance(data, Spectrum):
        axis, intensity = _transform_with_axis(
            data.intensity,
            data.axis,
            transform=transform,
            function_name=function_name,
        )
    else:
        flattened = data.intensity.reshape(-1, data.n_points)
        transformed_batch = None
        if batch_transform is not None:
            transformed_batch = batch_transform(flattened, data.axis)

        if transformed_batch is None:
            transformed_rows = [
                _transform_with_axis(
                    row,
                    data.axis,
                    transform=transform,
                    function_name=function_name,
                )
                for row in flattened
            ]
            axis = _validate_shared_axis(
                [transformed_axis for transformed_axis, _ in transformed_rows],
                function_name=function_name,
            )
            intensity = np.stack(
                [transformed_intensity for _, transformed_intensity in transformed_rows],
                axis=0,
            ).reshape(*data.intensity.shape[:-1], axis.shape[0])
        else:
            axis, transformed_intensity = transformed_batch
            axis = _validate_transformed_axis(axis, function_name=function_name)
            intensity = _validate_batch_transform(
                transformed_intensity,
                expected_shape=(flattened.shape[0], axis.shape[0]),
                function_name=function_name,
            ).reshape(*data.intensity.shape[:-1], axis.shape[0])

    provenance = data.provenance.append(
        build_provenance_step(function_name=function_name, method=method, parameters=parameters)
    )
    return rebuild_like(data, axis=axis, intensity=intensity, provenance=provenance)


def rebuild_like(
    data: SpectralDataT,
    *,
    axis: FloatArray | None = None,
    intensity: FloatArray,
    provenance: Provenance,
) -> SpectralDataT:
    """Rebuild a spectral container with new intensity data."""

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


def ensure_supported_method(method: str, *, allowed: tuple[str, ...], label: str) -> None:
    """Validate that a preprocessing method is supported."""

    if method not in allowed:
        allowed_methods = ", ".join(allowed)
        raise ValueError(f"Unsupported {label} '{method}'. Allowed methods: {allowed_methods}.")


def build_provenance_step(
    *,
    function_name: str,
    method: str,
    parameters: Mapping[str, object],
) -> ProvenanceStep:
    """Return a structured provenance step for preprocessing."""

    return ProvenanceStep(
        name=function_name,
        parameters={"method": method, **_serialize_parameters(parameters)},
    )


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


def _transform_with_axis(
    intensity: FloatArray,
    axis: FloatArray,
    *,
    transform: AxisTransform1D,
    function_name: str,
) -> tuple[FloatArray, FloatArray]:
    transformed_axis, transformed_intensity = transform(intensity.copy(), axis)
    axis_array, _ = coerce_axis(transformed_axis)
    intensity_array = np.array(transformed_intensity, dtype=np.float64, copy=True)
    if intensity_array.ndim != 1:
        raise ValueError(f"{function_name} must return 1D intensity data.")
    if axis_array.shape != intensity_array.shape:
        raise ValueError(
            f"{function_name} must return matching axis and intensity shapes; "
            f"got {axis_array.shape} and {intensity_array.shape}."
        )
    if not np.all(np.isfinite(intensity_array)):
        raise ValueError(f"{function_name} must return only finite intensity values.")
    return axis_array, intensity_array


def _validate_shared_axis(
    axes: list[FloatArray],
    *,
    function_name: str,
) -> FloatArray:
    if not axes:
        raise ValueError(f"{function_name} requires at least one transformed spectrum.")
    reference = axes[0]
    for axis in axes[1:]:
        if not np.array_equal(axis, reference):
            raise ValueError(f"{function_name} must return the same axis for every spectrum.")
    return reference


def _validate_batch_transform(
    transformed: FloatArray,
    *,
    expected_shape: tuple[int, int],
    function_name: str,
) -> FloatArray:
    transformed_array = np.array(transformed, dtype=np.float64, copy=True)
    if transformed_array.ndim != 2:
        raise ValueError(f"{function_name} must return 2D intensity data for batch transforms.")
    if transformed_array.shape != expected_shape:
        raise ValueError(
            f"{function_name} must preserve the input intensity shape for batch transforms; "
            f"got {transformed_array.shape} instead of {expected_shape}."
        )
    if not np.all(np.isfinite(transformed_array)):
        raise ValueError(f"{function_name} must return only finite intensity values.")
    return transformed_array


def _validate_transformed_axis(axis: FloatArray, *, function_name: str) -> FloatArray:
    axis_array, _ = coerce_axis(axis)
    return axis_array


def _serialize_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in parameters.items():
        if isinstance(value, np.generic):
            serialized[key] = value.item()
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized
