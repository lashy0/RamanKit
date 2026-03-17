from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ramankit.core._validation import validate_axis_compatibility
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.spectrum import Spectrum

type MetricInput = Spectrum | SpectrumCollection | RamanImage
type MetricResult = float | npt.NDArray[np.float64]
type FlatMetricData = tuple[npt.NDArray[np.float64], tuple[int, ...]]


def validate_pair_inputs(left: MetricInput, right: MetricInput) -> None:
    """Validate that two spectral containers can be compared pairwise."""

    if type(left) is not type(right):
        raise ValueError("Spectral metric operands must have the same container type.")
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
            "Expected spectral metric operand shapes to match; "
            f"got {left.intensity.shape} and {right.intensity.shape}."
        )


def flatten_metric_input(data: MetricInput) -> FlatMetricData:
    """Return spectral data flattened to 2D plus the target output shape."""

    if isinstance(data, Spectrum):
        return data.intensity.reshape(1, data.n_points), ()
    if isinstance(data, SpectrumCollection):
        return data.intensity, (data.n_spectra,)
    height, width = data.spatial_shape
    return data.intensity.reshape(height * width, data.n_points), (height, width)


def reshape_metric_result(values: npt.NDArray[np.float64], shape: tuple[int, ...]) -> MetricResult:
    """Reshape one metric result array back to the input container geometry."""

    if shape == ():
        return float(values[0])
    return values.reshape(shape)


def normalize_region(region: tuple[float, float], *, label: str) -> tuple[float, float]:
    """Return one inclusive region with validated finite bounds."""

    lower, upper = region
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Expected {label} bounds to be finite values.")
    if lower == upper:
        raise ValueError(f"Expected {label} bounds to span a non-zero interval.")
    return (lower, upper) if lower < upper else (upper, lower)


def resolve_region_slice(
    axis: npt.NDArray[np.float64],
    region: tuple[float, float],
    *,
    label: str,
) -> slice:
    """Return one slice spanning a spectral region on an ascending or descending axis."""

    lower, upper = normalize_region(region, label=label)
    ascending = axis[0] < axis[-1]
    if ascending:
        start = int(np.searchsorted(axis, lower, side="left"))
        end = int(np.searchsorted(axis, upper, side="right"))
    else:
        reversed_axis = axis[::-1]
        reversed_start = int(np.searchsorted(reversed_axis, lower, side="left"))
        reversed_end = int(np.searchsorted(reversed_axis, upper, side="right"))
        start = axis.shape[0] - reversed_end
        end = axis.shape[0] - reversed_start

    if start >= end:
        raise ValueError(f"No data points found in {label} {region}.")
    return slice(start, end)

