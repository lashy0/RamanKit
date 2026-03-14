from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

AxisDirection = Literal["ascending", "descending"]
NumericArray = npt.NDArray[np.float64]


def coerce_axis(values: npt.ArrayLike) -> tuple[NumericArray, AxisDirection]:
    axis = np.array(values, dtype=np.float64, copy=True)
    if axis.ndim != 1:
        raise ValueError(f"Expected spectral axis to be 1D; got ndim={axis.ndim}.")
    if axis.size == 0:
        raise ValueError("Spectral axis must not be empty.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("Spectral axis must contain only finite values.")

    differences = np.diff(axis)
    if differences.size == 0 or np.all(differences > 0):
        return axis, "ascending"
    if np.all(differences < 0):
        return axis, "descending"

    if np.any(differences == 0):
        raise ValueError("Spectral axis must not contain duplicate values.")
    raise ValueError("Spectral axis must be strictly monotonic.")


def coerce_intensity(values: npt.ArrayLike, *, ndim: int, label: str) -> NumericArray:
    intensity = np.array(values, dtype=np.float64, copy=True)
    if intensity.ndim != ndim:
        raise ValueError(f"Expected {label} to be {ndim}D; got ndim={intensity.ndim}.")
    if intensity.size == 0:
        raise ValueError(f"{label} must not be empty.")
    if not np.issubdtype(intensity.dtype, np.number):
        raise ValueError(f"{label} must contain numeric values.")
    return intensity


def validate_axis_length(axis: NumericArray, last_dimension: int, *, label: str) -> None:
    if axis.shape[0] != last_dimension:
        raise ValueError(
            f"Expected spectral axis length {axis.shape[0]} to match "
            f"{label} last dimension {last_dimension}."
        )


def validate_axis_compatibility(
    left_axis: NumericArray,
    right_axis: NumericArray,
    *,
    left_name: str | None,
    right_name: str | None,
    left_unit: str | None,
    right_unit: str | None,
) -> None:
    if not np.array_equal(left_axis, right_axis):
        raise ValueError("Spectral axes must match exactly.")
    if left_name != right_name:
        raise ValueError("Spectral axis names must match exactly.")
    if left_unit != right_unit:
        raise ValueError("Spectral axis units must match exactly.")
