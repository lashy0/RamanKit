from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit  # type: ignore[import-untyped]

from ramankit.core.spectrum import Spectrum
from ramankit.peaks.detect import DetectedPeak

PeakModel = Literal["gaussian", "lorentzian"]
PeakModelFunction = Callable[
    [npt.NDArray[np.float64], float, float, float, float],
    npt.NDArray[np.float64],
]


@dataclass(frozen=True, slots=True)
class PeakFitResult:
    """Store the fitted parameters and fitted curve for one peak."""

    model: PeakModel
    center: float
    amplitude: float
    width: float
    offset: float
    window_axis: npt.NDArray[np.float64]
    window_intensity: npt.NDArray[np.float64]
    fitted_intensity: npt.NDArray[np.float64]


def fit_peak(
    spectrum: Spectrum,
    peak: DetectedPeak,
    *,
    window: tuple[float, float],
    model: PeakModel = "gaussian",
) -> PeakFitResult:
    """Fit one detected peak inside an explicit spectral-axis window."""

    model_function = _resolve_model_function(model)
    lower, upper = sorted(window)
    if lower == upper:
        raise ValueError("Expected window bounds to define a non-empty spectral range.")
    if peak.position < lower or peak.position > upper:
        raise ValueError("Expected the detected peak to lie inside the requested fit window.")

    mask = (spectrum.axis >= lower) & (spectrum.axis <= upper)
    window_axis = spectrum.axis[mask]
    window_intensity = spectrum.intensity[mask]
    if window_axis.size < 4:
        raise ValueError("Expected the fit window to contain at least 4 sampled points.")

    offset_guess = float(np.min(window_intensity))
    amplitude_guess = float(np.max(window_intensity) - offset_guess)
    if amplitude_guess <= 0.0:
        raise ValueError(
            "Expected the fit window to contain a positive peak above the local offset."
        )

    span = float(max(upper - lower, np.finfo(np.float64).eps))
    width_guess = span / 6.0
    initial_guess = np.asarray(
        [amplitude_guess, float(peak.position), width_guess, offset_guess],
        dtype=np.float64,
    )
    lower_bounds = np.asarray([0.0, lower, np.finfo(np.float64).eps, -np.inf], dtype=np.float64)
    upper_bounds = np.asarray([np.inf, upper, span, np.inf], dtype=np.float64)

    try:
        parameters, _ = curve_fit(
            model_function,
            window_axis,
            window_intensity,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Failed to fit a {model} peak in the requested window.") from error

    fitted_intensity = model_function(window_axis, *parameters)
    return PeakFitResult(
        model=model,
        center=float(parameters[1]),
        amplitude=float(parameters[0]),
        width=float(parameters[2]),
        offset=float(parameters[3]),
        window_axis=window_axis.copy(),
        window_intensity=window_intensity.copy(),
        fitted_intensity=np.asarray(fitted_intensity, dtype=np.float64),
    )


def _resolve_model_function(model: PeakModel) -> PeakModelFunction:
    if model == "gaussian":
        return _gaussian
    if model == "lorentzian":
        return _lorentzian
    raise ValueError(f"Unsupported peak model '{model}'.")


def _gaussian(
    x: npt.NDArray[np.float64],
    amplitude: float,
    center: float,
    width: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2) + offset


def _lorentzian(
    x: npt.NDArray[np.float64],
    amplitude: float,
    center: float,
    width: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return amplitude / (1.0 + ((x - center) / width) ** 2) + offset
