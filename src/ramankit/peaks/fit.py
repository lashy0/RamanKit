from __future__ import annotations

from collections.abc import Callable, Sequence
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


@dataclass(frozen=True, slots=True)
class MultiPeakFitComponent:
    """Store the fitted parameters and component curve for one peak contribution."""

    center: float
    amplitude: float
    width: float
    fitted_intensity: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MultiPeakFitResult:
    """Store the fitted parameters and curves for one multi-peak fit."""

    model: PeakModel
    offset: float
    components: tuple[MultiPeakFitComponent, ...]
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
    lower, upper = _normalize_window(window)
    _validate_peaks_in_window((peak,), lower, upper)

    window_axis, window_intensity = _extract_fit_window(
        spectrum,
        lower=lower,
        upper=upper,
        minimum_points=4,
    )

    offset_guess = float(np.min(window_intensity))
    amplitude_guess = float(np.max(window_intensity) - offset_guess)
    if amplitude_guess <= 0.0:
        raise ValueError(
            "Expected the fit window to contain a positive peak above the local offset."
        )

    span = float(max(upper - lower, np.finfo(np.float64).eps))
    width_guess = _resolve_width_guess(span=span, peak=peak, peak_count=1)
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


def fit_peaks(
    spectrum: Spectrum,
    peaks: Sequence[DetectedPeak],
    *,
    window: tuple[float, float],
    model: PeakModel = "gaussian",
) -> MultiPeakFitResult:
    """Fit multiple detected peaks simultaneously inside one spectral window."""

    sorted_peaks = tuple(sorted(peaks, key=lambda peak: peak.position))
    if len(sorted_peaks) < 2:
        raise ValueError("Expected at least 2 detected peaks for multi-peak fitting.")

    model_function = _resolve_model_function(model)
    lower, upper = _normalize_window(window)
    _validate_peaks_in_window(sorted_peaks, lower, upper)

    minimum_points = 3 * len(sorted_peaks) + 1
    window_axis, window_intensity = _extract_fit_window(
        spectrum,
        lower=lower,
        upper=upper,
        minimum_points=minimum_points,
    )

    offset_guess = float(np.min(window_intensity))
    span = float(max(upper - lower, np.finfo(np.float64).eps))

    initial_values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for peak in sorted_peaks:
        amplitude_guess = max(float(peak.height - offset_guess), np.finfo(np.float64).eps)
        width_guess = _resolve_width_guess(span=span, peak=peak, peak_count=len(sorted_peaks))
        initial_values.extend([amplitude_guess, float(peak.position), width_guess])
        lower_bounds.extend([0.0, lower, np.finfo(np.float64).eps])
        upper_bounds.extend([np.inf, upper, span])

    initial_values.append(offset_guess)
    lower_bounds.append(-np.inf)
    upper_bounds.append(np.inf)

    def multi_peak_model(x: npt.NDArray[np.float64], *parameters: float) -> npt.NDArray[np.float64]:
        offset = float(parameters[-1])
        total = np.full_like(x, offset, dtype=np.float64)
        for index in range(0, len(parameters) - 1, 3):
            amplitude = float(parameters[index])
            center = float(parameters[index + 1])
            width = float(parameters[index + 2])
            total += model_function(x, amplitude, center, width, 0.0)
        return total

    try:
        parameters, _ = curve_fit(
            multi_peak_model,
            window_axis,
            window_intensity,
            p0=np.asarray(initial_values, dtype=np.float64),
            bounds=(
                np.asarray(lower_bounds, dtype=np.float64),
                np.asarray(upper_bounds, dtype=np.float64),
            ),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Failed to fit {len(sorted_peaks)} {model} peaks in the requested window."
        ) from error

    offset = float(parameters[-1])
    components: list[MultiPeakFitComponent] = []
    for index in range(0, len(parameters) - 1, 3):
        amplitude = float(parameters[index])
        center = float(parameters[index + 1])
        width = float(parameters[index + 2])
        component_curve = model_function(window_axis, amplitude, center, width, 0.0)
        components.append(
            MultiPeakFitComponent(
                center=center,
                amplitude=amplitude,
                width=width,
                fitted_intensity=np.asarray(component_curve, dtype=np.float64),
            )
        )

    fitted_intensity = multi_peak_model(window_axis, *parameters)
    return MultiPeakFitResult(
        model=model,
        offset=offset,
        components=tuple(components),
        window_axis=window_axis.copy(),
        window_intensity=window_intensity.copy(),
        fitted_intensity=np.asarray(fitted_intensity, dtype=np.float64),
    )


def _normalize_window(window: tuple[float, float]) -> tuple[float, float]:
    lower, upper = sorted(window)
    if lower == upper:
        raise ValueError("Expected window bounds to define a non-empty spectral range.")
    return lower, upper


def _validate_peaks_in_window(peaks: Sequence[DetectedPeak], lower: float, upper: float) -> None:
    for peak in peaks:
        if peak.position < lower or peak.position > upper:
            raise ValueError("Expected every detected peak to lie inside the requested fit window.")


def _extract_fit_window(
    spectrum: Spectrum,
    *,
    lower: float,
    upper: float,
    minimum_points: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mask = (spectrum.axis >= lower) & (spectrum.axis <= upper)
    window_axis = spectrum.axis[mask]
    window_intensity = spectrum.intensity[mask]
    if window_axis.size < minimum_points:
        raise ValueError(
            f"Expected the fit window to contain at least {minimum_points} sampled points."
        )
    return window_axis, window_intensity


def _resolve_width_guess(*, span: float, peak: DetectedPeak, peak_count: int) -> float:
    if peak.width is not None and peak.width > 0.0:
        return float(peak.width)
    return span / max(4.0 * peak_count, 6.0)


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
