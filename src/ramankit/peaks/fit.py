from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
from scipy.special import voigt_profile  # type: ignore[import-untyped]

from ramankit.core.spectrum import Spectrum
from ramankit.peaks.detect import DetectedPeak

PeakModel = Literal["gaussian", "lorentzian", "voigt"]
PeakFunction = Callable[..., npt.NDArray[np.float64]]
_VOIGT_SHARED_FWHM_FACTOR = 3.6013


@dataclass(frozen=True, slots=True)
class PeakFitResult:
    """Store the fitted parameters and fitted curve for one peak."""

    model: PeakModel
    center: float
    amplitude: float
    width: float | None
    sigma: float | None
    gamma: float | None
    offset: float
    window_axis: npt.NDArray[np.float64]
    window_intensity: npt.NDArray[np.float64]
    fitted_intensity: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MultiPeakFitComponent:
    """Store the fitted parameters and component curve for one peak contribution."""

    center: float
    amplitude: float
    width: float | None
    sigma: float | None
    gamma: float | None
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


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    model: PeakModel
    component_parameter_count: int
    peak_function: PeakFunction
    initial_component: Callable[[DetectedPeak, float, float, int], tuple[float, ...]]
    component_bounds: Callable[[float, float, float], tuple[tuple[float, ...], tuple[float, ...]]]


def fit_peak(
    spectrum: Spectrum,
    peak: DetectedPeak,
    *,
    window: tuple[float, float],
    model: PeakModel = "gaussian",
) -> PeakFitResult:
    """Fit one detected peak inside an explicit spectral-axis window."""

    spec = _resolve_model_spec(model)
    return _fit_single_peak_with_spec(spectrum, peak, window=window, spec=spec)


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

    spec = _resolve_model_spec(model)
    return _fit_multiple_peaks_with_spec(spectrum, sorted_peaks, window=window, spec=spec)


def _fit_single_peak_with_spec(
    spectrum: Spectrum,
    peak: DetectedPeak,
    *,
    window: tuple[float, float],
    spec: _ModelSpec,
) -> PeakFitResult:
    lower, upper = _normalize_window(window)
    _validate_peaks_in_window((peak,), lower, upper)

    minimum_points = spec.component_parameter_count + 1
    window_axis, window_intensity = _extract_fit_window(
        spectrum,
        lower=lower,
        upper=upper,
        minimum_points=minimum_points,
    )

    offset_guess = float(np.min(window_intensity))
    peak_height = float(np.max(window_intensity) - offset_guess)
    if peak_height <= 0.0:
        raise ValueError(
            "Expected the fit window to contain a positive peak above the local offset."
        )

    span = float(max(upper - lower, np.finfo(np.float64).eps))
    initial_component = spec.initial_component(peak, offset_guess, span, 1)
    lower_component_bounds, upper_component_bounds = spec.component_bounds(lower, upper, span)
    initial_guess = np.asarray([*initial_component, offset_guess], dtype=np.float64)
    lower_bounds = np.asarray([*lower_component_bounds, -np.inf], dtype=np.float64)
    upper_bounds = np.asarray([*upper_component_bounds, np.inf], dtype=np.float64)

    try:
        parameters, _ = curve_fit(
            spec.peak_function,
            window_axis,
            window_intensity,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=15000,
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Failed to fit a {spec.model} peak in the requested window."
        ) from error

    fitted_intensity = spec.peak_function(window_axis, *parameters)
    return _build_peak_fit_result(
        model=spec.model,
        parameters=np.asarray(parameters, dtype=np.float64),
        window_axis=window_axis,
        window_intensity=window_intensity,
        fitted_intensity=np.asarray(fitted_intensity, dtype=np.float64),
    )


def _fit_multiple_peaks_with_spec(
    spectrum: Spectrum,
    peaks: Sequence[DetectedPeak],
    *,
    window: tuple[float, float],
    spec: _ModelSpec,
) -> MultiPeakFitResult:
    lower, upper = _normalize_window(window)
    _validate_peaks_in_window(peaks, lower, upper)

    minimum_points = spec.component_parameter_count * len(peaks) + 1
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
    lower_component_bounds, upper_component_bounds = spec.component_bounds(lower, upper, span)
    for peak in peaks:
        initial_values.extend(spec.initial_component(peak, offset_guess, span, len(peaks)))
        lower_bounds.extend(lower_component_bounds)
        upper_bounds.extend(upper_component_bounds)

    initial_values.append(offset_guess)
    lower_bounds.append(-np.inf)
    upper_bounds.append(np.inf)

    def multi_peak_model(x: npt.NDArray[np.float64], *parameters: float) -> npt.NDArray[np.float64]:
        offset = float(parameters[-1])
        total = np.full_like(x, offset, dtype=np.float64)
        for index in range(0, len(parameters) - 1, spec.component_parameter_count):
            component_parameters = parameters[index : index + spec.component_parameter_count]
            total += _evaluate_component(spec, x, component_parameters, offset=0.0)
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
            maxfev=30000,
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Failed to fit {len(peaks)} {spec.model} peaks in the requested window."
        ) from error

    offset = float(parameters[-1])
    components: list[MultiPeakFitComponent] = []
    for index in range(0, len(parameters) - 1, spec.component_parameter_count):
        component_parameters = np.asarray(
            parameters[index : index + spec.component_parameter_count],
            dtype=np.float64,
        )
        component_curve = _evaluate_component(spec, window_axis, component_parameters, offset=0.0)
        components.append(
            _build_multi_peak_component(
                model=spec.model,
                component_parameters=component_parameters,
                fitted_intensity=np.asarray(component_curve, dtype=np.float64),
            )
        )

    fitted_intensity = multi_peak_model(window_axis, *parameters)
    return MultiPeakFitResult(
        model=spec.model,
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


def _resolve_model_spec(model: PeakModel) -> _ModelSpec:
    if model == "gaussian":
        return _ModelSpec(
            model="gaussian",
            component_parameter_count=3,
            peak_function=_gaussian,
            initial_component=_initial_standard_component,
            component_bounds=_standard_component_bounds,
        )
    if model == "lorentzian":
        return _ModelSpec(
            model="lorentzian",
            component_parameter_count=3,
            peak_function=_lorentzian,
            initial_component=_initial_standard_component,
            component_bounds=_standard_component_bounds,
        )
    if model == "voigt":
        return _ModelSpec(
            model="voigt",
            component_parameter_count=4,
            peak_function=_voigt,
            initial_component=_initial_voigt_component,
            component_bounds=_voigt_component_bounds,
        )
    raise ValueError(f"Unsupported peak model '{model}'.")


def _initial_standard_component(
    peak: DetectedPeak,
    offset_guess: float,
    span: float,
    peak_count: int,
) -> tuple[float, ...]:
    amplitude_guess = max(float(peak.height - offset_guess), np.finfo(np.float64).eps)
    width_guess = _resolve_width_guess(span=span, peak=peak, peak_count=peak_count)
    return amplitude_guess, float(peak.position), width_guess


def _initial_voigt_component(
    peak: DetectedPeak,
    offset_guess: float,
    span: float,
    peak_count: int,
) -> tuple[float, ...]:
    peak_height = max(float(peak.height - offset_guess), np.finfo(np.float64).eps)
    width_guess = _resolve_width_guess(span=span, peak=peak, peak_count=peak_count)
    sigma_guess = max(width_guess / _VOIGT_SHARED_FWHM_FACTOR, np.finfo(np.float64).eps)
    gamma_guess = max(width_guess / _VOIGT_SHARED_FWHM_FACTOR, np.finfo(np.float64).eps)
    amplitude_guess = _resolve_voigt_amplitude_guess(
        peak_height=peak_height,
        sigma=sigma_guess,
        gamma=gamma_guess,
    )
    return amplitude_guess, float(peak.position), sigma_guess, gamma_guess


def _standard_component_bounds(
    lower: float,
    upper: float,
    span: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (0.0, lower, np.finfo(np.float64).eps),
        (np.inf, upper, span),
    )


def _voigt_component_bounds(
    lower: float,
    upper: float,
    span: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (0.0, lower, np.finfo(np.float64).eps, np.finfo(np.float64).eps),
        (np.inf, upper, span, span),
    )


def _evaluate_component(
    spec: _ModelSpec,
    axis: npt.NDArray[np.float64],
    component_parameters: Sequence[float] | npt.NDArray[np.float64],
    *,
    offset: float,
) -> npt.NDArray[np.float64]:
    return np.asarray(spec.peak_function(axis, *component_parameters, offset), dtype=np.float64)


def _build_peak_fit_result(
    *,
    model: PeakModel,
    parameters: npt.NDArray[np.float64],
    window_axis: npt.NDArray[np.float64],
    window_intensity: npt.NDArray[np.float64],
    fitted_intensity: npt.NDArray[np.float64],
) -> PeakFitResult:
    width, sigma, gamma = _resolve_shape_parameters(model, parameters[:-1])
    return PeakFitResult(
        model=model,
        center=float(parameters[1]),
        amplitude=float(parameters[0]),
        width=width,
        sigma=sigma,
        gamma=gamma,
        offset=float(parameters[-1]),
        window_axis=window_axis.copy(),
        window_intensity=window_intensity.copy(),
        fitted_intensity=np.asarray(fitted_intensity, dtype=np.float64),
    )


def _build_multi_peak_component(
    *,
    model: PeakModel,
    component_parameters: npt.NDArray[np.float64],
    fitted_intensity: npt.NDArray[np.float64],
) -> MultiPeakFitComponent:
    width, sigma, gamma = _resolve_shape_parameters(model, component_parameters)
    return MultiPeakFitComponent(
        center=float(component_parameters[1]),
        amplitude=float(component_parameters[0]),
        width=width,
        sigma=sigma,
        gamma=gamma,
        fitted_intensity=np.asarray(fitted_intensity, dtype=np.float64),
    )


def _resolve_shape_parameters(
    model: PeakModel,
    component_parameters: Sequence[float] | npt.NDArray[np.float64],
) -> tuple[float | None, float | None, float | None]:
    if model == "voigt":
        return None, float(component_parameters[2]), float(component_parameters[3])
    return float(component_parameters[2]), None, None


def _resolve_voigt_amplitude_guess(*, peak_height: float, sigma: float, gamma: float) -> float:
    peak_scale = float(_voigt(np.asarray([0.0], dtype=np.float64), 1.0, 0.0, sigma, gamma, 0.0)[0])
    if peak_scale <= 0.0:
        return peak_height
    return peak_height / peak_scale


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


def _voigt(
    x: npt.NDArray[np.float64],
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return amplitude * voigt_profile(x - center, sigma, gamma) + offset
