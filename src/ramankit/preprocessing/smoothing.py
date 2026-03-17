from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pybaselines import Baseline  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D, Array2D


@dataclass(frozen=True, slots=True)
class SavGol(PreprocessingStep):
    """Savitzky-Golay smoothing step."""

    function_name = "smooth"
    method_name = "savitzky_golay"

    window_length: int = 9
    polyorder: int = 3
    deriv: int = 0
    delta: float = 1.0
    mode: str = "interp"

    def parameters(self) -> dict[str, object]:
        return {
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "deriv": self.deriv,
            "delta": self.delta,
            "mode": self.mode,
        }

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        _validate_savgol_parameters(
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            n_points=intensity.shape[0],
        )
        delta_value = self.delta
        if self.deriv > 0 and axis.shape[0] > 1:
            delta_value = float(np.mean(np.abs(np.diff(axis))))
        return np.asarray(
            savgol_filter(
                intensity,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                delta=delta_value,
                mode=self.mode,
            ),
            dtype=np.float64,
        )

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        _validate_savgol_parameters(
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            n_points=intensity.shape[-1],
        )
        delta_value = self.delta
        if self.deriv > 0 and axis.shape[0] > 1:
            delta_value = float(np.mean(np.abs(np.diff(axis))))
        return np.asarray(
            savgol_filter(
                intensity,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                delta=delta_value,
                mode=self.mode,
                axis=-1,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class Whittaker(PreprocessingStep):
    """Whittaker-like smoothing via symmetric ASLS weighting."""

    function_name = "smooth"
    method_name = "whittaker"

    lam: float = 1e3

    def __post_init__(self) -> None:
        """Validate the configured smoothing strength."""

        if self.lam <= 0:
            raise ValueError("Expected lam to be positive.")

    def parameters(self) -> dict[str, object]:
        return {"lam": self.lam}

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        return _whittaker_smooth(intensity, axis, lam=self.lam)


@dataclass(frozen=True, slots=True)
class Gaussian(PreprocessingStep):
    """Gaussian smoothing step."""

    function_name = "smooth"
    method_name = "gaussian"

    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate the configured Gaussian width."""

        if self.sigma <= 0:
            raise ValueError("Expected sigma to be positive.")

    def parameters(self) -> dict[str, object]:
        return {"sigma": self.sigma}

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        return np.asarray(
            gaussian_filter1d(intensity, sigma=self.sigma, mode="nearest"),
            dtype=np.float64,
        )

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        return np.asarray(
            gaussian_filter1d(intensity, sigma=self.sigma, mode="nearest", axis=-1),
            dtype=np.float64,
        )


def _whittaker_smooth(intensity: Array1D, axis: Array1D, *, lam: float) -> Array1D:
    """Return a Whittaker-like smoother via symmetric ASLS weighting."""

    baseline_fitter = Baseline(x_data=axis)
    smoothed, _ = baseline_fitter.asls(intensity, lam=lam, p=0.5)
    return np.asarray(smoothed, dtype=np.float64)


def _validate_savgol_parameters(
    *,
    window_length: int,
    polyorder: int,
    deriv: int,
    n_points: int,
) -> None:
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError("Expected window_length to be a positive odd integer.")
    if window_length > n_points:
        raise ValueError(
            f"Expected window_length {window_length} to be less than or equal to {n_points}."
        )
    if polyorder < 0:
        raise ValueError("Expected polyorder to be non-negative.")
    if polyorder >= window_length:
        raise ValueError("Expected polyorder to be smaller than window_length.")
    if deriv < 0:
        raise ValueError("Expected deriv to be non-negative.")
