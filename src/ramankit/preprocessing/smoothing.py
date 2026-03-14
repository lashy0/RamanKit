from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from ramankit.preprocessing._base import PreprocessingStep
from ramankit.preprocessing._types import Array1D


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
