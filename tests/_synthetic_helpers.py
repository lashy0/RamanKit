from __future__ import annotations

import numpy as np
from scipy.special import voigt_profile  # type: ignore[import-untyped]


def gaussian(
    axis: np.ndarray,
    *,
    amplitude: float,
    center: float,
    width: float,
    offset: float = 0.0,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((axis - center) / width) ** 2) + offset


def lorentzian(
    axis: np.ndarray,
    *,
    amplitude: float,
    center: float,
    width: float,
    offset: float = 0.0,
) -> np.ndarray:
    return amplitude / (1.0 + ((axis - center) / width) ** 2) + offset


def voigt(
    axis: np.ndarray,
    *,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    offset: float = 0.0,
) -> np.ndarray:
    return amplitude * voigt_profile(axis - center, sigma, gamma) + offset
