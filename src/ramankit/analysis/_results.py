from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ramankit.core.collection import SpectrumCollection


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Result of principal component analysis on spectral data."""

    components: SpectrumCollection
    scores: npt.NDArray[np.float64]
    explained_variance_ratio: npt.NDArray[np.float64]
    n_components: int
    input_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NMFResult:
    """Result of non-negative matrix factorization on spectral data."""

    components: SpectrumCollection
    scores: npt.NDArray[np.float64]
    reconstruction_error: float
    n_components: int
    input_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ICAResult:
    """Result of independent component analysis on spectral data."""

    components: SpectrumCollection
    scores: npt.NDArray[np.float64]
    mixing_matrix: npt.NDArray[np.float64]
    n_components: int
    input_shape: tuple[int, ...]
