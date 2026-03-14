from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class WhitakerHayes(PreprocessingStep):
    """Whitaker-Hayes despiking based on modified z-scores of first differences."""

    function_name = "despike"
    method_name = "whitaker_hayes"

    threshold: float = 8.0
    kernel_size: int = 3
    max_iter: int = 10

    def parameters(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "kernel_size": self.kernel_size,
            "max_iter": self.max_iter,
        }

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        _validate_whitaker_hayes_parameters(
            threshold=self.threshold,
            kernel_size=self.kernel_size,
            max_iter=self.max_iter,
            n_points=intensity.shape[0],
        )

        corrected = intensity.copy()
        radius = self.kernel_size // 2

        for _ in range(self.max_iter):
            spike_mask = _detect_spikes(corrected, threshold=self.threshold)
            if not np.any(spike_mask):
                break

            updated = corrected.copy()
            for raw_index in np.flatnonzero(spike_mask):
                index = int(raw_index)
                start = max(0, index - radius)
                stop = min(corrected.shape[0], index + radius + 1)
                local_values = corrected[start:stop]
                local_mask = spike_mask[start:stop]
                clean_values = local_values[~local_mask]
                replacement = np.median(clean_values if clean_values.size > 0 else local_values)
                updated[index] = float(replacement)
            corrected = updated

        return corrected


def _detect_spikes(intensity: Array1D, *, threshold: float) -> np.ndarray:
    differences = np.diff(intensity)
    if differences.size == 0:
        return np.zeros_like(intensity, dtype=bool)

    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    if np.isclose(mad, 0.0):
        return np.zeros_like(intensity, dtype=bool)

    modified_z_score = 0.6745 * (differences - median) / mad
    jump_mask = np.abs(modified_z_score) > threshold

    point_mask = np.zeros_like(intensity, dtype=bool)
    point_mask[1:] |= jump_mask
    point_mask[:-1] |= jump_mask
    return point_mask


def _validate_whitaker_hayes_parameters(
    *,
    threshold: float,
    kernel_size: int,
    max_iter: int,
    n_points: int,
) -> None:
    if threshold <= 0:
        raise ValueError("Expected threshold to be positive.")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Expected kernel_size to be an odd integer greater than or equal to 3.")
    if kernel_size > n_points:
        raise ValueError(
            f"Expected kernel_size {kernel_size} to be less than or equal to {n_points}."
        )
    if max_iter <= 0:
        raise ValueError("Expected max_iter to be positive.")
