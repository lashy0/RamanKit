from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class Vector(PreprocessingStep):
    """Vector normalization step based on the L2 norm."""

    function_name = "normalize"
    method_name = "vector"

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        denominator = float(np.linalg.norm(intensity, ord=2))
        if np.isclose(denominator, 0.0):
            raise ValueError("Expected vector normalization denominator to be non-zero.")
        return intensity / denominator


@dataclass(frozen=True, slots=True)
class Area(PreprocessingStep):
    """Area normalization step based on the absolute spectral area."""

    function_name = "normalize"
    method_name = "area"

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        denominator = float(abs(np.trapezoid(intensity, axis)))
        if np.isclose(denominator, 0.0):
            raise ValueError("Expected area normalization denominator to be non-zero.")
        return intensity / denominator


@dataclass(frozen=True, slots=True)
class Max(PreprocessingStep):
    """Maximum-intensity normalization step."""

    function_name = "normalize"
    method_name = "max"

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        denominator = float(np.max(intensity))
        if np.isclose(denominator, 0.0):
            raise ValueError("Expected max normalization denominator to be non-zero.")
        return intensity / denominator
