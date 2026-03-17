from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D, Array2D


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

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        denominator = np.linalg.norm(intensity, ord=2, axis=-1, keepdims=True)
        if np.any(np.isclose(denominator, 0.0)):
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

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        denominator = np.abs(np.trapezoid(intensity, axis, axis=-1)).reshape(-1, 1)
        if np.any(np.isclose(denominator, 0.0)):
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

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        denominator = np.max(intensity, axis=-1, keepdims=True)
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("Expected max normalization denominator to be non-zero.")
        return intensity / denominator


@dataclass(frozen=True, slots=True)
class MinMax(PreprocessingStep):
    """Min-max normalization step that maps each spectrum into the [0, 1] range."""

    function_name = "normalize"
    method_name = "minmax"

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        minimum = float(np.min(intensity))
        maximum = float(np.max(intensity))
        denominator = maximum - minimum
        if np.isclose(denominator, 0.0):
            raise ValueError("Expected min-max normalization denominator to be non-zero.")
        return (intensity - minimum) / denominator

    def _transform_batch(self, intensity: Array2D, axis: Array1D) -> Array2D | None:
        minimum = np.min(intensity, axis=-1, keepdims=True)
        maximum = np.max(intensity, axis=-1, keepdims=True)
        denominator = maximum - minimum
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("Expected min-max normalization denominator to be non-zero.")
        return (intensity - minimum) / denominator
