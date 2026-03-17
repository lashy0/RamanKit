from __future__ import annotations

import numpy as np

from ramankit.metrics._shared import (
    MetricInput,
    MetricResult,
    flatten_metric_input,
    reshape_metric_result,
    validate_pair_inputs,
)


def cosine_similarity(left: MetricInput, right: MetricInput) -> MetricResult:
    """Return cosine similarity for one spectrum or a per-item batch result."""

    validate_pair_inputs(left, right)
    left_values, output_shape = flatten_metric_input(left)
    right_values, _ = flatten_metric_input(right)

    numerator = np.sum(left_values * right_values, axis=-1)
    left_norm = np.linalg.norm(left_values, axis=-1)
    right_norm = np.linalg.norm(right_values, axis=-1)
    denominator = left_norm * right_norm
    result = np.where(np.isclose(denominator, 0.0), 0.0, numerator / denominator)
    return reshape_metric_result(np.asarray(result, dtype=np.float64), output_shape)


def pearson_correlation(left: MetricInput, right: MetricInput) -> MetricResult:
    """Return Pearson correlation for one spectrum or a per-item batch result."""

    validate_pair_inputs(left, right)
    left_values, output_shape = flatten_metric_input(left)
    right_values, _ = flatten_metric_input(right)

    centered_left = left_values - np.mean(left_values, axis=-1, keepdims=True)
    centered_right = right_values - np.mean(right_values, axis=-1, keepdims=True)
    numerator = np.sum(centered_left * centered_right, axis=-1)
    denominator = np.linalg.norm(centered_left, axis=-1) * np.linalg.norm(centered_right, axis=-1)
    result = np.where(np.isclose(denominator, 0.0), 0.0, numerator / denominator)
    return reshape_metric_result(np.asarray(result, dtype=np.float64), output_shape)


def mse(left: MetricInput, right: MetricInput) -> MetricResult:
    """Return mean squared error for one spectrum or a per-item batch result."""

    validate_pair_inputs(left, right)
    left_values, output_shape = flatten_metric_input(left)
    right_values, _ = flatten_metric_input(right)
    result = np.mean((left_values - right_values) ** 2, axis=-1)
    return reshape_metric_result(np.asarray(result, dtype=np.float64), output_shape)
