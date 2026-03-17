from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import simpson  # type: ignore[import-untyped]

from ramankit.metrics._shared import (
    MetricInput,
    MetricResult,
    flatten_metric_input,
    reshape_metric_result,
    resolve_region_slice,
)

type AreaMethod = Literal["trapezoid", "simpson"]


def snr(
    data: MetricInput,
    noise_region: tuple[float, float],
    signal_region: tuple[float, float] | None = None,
) -> MetricResult:
    """Return signal-to-noise ratio for one spectrum or per-item batch results."""

    flattened, output_shape = flatten_metric_input(data)
    noise_slice = resolve_region_slice(data.axis, noise_region, label="noise_region")
    noise_values = flattened[:, noise_slice]
    noise_std = np.std(noise_values, axis=-1)

    if signal_region is None:
        signal = np.max(flattened, axis=-1)
    else:
        signal_slice = resolve_region_slice(data.axis, signal_region, label="signal_region")
        signal = np.max(flattened[:, signal_slice], axis=-1)

    result = np.divide(
        signal,
        noise_std,
        out=np.zeros_like(signal, dtype=np.float64),
        where=~np.isclose(noise_std, 0.0),
    )
    return reshape_metric_result(np.asarray(result, dtype=np.float64), output_shape)


def band_area(
    data: MetricInput,
    region: tuple[float, float],
    *,
    method: AreaMethod = "trapezoid",
) -> MetricResult:
    """Return absolute integrated band area for one spectrum or per-item batch results."""

    flattened, output_shape = flatten_metric_input(data)
    region_slice = resolve_region_slice(data.axis, region, label="region")
    segment_axis = data.axis[region_slice]
    segment_values = flattened[:, region_slice]

    if method == "trapezoid":
        area = np.trapezoid(segment_values, segment_axis, axis=-1)
    elif method == "simpson":
        area = simpson(y=segment_values, x=segment_axis, axis=-1)
    else:
        raise ValueError("Unsupported band area method. Allowed methods: trapezoid, simpson.")

    return reshape_metric_result(np.asarray(np.abs(area), dtype=np.float64), output_shape)

