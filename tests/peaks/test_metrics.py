from __future__ import annotations

import numpy as np
import pytest

import ramankit.peaks.detect as rpd
import ramankit.peaks.metrics as rpm
from ramankit import Spectrum
from tests._synthetic_helpers import gaussian


def test_peak_metrics_return_detected_peak_properties() -> None:
    """Expose model-free metrics from one detected peak."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=5.0, center=180.0, width=5.0, offset=0.1)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5, width=1.0)[0]

    assert rpm.peak_position(peak) == pytest.approx(180.0, abs=0.5)
    assert rpm.peak_height(peak) > 5.0
    assert rpm.peak_prominence(peak) is not None
    assert rpm.peak_width(peak) is not None
