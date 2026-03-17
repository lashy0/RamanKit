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


def test_peak_position_returns_exact_value() -> None:
    """peak_position returns the position field of a DetectedPeak."""
    peak = rpd.DetectedPeak(index=10, position=532.7, height=100.0)
    assert rpm.peak_position(peak) == 532.7


def test_peak_height_returns_exact_value() -> None:
    """peak_height returns the height field of a DetectedPeak."""
    peak = rpd.DetectedPeak(index=5, position=200.0, height=42.5)
    assert rpm.peak_height(peak) == 42.5


def test_peak_prominence_returns_value_when_present() -> None:
    """peak_prominence returns the prominence when set."""
    peak = rpd.DetectedPeak(index=5, position=200.0, height=10.0, prominence=7.3)
    assert rpm.peak_prominence(peak) == 7.3


def test_peak_prominence_returns_none_when_absent() -> None:
    """peak_prominence returns None when prominence was not computed."""
    peak = rpd.DetectedPeak(index=5, position=200.0, height=10.0)
    assert rpm.peak_prominence(peak) is None


def test_peak_width_returns_value_when_present() -> None:
    """peak_width returns the width when set."""
    peak = rpd.DetectedPeak(index=5, position=200.0, height=10.0, width=3.8)
    assert rpm.peak_width(peak) == 3.8


def test_peak_width_returns_none_when_absent() -> None:
    """peak_width returns None when width was not computed."""
    peak = rpd.DetectedPeak(index=5, position=200.0, height=10.0)
    assert rpm.peak_width(peak) is None


def test_peak_position_negative() -> None:
    """peak_position handles negative positions (anti-Stokes shifts)."""
    peak = rpd.DetectedPeak(index=0, position=-35.6, height=7.4)
    assert rpm.peak_position(peak) == -35.6


def test_peak_height_near_zero() -> None:
    """peak_height handles near-zero heights."""
    peak = rpd.DetectedPeak(index=0, position=100.0, height=1e-12)
    assert rpm.peak_height(peak) == 1e-12
