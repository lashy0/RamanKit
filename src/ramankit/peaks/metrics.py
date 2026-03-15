from __future__ import annotations

from ramankit.peaks.detect import DetectedPeak


def peak_position(peak: DetectedPeak) -> float:
    """Return the spectral-axis position of one detected peak."""

    return peak.position


def peak_height(peak: DetectedPeak) -> float:
    """Return the sampled height of one detected peak."""

    return peak.height


def peak_prominence(peak: DetectedPeak) -> float | None:
    """Return the prominence of one detected peak when available."""

    return peak.prominence


def peak_width(peak: DetectedPeak) -> float | None:
    """Return the width of one detected peak when available."""

    return peak.width
