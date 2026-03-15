"""Expose RamanKit peak analysis helpers."""

from ramankit.peaks import detect, fit, metrics
from ramankit.peaks.detect import DetectedPeak, PeakDetectionResult, find_peaks
from ramankit.peaks.fit import PeakFitResult, fit_peak

__all__ = [
    "DetectedPeak",
    "PeakDetectionResult",
    "PeakFitResult",
    "detect",
    "find_peaks",
    "fit",
    "fit_peak",
    "metrics",
]
