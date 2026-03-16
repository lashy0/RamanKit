"""Expose RamanKit peak analysis helpers."""

from ramankit.peaks import detect, fit, metrics
from ramankit.peaks.detect import DetectedPeak, PeakDetectionResult, find_peaks, find_peaks_batch
from ramankit.peaks.fit import (
    MultiPeakFitComponent,
    MultiPeakFitResult,
    PeakFitResult,
    fit_peak,
    fit_peaks,
)

__all__ = [
    "DetectedPeak",
    "MultiPeakFitComponent",
    "MultiPeakFitResult",
    "PeakDetectionResult",
    "PeakFitResult",
    "detect",
    "find_peaks",
    "find_peaks_batch",
    "fit",
    "fit_peak",
    "fit_peaks",
    "metrics",
]
