from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ramankit.core.spectrum import Spectrum
from ramankit.peaks.detect import DetectedPeak, PeakDetectionResult
from ramankit.peaks.fit import PeakFitResult
from ramankit.plotting._utils import apply_axis_labels, resolve_axes


def plot_detected_peaks(
    spectrum: Spectrum,
    detection_result: PeakDetectionResult,
    *,
    ax: Axes | None = None,
    spectrum_color: str | None = None,
    peak_marker: str = "o",
    peak_color: str = "crimson",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Intensity",
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one spectrum and overlay markers for detected peaks."""

    figure, axes = resolve_axes(ax)
    axes.plot(spectrum.axis, spectrum.intensity, color=spectrum_color)
    if len(detection_result) > 0:
        axes.scatter(
            detection_result.positions,
            detection_result.heights,
            color=peak_color,
            marker=peak_marker,
            zorder=3,
        )
    apply_axis_labels(axes, spectrum.spectral_axis_name, spectrum.spectral_unit, xlabel, ylabel)
    if title is not None:
        axes.set_title(title)
    if show:
        plt.show()
    return figure, axes


def plot_peak_fit(
    spectrum: Spectrum,
    detection_peak: DetectedPeak,
    fit_result: PeakFitResult,
    *,
    ax: Axes | None = None,
    spectrum_color: str | None = None,
    peak_color: str = "crimson",
    fit_color: str = "darkorange",
    window_color: str = "gold",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Intensity",
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one spectrum, a fitted peak curve, and the fit window."""

    window_lower = float(min(fit_result.window_axis))
    window_upper = float(max(fit_result.window_axis))
    if detection_peak.position < window_lower or detection_peak.position > window_upper:
        raise ValueError("Expected detection_peak to fall inside the fitted window range.")

    figure, axes = resolve_axes(ax)
    axes.plot(spectrum.axis, spectrum.intensity, color=spectrum_color)
    axes.axvspan(window_lower, window_upper, color=window_color, alpha=0.2)
    axes.scatter(
        [detection_peak.position],
        [detection_peak.height],
        color=peak_color,
        marker="o",
        zorder=3,
    )
    axes.plot(fit_result.window_axis, fit_result.fitted_intensity, color=fit_color)
    apply_axis_labels(axes, spectrum.spectral_axis_name, spectrum.spectral_unit, xlabel, ylabel)
    if title is not None:
        axes.set_title(title)
    if show:
        plt.show()
    return figure, axes
