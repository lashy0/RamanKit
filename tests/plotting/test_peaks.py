from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
import ramankit.plotting.peaks as rpp
from ramankit import Spectrum
from tests._synthetic_helpers import gaussian


def test_plot_detected_peaks_returns_figure_and_peak_markers() -> None:
    """Overlay detected peak markers on top of one spectrum."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=5.0, center=160.0, width=6.0, offset=0.2)
    spectrum = Spectrum(
        axis=axis,
        intensity=intensity,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    detection_result = rpd.find_peaks(spectrum, prominence=0.5)

    figure, axes = rpp.plot_detected_peaks(spectrum, detection_result)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.lines) == 1
    assert len(axes.collections) == 1
    assert axes.collections[0].get_offsets().shape[0] == len(detection_result)
    plt.close(figure)


def test_plot_peak_fit_returns_full_spectrum_fit_and_window() -> None:
    """Render the full spectrum, the fit window, and the fitted peak curve."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=5.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(
        axis=axis,
        intensity=intensity,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    detection_peak = rpd.find_peaks(spectrum, prominence=0.5)[0]
    fit_result = rpf.fit_peak(spectrum, detection_peak, window=(160.0, 200.0), model="gaussian")

    figure, axes = rpp.plot_peak_fit(spectrum, detection_peak, fit_result)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.lines) == 2
    assert len(axes.collections) == 1
    assert len(axes.patches) == 1
    plt.close(figure)


def test_plot_peak_fit_raises_for_peak_outside_fit_window() -> None:
    """Reject fit overlays when the selected peak is outside the fitted window."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = (
        gaussian(axis, amplitude=5.0, center=150.0, width=5.0, offset=0.2)
        + gaussian(axis, amplitude=4.0, center=230.0, width=6.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)
    detection_result = rpd.find_peaks(spectrum, prominence=0.5)
    first_peak = detection_result[0]
    second_peak = detection_result[1]
    fit_result = rpf.fit_peak(spectrum, first_peak, window=(135.0, 165.0), model="gaussian")

    with pytest.raises(ValueError, match="fall inside the fitted window"):
        rpp.plot_peak_fit(spectrum, second_peak, fit_result)

