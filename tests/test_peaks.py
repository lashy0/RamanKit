from __future__ import annotations

import numpy as np
import pytest

import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
import ramankit.peaks.metrics as rpm
from ramankit import Spectrum


def _gaussian(
    axis: np.ndarray,
    *,
    amplitude: float,
    center: float,
    width: float,
    offset: float = 0.0,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((axis - center) / width) ** 2) + offset


def _lorentzian(
    axis: np.ndarray,
    *,
    amplitude: float,
    center: float,
    width: float,
    offset: float = 0.0,
) -> np.ndarray:
    return amplitude / (1.0 + ((axis - center) / width) ** 2) + offset


def test_find_peaks_returns_typed_result_with_positions() -> None:
    """Detect multiple peaks and expose them in spectral coordinates."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = (
        _gaussian(axis, amplitude=5.0, center=130.0, width=4.0, offset=0.2)
        + _gaussian(axis, amplitude=3.5, center=230.0, width=6.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)

    result = rpd.find_peaks(spectrum, prominence=0.5, width=1.0, distance=200.0)

    assert isinstance(result, rpd.PeakDetectionResult)
    assert len(result) == 2
    assert np.allclose(result.positions, np.array([130.0, 230.0]), atol=0.5)
    assert np.all(result.heights > 0.0)
    assert result[0].prominence is not None
    assert result[0].width is not None


def test_find_peaks_returns_empty_result_when_no_peaks_exist() -> None:
    """Return an empty typed result when no local maxima satisfy the filters."""

    spectrum = Spectrum(axis=np.linspace(100.0, 300.0, 201), intensity=np.zeros(201))

    result = rpd.find_peaks(spectrum, prominence=1.0)

    assert isinstance(result, rpd.PeakDetectionResult)
    assert len(result) == 0
    assert result.indices.size == 0


def test_peak_metrics_return_detected_peak_properties() -> None:
    """Expose model-free metrics from one detected peak."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = _gaussian(axis, amplitude=5.0, center=180.0, width=5.0, offset=0.1)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5, width=1.0)[0]

    assert rpm.peak_position(peak) == pytest.approx(180.0, abs=0.5)
    assert rpm.peak_height(peak) > 5.0
    assert rpm.peak_prominence(peak) is not None
    assert rpm.peak_width(peak) is not None


def test_fit_peak_recovers_gaussian_parameters() -> None:
    """Fit one Gaussian peak inside an explicit spectral window."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = _gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    result = rpf.fit_peak(spectrum, peak, window=(160.0, 200.0), model="gaussian")

    assert result.model == "gaussian"
    assert result.center == pytest.approx(180.0, abs=0.05)
    assert result.amplitude == pytest.approx(4.0, rel=1e-2)
    assert result.width == pytest.approx(7.0, rel=1e-2)
    assert result.offset == pytest.approx(0.3, rel=1e-2)
    assert result.window_axis.shape == result.fitted_intensity.shape


def test_fit_peak_recovers_lorentzian_parameters() -> None:
    """Fit one Lorentzian peak inside an explicit spectral window."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = _lorentzian(axis, amplitude=6.0, center=220.0, width=5.0, offset=0.4)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    result = rpf.fit_peak(spectrum, peak, window=(200.0, 240.0), model="lorentzian")

    assert result.model == "lorentzian"
    assert result.center == pytest.approx(220.0, abs=0.05)
    assert result.amplitude == pytest.approx(6.0, rel=1e-2)
    assert result.width == pytest.approx(5.0, rel=1e-2)
    assert result.offset == pytest.approx(0.4, rel=1e-2)


def test_fit_peak_raises_for_invalid_window() -> None:
    """Reject windows that exclude the peak or contain too few samples."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = _gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    with pytest.raises(ValueError, match="lie inside the requested fit window"):
        rpf.fit_peak(spectrum, peak, window=(100.0, 120.0))

    with pytest.raises(ValueError, match="at least 4 sampled points"):
        rpf.fit_peak(spectrum, peak, window=(179.9, 180.1))


def test_fit_peak_raises_for_unsupported_model() -> None:
    """Reject unsupported line-shape models in the fitting API."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = _gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    with pytest.raises(ValueError, match="Unsupported peak model"):
        rpf.fit_peak(spectrum, peak, window=(160.0, 200.0), model="voigt")
