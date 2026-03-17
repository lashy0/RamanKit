from __future__ import annotations

import numpy as np
import pytest

import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
from ramankit import Spectrum
from tests._synthetic_helpers import gaussian, lorentzian, voigt


def test_fit_peak_recovers_gaussian_parameters() -> None:
    """Fit one Gaussian peak inside an explicit spectral window."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    result = rpf.fit_peak(spectrum, peak, window=(160.0, 200.0), model="gaussian")

    assert result.model == "gaussian"
    assert result.center == pytest.approx(180.0, abs=0.05)
    assert result.amplitude == pytest.approx(4.0, rel=1e-2)
    assert result.width == pytest.approx(7.0, rel=1e-2)
    assert result.sigma is None
    assert result.gamma is None
    assert result.offset == pytest.approx(0.3, rel=1e-2)
    assert result.window_axis.shape == result.fitted_intensity.shape

def test_fit_peak_recovers_lorentzian_parameters() -> None:
    """Fit one Lorentzian peak inside an explicit spectral window."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = lorentzian(axis, amplitude=6.0, center=220.0, width=5.0, offset=0.4)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    result = rpf.fit_peak(spectrum, peak, window=(200.0, 240.0), model="lorentzian")

    assert result.model == "lorentzian"
    assert result.center == pytest.approx(220.0, abs=0.05)
    assert result.amplitude == pytest.approx(6.0, rel=1e-2)
    assert result.width == pytest.approx(5.0, rel=1e-2)
    assert result.sigma is None
    assert result.gamma is None
    assert result.offset == pytest.approx(0.4, rel=1e-2)

def test_fit_peak_recovers_voigt_parameters() -> None:
    """Fit one exact Voigt peak inside an explicit spectral window."""

    axis = np.linspace(100.0, 300.0, 2001)
    intensity = voigt(axis, amplitude=40.0, center=185.0, sigma=2.5, gamma=3.5, offset=0.25)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.1, width=1.0)[0]

    result = rpf.fit_peak(spectrum, peak, window=(165.0, 205.0), model="voigt")

    assert result.model == "voigt"
    assert result.center == pytest.approx(185.0, abs=0.08)
    assert result.amplitude == pytest.approx(40.0, rel=8e-2)
    assert result.width is None
    assert result.sigma == pytest.approx(2.5, rel=1.5e-1)
    assert result.gamma == pytest.approx(3.5, rel=1.5e-1)
    assert result.offset == pytest.approx(0.25, rel=5e-2)

def test_fit_peaks_recovers_gaussian_components() -> None:
    """Fit overlapping Gaussian peaks inside one shared spectral window."""

    axis = np.linspace(100.0, 300.0, 2001)
    intensity = (
        gaussian(axis, amplitude=5.0, center=160.0, width=4.5, offset=0.3)
        + gaussian(axis, amplitude=3.5, center=172.0, width=4.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peaks = tuple(rpd.find_peaks(spectrum, prominence=0.2, width=1.0))

    result = rpf.fit_peaks(spectrum, peaks, window=(148.0, 184.0), model="gaussian")

    assert result.model == "gaussian"
    assert len(result.components) == 2
    assert result.offset == pytest.approx(0.3, rel=5e-2)
    assert result.components[0].center == pytest.approx(160.0, abs=0.1)
    assert result.components[0].amplitude == pytest.approx(5.0, rel=5e-2)
    assert result.components[0].width == pytest.approx(4.5, rel=5e-2)
    assert result.components[0].sigma is None
    assert result.components[0].gamma is None
    assert result.components[1].center == pytest.approx(172.0, abs=0.1)
    assert result.components[1].amplitude == pytest.approx(3.5, rel=5e-2)
    assert result.components[1].width == pytest.approx(4.0, rel=5e-2)
    assert result.window_axis.shape == result.fitted_intensity.shape
    assert result.components[0].fitted_intensity.shape == result.window_axis.shape

def test_fit_peaks_recovers_lorentzian_components() -> None:
    """Fit overlapping Lorentzian peaks inside one shared spectral window."""

    axis = np.linspace(100.0, 300.0, 2001)
    intensity = (
        lorentzian(axis, amplitude=4.5, center=210.0, width=4.0, offset=0.25)
        + lorentzian(axis, amplitude=3.0, center=222.0, width=5.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peaks = tuple(rpd.find_peaks(spectrum, prominence=0.2, width=1.0))

    result = rpf.fit_peaks(spectrum, peaks, window=(198.0, 236.0), model="lorentzian")

    assert result.model == "lorentzian"
    assert len(result.components) == 2
    assert result.offset == pytest.approx(0.25, rel=5e-2)
    assert result.components[0].center == pytest.approx(210.0, abs=0.1)
    assert result.components[0].amplitude == pytest.approx(4.5, rel=5e-2)
    assert result.components[0].width == pytest.approx(4.0, rel=5e-2)
    assert result.components[0].sigma is None
    assert result.components[0].gamma is None
    assert result.components[1].center == pytest.approx(222.0, abs=0.1)
    assert result.components[1].amplitude == pytest.approx(3.0, rel=5e-2)
    assert result.components[1].width == pytest.approx(5.0, rel=5e-2)

def test_fit_peaks_recovers_voigt_components() -> None:
    """Fit overlapping Voigt peaks inside one shared spectral window."""

    axis = np.linspace(100.0, 300.0, 3001)
    intensity = (
        voigt(axis, amplitude=35.0, center=205.0, sigma=2.0, gamma=3.0, offset=0.2)
        + voigt(axis, amplitude=28.0, center=216.0, sigma=2.5, gamma=2.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peaks = tuple(rpd.find_peaks(spectrum, prominence=0.05, width=1.0, distance=80.0))

    result = rpf.fit_peaks(spectrum, peaks, window=(194.0, 226.0), model="voigt")

    assert result.model == "voigt"
    assert len(result.components) == 2
    assert result.offset == pytest.approx(0.2, rel=8e-2)
    assert result.components[0].center == pytest.approx(205.0, abs=0.15)
    assert result.components[0].amplitude == pytest.approx(35.0, rel=1.5e-1)
    assert result.components[0].width is None
    assert result.components[0].sigma == pytest.approx(2.0, rel=1.5e-1)
    assert result.components[0].gamma == pytest.approx(3.0, rel=1.5e-1)
    assert result.components[1].center == pytest.approx(216.0, abs=0.15)
    assert result.components[1].amplitude == pytest.approx(28.0, rel=1.5e-1)
    assert result.components[1].width is None
    assert result.components[1].sigma == pytest.approx(2.5, rel=1.5e-1)
    assert result.components[1].gamma == pytest.approx(2.0, rel=1.5e-1)

def test_fit_peak_raises_for_invalid_window() -> None:
    """Reject windows that exclude the peak or contain too few samples."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    with pytest.raises(ValueError, match="lie inside the requested fit window"):
        rpf.fit_peak(spectrum, peak, window=(100.0, 120.0))

    with pytest.raises(ValueError, match="at least 4 sampled points"):
        rpf.fit_peak(spectrum, peak, window=(179.9, 180.1))

def test_fit_peaks_raises_for_invalid_inputs() -> None:
    """Reject invalid multi-peak fitting inputs and unsupported models."""

    axis = np.linspace(100.0, 300.0, 2001)
    intensity = (
        gaussian(axis, amplitude=5.0, center=160.0, width=4.5, offset=0.3)
        + gaussian(axis, amplitude=3.5, center=172.0, width=4.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peaks = tuple(rpd.find_peaks(spectrum, prominence=0.2, width=1.0))

    with pytest.raises(ValueError, match="at least 2 detected peaks"):
        rpf.fit_peaks(spectrum, peaks[:1], window=(148.0, 184.0), model="gaussian")

    with pytest.raises(ValueError, match="lie inside the requested fit window"):
        rpf.fit_peaks(spectrum, peaks, window=(100.0, 150.0), model="gaussian")

    tiny_window_peaks = (
        rpd.DetectedPeak(index=600, position=160.0, height=5.0),
        rpd.DetectedPeak(index=601, position=160.1, height=4.5),
    )
    with pytest.raises(ValueError, match="at least 7 sampled points"):
        rpf.fit_peaks(spectrum, tiny_window_peaks, window=(159.95, 160.15), model="gaussian")

    with pytest.raises(ValueError, match="Unsupported peak model"):
        rpf.fit_peaks(spectrum, peaks, window=(148.0, 184.0), model="invalid")  # type: ignore[arg-type]

def test_fit_peak_raises_for_unsupported_model() -> None:
    """Reject unsupported line-shape models in the fitting API."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = gaussian(axis, amplitude=4.0, center=180.0, width=7.0, offset=0.3)
    spectrum = Spectrum(axis=axis, intensity=intensity)
    peak = rpd.find_peaks(spectrum, prominence=0.5)[0]

    with pytest.raises(ValueError, match="Unsupported peak model"):
        rpf.fit_peak(spectrum, peak, window=(160.0, 200.0), model="invalid")  # type: ignore[arg-type]
