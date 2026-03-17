from __future__ import annotations

import numpy as np
import pytest

from ramankit import Spectrum


def test_spectrum_raises_for_axis_shape_mismatch() -> None:
    """Reject spectra whose axis and intensity shapes differ."""

    with pytest.raises(ValueError, match="same shape"):
        Spectrum(axis=[100.0, 200.0], intensity=[1.0])

def test_spectrum_accepts_descending_axis() -> None:
    """Preserve descending spectral axes when they remain monotonic."""

    spectrum = Spectrum(
        axis=[300.0, 200.0, 100.0],
        intensity=[1.0, 2.0, 3.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    assert spectrum.axis_direction == "descending"

def test_spectrum_raises_for_duplicate_axis_values() -> None:
    """Reject spectra whose spectral axis contains duplicate values."""

    with pytest.raises(ValueError, match="duplicate values"):
        Spectrum(axis=[100.0, 100.0, 200.0], intensity=[1.0, 2.0, 3.0])

def test_spectrum_copies_input_arrays_on_creation() -> None:
    """Copy user-provided arrays so later external mutation is harmless."""

    axis = np.array([100.0, 200.0, 300.0])
    intensity = np.array([1.0, 2.0, 3.0])

    spectrum = Spectrum(axis=axis, intensity=intensity)
    axis[0] = -1.0
    intensity[0] = -1.0

    assert spectrum.axis[0] == 100.0
    assert spectrum.intensity[0] == 1.0

def test_spectrum_add_raises_for_axis_metadata_mismatch() -> None:
    """Reject arithmetic between spectra with incompatible axis semantics."""

    left = Spectrum(
        axis=[100.0, 200.0],
        intensity=[1.0, 2.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    right = Spectrum(
        axis=[100.0, 200.0],
        intensity=[3.0, 4.0],
        spectral_axis_name="wavelength",
        spectral_unit="nm",
    )

    with pytest.raises(ValueError, match="axis names must match exactly"):
        left.add(right)
