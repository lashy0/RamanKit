from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import RamanImage, Spectrum


def test_vector_apply_scales_to_unit_norm() -> None:
    """Scale a spectrum to unit L2 norm."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[3.0, 4.0, 0.0])

    normalized = pp.normalization.Vector().apply(spectrum)

    assert np.isclose(np.linalg.norm(normalized.intensity), 1.0)
    assert normalized.provenance.steps[-1].parameters["method"] == "vector"

def test_area_apply_uses_spectral_axis_values() -> None:
    """Normalize by the absolute area under the spectrum."""

    axis = np.array([300.0, 200.0, 100.0])
    spectrum = Spectrum(axis=axis, intensity=[1.0, 1.0, 1.0])

    normalized = pp.normalization.Area().apply(spectrum)

    assert np.isclose(abs(np.trapezoid(normalized.intensity, normalized.axis)), 1.0)

def test_max_apply_preserves_raman_image_type() -> None:
    """Normalize image intensities by their per-spectrum maximum."""

    axis = np.linspace(100.0, 400.0, 4)
    image = RamanImage(
        axis=axis,
        intensity=np.array([[[1.0, 2.0, 4.0, 2.0], [2.0, 3.0, 6.0, 3.0]]]),
    )

    normalized = pp.normalization.Max().apply(image)

    assert isinstance(normalized, RamanImage)
    assert np.isclose(np.max(normalized.intensity[0, 0]), 1.0)
    assert np.isclose(np.max(normalized.intensity[0, 1]), 1.0)

def test_minmax_apply_scales_spectrum_to_unit_interval() -> None:
    """Scale a spectrum so its minimum is zero and its maximum is one."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[2.0, 5.0, 3.0])

    normalized = pp.normalization.MinMax().apply(spectrum)

    assert np.isclose(np.min(normalized.intensity), 0.0)
    assert np.isclose(np.max(normalized.intensity), 1.0)
    assert normalized.provenance.steps[-1].parameters["method"] == "minmax"

def test_minmax_apply_preserves_raman_image_type() -> None:
    """Normalize each image spectrum independently into the [0, 1] range."""

    axis = np.linspace(100.0, 400.0, 4)
    image = RamanImage(
        axis=axis,
        intensity=np.array([[[1.0, 3.0, 5.0, 2.0], [4.0, 6.0, 8.0, 5.0]]]),
    )

    normalized = pp.normalization.MinMax().apply(image)

    assert isinstance(normalized, RamanImage)
    assert np.isclose(np.min(normalized.intensity[0, 0]), 0.0)
    assert np.isclose(np.max(normalized.intensity[0, 0]), 1.0)
    assert np.isclose(np.min(normalized.intensity[0, 1]), 0.0)
    assert np.isclose(np.max(normalized.intensity[0, 1]), 1.0)

def test_minmax_apply_raises_for_zero_denominator() -> None:
    """Reject min-max normalization for constant spectra."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="denominator"):
        pp.normalization.MinMax().apply(spectrum)

def test_vector_apply_raises_for_zero_denominator() -> None:
    """Reject normalization when the denominator is zero."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="denominator"):
        pp.normalization.Vector().apply(spectrum)
