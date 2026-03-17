from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Spectrum, SpectrumCollection


def test_linear_resample_apply_returns_same_container_type() -> None:
    """Return a collection with the requested target axis after interpolation."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
    )

    resampled = pp.resample.Linear(target_axis=np.array([150.0, 250.0])).apply(collection)

    assert isinstance(resampled, SpectrumCollection)
    assert np.array_equal(resampled.axis, np.array([150.0, 250.0]))
    assert resampled.intensity.shape == (2, 2)
    assert np.allclose(resampled.intensity[0], np.array([1.5, 2.5]))
    assert resampled.provenance.steps[-1].name == "resample"

def test_linear_resample_apply_supports_descending_target_axis() -> None:
    """Support descending target axes without changing their direction."""

    spectrum = Spectrum(
        axis=np.array([300.0, 200.0, 100.0]),
        intensity=np.array([3.0, 2.0, 1.0]),
    )

    resampled = pp.resample.Linear(target_axis=np.array([250.0, 150.0])).apply(spectrum)

    assert np.array_equal(resampled.axis, np.array([250.0, 150.0]))
    assert np.allclose(resampled.intensity, np.array([2.5, 1.5]))

def test_linear_resample_apply_raises_for_out_of_range_target_axis() -> None:
    """Reject target axes that extend outside the source axis domain."""

    spectrum = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="target_axis"):
        pp.resample.Linear(target_axis=np.array([50.0, 150.0])).apply(spectrum)
