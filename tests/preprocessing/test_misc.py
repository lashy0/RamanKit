from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Metadata, Spectrum, SpectrumCollection


def test_cropper_apply_reduces_axis_to_requested_range() -> None:
    """Crop a spectrum to the requested inclusive spectral interval."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
    )

    cropped = pp.misc.Cropper(lower_bound=200.0, upper_bound=400.0).apply(spectrum)

    assert np.array_equal(cropped.axis, np.array([200.0, 300.0, 400.0]))
    assert np.array_equal(cropped.intensity, np.array([2.0, 3.0, 2.0]))
    assert cropped.provenance.steps[-1].name == "crop"

def test_cropper_apply_raises_for_out_of_range_bounds() -> None:
    """Reject crop bounds outside the source spectral range."""

    spectrum = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="crop bounds"):
        pp.misc.Cropper(lower_bound=50.0, upper_bound=250.0).apply(spectrum)

def test_background_subtractor_apply_subtracts_reference_spectrum() -> None:
    """Subtract a background spectrum while preserving the container type."""

    background = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([0.5, 0.5, 0.5]),
        metadata=Metadata(sample="background"),
    )
    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )

    subtracted = pp.misc.BackgroundSubtractor(background).apply(collection)

    assert isinstance(subtracted, SpectrumCollection)
    assert np.allclose(subtracted.intensity, np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]))
    assert subtracted.provenance.steps[-1].name == "background_subtract"
    assert subtracted.provenance.steps[-1].parameters["background_sample"] == "background"

def test_background_subtractor_apply_raises_for_axis_mismatch() -> None:
    """Reject background subtraction when spectral axes do not match."""

    background = Spectrum(axis=np.array([100.0, 250.0, 300.0]), intensity=np.array([0.5, 0.5, 0.5]))
    spectrum = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="axes must match"):
        pp.misc.BackgroundSubtractor(background).apply(spectrum)
