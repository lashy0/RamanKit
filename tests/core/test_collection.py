from __future__ import annotations

import numpy as np
import pytest

from ramankit import Metadata, Spectrum, SpectrumCollection


def test_collection_raises_for_axis_length_mismatch() -> None:
    """Reject collections whose shared axis length mismatches the last dimension."""

    with pytest.raises(ValueError, match="to match"):
        SpectrumCollection(
            axis=[100.0, 200.0],
            intensity=[[1.0, 2.0, 3.0]],
        )

def test_collection_from_spectra_raises_for_axis_mismatch() -> None:
    """Reject stacking spectra that do not share the same spectral axis."""

    left = Spectrum(axis=[100.0, 200.0], intensity=[1.0, 2.0], spectral_unit="cm^-1")
    right = Spectrum(axis=[100.0, 250.0], intensity=[1.0, 2.0], spectral_unit="cm^-1")

    with pytest.raises(ValueError, match="axes must match exactly"):
        SpectrumCollection.from_spectra([left, right])

def test_collection_mean_returns_spectrum() -> None:
    """Reduce a collection to a mean spectrum with provenance preserved."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    averaged = collection.mean()

    assert isinstance(averaged, Spectrum)
    assert np.allclose(averaged.intensity, [2.0, 3.0, 4.0])
    assert averaged.provenance.steps[-1].name == "mean"

def test_collection_slice_preserves_metadata() -> None:
    """Keep collection metadata when slicing out a single spectrum."""

    metadata = Metadata(sample="sample-2")
    collection = SpectrumCollection(
        axis=[100.0, 200.0],
        intensity=[[1.0, 2.0], [3.0, 4.0]],
        metadata=metadata,
    )

    sliced = collection[0]

    assert isinstance(sliced, Spectrum)
    assert sliced.metadata == metadata
