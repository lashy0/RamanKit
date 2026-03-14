"""Regression tests for the RamanKit core spectral models and methods."""

from __future__ import annotations

import numpy as np
import pytest

from ramankit import Metadata, RamanImage, Spectrum, SpectrumCollection


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


def test_raman_image_raises_for_axis_length_mismatch() -> None:
    """Reject images whose spectral axis length mismatches the cube depth."""

    with pytest.raises(ValueError, match="to match"):
        RamanImage(
            axis=[100.0, 200.0, 300.0],
            intensity=np.ones((2, 2, 2)),
        )


def test_raman_image_flatten_preserves_metadata_and_shape() -> None:
    """Flatten images without losing metadata or provenance continuity."""

    metadata = Metadata(sample="sample-1", extras={"laser_nm": 785})
    image = RamanImage(
        axis=[100.0, 200.0],
        intensity=np.arange(12.0).reshape(2, 3, 2),
        metadata=metadata,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    flattened = image.flatten()

    assert flattened.intensity.shape == (6, 2)
    assert flattened.metadata == metadata
    assert flattened.provenance.steps[-1].name == "flatten_image"


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
