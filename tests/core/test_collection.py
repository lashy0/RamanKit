from __future__ import annotations

import numpy as np
import pytest

from ramankit import Metadata, RamanImage, Spectrum, SpectrumCollection


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


def test_collection_n_spectra_and_n_points() -> None:
    """n_spectra and n_points reflect the intensity shape."""
    col = SpectrumCollection(axis=[1.0, 2.0, 3.0], intensity=[[1, 2, 3], [4, 5, 6]])
    assert col.n_spectra == 2
    assert col.n_points == 3


def test_collection_len() -> None:
    """len() returns the number of spectra."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1, 2], [3, 4], [5, 6]])
    assert len(col) == 3


def test_collection_getitem_slice_returns_collection() -> None:
    """Slice indexing returns a SpectrumCollection with the correct shape."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1, 2], [3, 4], [5, 6]])
    sub = col[0:2]
    assert isinstance(sub, SpectrumCollection)
    assert sub.n_spectra == 2
    assert np.array_equal(sub.intensity, [[1, 2], [3, 4]])


def test_collection_from_spectra_stacks_correctly() -> None:
    """from_spectra builds a collection with the right shape and axis."""
    axis = [100.0, 200.0, 300.0]
    s1 = Spectrum(axis=axis, intensity=[1.0, 2.0, 3.0])
    s2 = Spectrum(axis=axis, intensity=[4.0, 5.0, 6.0])
    s3 = Spectrum(axis=axis, intensity=[7.0, 8.0, 9.0])
    col = SpectrumCollection.from_spectra([s1, s2, s3])
    assert col.n_spectra == 3
    assert col.n_points == 3
    assert np.array_equal(col.axis, [100.0, 200.0, 300.0])


def test_collection_from_spectra_preserves_provenance() -> None:
    """from_spectra adds a stack_spectra provenance step."""
    s1 = Spectrum(axis=[1.0, 2.0], intensity=[1.0, 2.0])
    s2 = Spectrum(axis=[1.0, 2.0], intensity=[3.0, 4.0])
    col = SpectrumCollection.from_spectra([s1, s2])
    assert col.provenance.steps[-1].name == "stack_spectra"
    assert col.provenance.steps[-1].parameters["count"] == 2


def test_collection_copy_is_independent() -> None:
    """Mutating the original intensity does not affect the copy."""
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=raw)
    copied = col.copy()
    raw[0, 0] = -999.0
    assert copied.intensity[0, 0] == 1.0


def test_collection_sum_returns_spectrum() -> None:
    """sum reduces to the element-wise sum with provenance."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0], [3.0, 4.0]])
    result = col.sum()
    assert isinstance(result, Spectrum)
    np.testing.assert_allclose(result.intensity, [4.0, 6.0])
    assert result.provenance.steps[-1].name == "sum"


def test_collection_std_returns_spectrum() -> None:
    """std reduces to the standard deviation with provenance."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0], [3.0, 4.0]])
    result = col.std()
    assert isinstance(result, Spectrum)
    np.testing.assert_allclose(result.intensity, [1.0, 1.0])
    assert result.provenance.steps[-1].name == "std"


def test_collection_add_scalar() -> None:
    """Adding a scalar shifts all intensities."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0]])
    result = col.add(10.0)
    np.testing.assert_allclose(result.intensity, [[11.0, 12.0]])


def test_collection_subtract_collection() -> None:
    """Subtracting a matching collection gives element-wise difference."""
    a = SpectrumCollection(axis=[1.0, 2.0], intensity=[[5.0, 6.0]])
    b = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0]])
    result = a.subtract(b)
    np.testing.assert_allclose(result.intensity, [[4.0, 4.0]])


def test_collection_multiply_scalar() -> None:
    """Multiplying by a scalar scales all intensities."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0]])
    np.testing.assert_allclose(col.multiply(3.0).intensity, [[3.0, 6.0]])


def test_collection_divide_scalar() -> None:
    """Dividing by a scalar scales all intensities."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[4.0, 8.0]])
    np.testing.assert_allclose(col.divide(2.0).intensity, [[2.0, 4.0]])


def test_collection_dunder_operators() -> None:
    """Dunder operators +, -, *, / produce correct results."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[10.0, 20.0]])
    np.testing.assert_allclose((col + 1.0).intensity, [[11.0, 21.0]])
    np.testing.assert_allclose((col - 1.0).intensity, [[9.0, 19.0]])
    np.testing.assert_allclose((col * 2.0).intensity, [[20.0, 40.0]])
    np.testing.assert_allclose((col / 2.0).intensity, [[5.0, 10.0]])


def test_collection_arithmetic_preserves_metadata() -> None:
    """Arithmetic preserves metadata, spectral_axis_name, and spectral_unit."""
    col = SpectrumCollection(
        axis=[1.0, 2.0],
        intensity=[[1.0, 2.0]],
        metadata=Metadata(sample="s1"),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    result = col + 1.0
    assert result.metadata == col.metadata
    assert result.spectral_axis_name == "raman_shift"
    assert result.spectral_unit == "cm^-1"


def test_collection_descending_axis() -> None:
    """A descending axis is accepted and direction is recorded."""
    col = SpectrumCollection(axis=[300.0, 200.0, 100.0], intensity=[[1, 2, 3]])
    assert col.axis_direction == "descending"


def test_collection_default_metadata_and_provenance() -> None:
    """Default metadata and provenance are non-None when not provided."""
    col = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0]])
    assert col.metadata is not None
    assert col.provenance is not None
    assert col.metadata.sample is None
    assert col.provenance.steps == ()


def test_collection_add_preserves_descending_axis_direction() -> None:
    """Arithmetic rebuilds descending-axis collections without changing direction."""

    col = SpectrumCollection(axis=[3.0, 2.0, 1.0], intensity=[[1.0, 2.0, 3.0]])

    result = col.add(1.0)

    assert result.axis_direction == "descending"


def test_collection_add_raises_for_container_type_mismatch() -> None:
    """Shared arithmetic rejects different public container types."""

    collection = SpectrumCollection(axis=[1.0, 2.0], intensity=[[1.0, 2.0]])

    with pytest.raises(ValueError, match="same container type"):
        collection.add(RamanImage(axis=[1.0, 2.0], intensity=np.ones((1, 1, 2))))
