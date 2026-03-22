from __future__ import annotations

import numpy as np
import pytest

import ramankit.analysis as ra
from ramankit import Metadata, RamanImage, Spectrum, SpectrumCollection
from ramankit.analysis._results import NMFResult
from tests._synthetic_helpers import gaussian


def _make_collection(
    n_spectra: int = 10,
    n_points: int = 50,
    n_true: int = 3,
    seed: int = 42,
) -> SpectrumCollection:
    rng = np.random.default_rng(seed)
    axis = np.linspace(100.0, 500.0, n_points)
    centers = np.linspace(150.0, 450.0, n_true)
    components = np.vstack(
        [gaussian(axis, amplitude=1.0, center=c, width=20.0) for c in centers]
    )
    weights = rng.uniform(0.1, 1.0, size=(n_spectra, n_true))
    intensity = weights @ components + 0.01
    return SpectrumCollection(
        axis=axis,
        intensity=intensity,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )


def _make_image(
    height: int = 4,
    width: int = 5,
    n_points: int = 50,
    n_true: int = 3,
    seed: int = 42,
) -> RamanImage:
    rng = np.random.default_rng(seed)
    axis = np.linspace(100.0, 500.0, n_points)
    centers = np.linspace(150.0, 450.0, n_true)
    components = np.vstack(
        [gaussian(axis, amplitude=1.0, center=c, width=20.0) for c in centers]
    )
    weights = rng.uniform(0.1, 1.0, size=(height * width, n_true))
    intensity = (weights @ components + 0.01).reshape(height, width, n_points)
    return RamanImage(
        axis=axis,
        intensity=intensity,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )


def test_nmf_collection_returns_correct_types_and_shapes() -> None:
    """Decompose a collection and return components, scores, and reconstruction error with correct shapes."""

    collection = _make_collection(n_spectra=10, n_points=50)
    result = ra.nmf(collection, n_components=3)

    assert isinstance(result, NMFResult)
    assert isinstance(result.components, SpectrumCollection)
    assert result.components.n_spectra == 3
    assert result.components.n_points == 50
    assert np.array_equal(result.components.axis, collection.axis)
    assert result.components.spectral_axis_name == "raman_shift"
    assert result.components.spectral_unit == "cm^-1"
    assert result.scores.shape == (10, 3)
    assert result.reconstruction_error >= 0.0
    assert result.n_components == 3
    assert result.input_shape == (10,)


def test_nmf_image_returns_spatial_scores() -> None:
    """Reshape scores to match the spatial dimensions of a RamanImage input."""

    image = _make_image(height=4, width=5, n_points=50)
    result = ra.nmf(image, n_components=2)

    assert result.scores.shape == (4, 5, 2)
    assert result.input_shape == (4, 5)
    assert result.components.n_spectra == 2


def test_nmf_provenance_records_all_parameters() -> None:
    """Record all explicit parameters in the provenance step on the component collection."""

    collection = _make_collection()
    result = ra.nmf(
        collection,
        n_components=2,
        init="nndsvd",
        max_iter=500,
        random_state=42,
    )

    steps = result.components.provenance.steps
    assert len(steps) == 1
    assert steps[0].name == "nmf"
    assert steps[0].parameters["n_components"] == 2
    assert steps[0].parameters["init"] == "nndsvd"
    assert steps[0].parameters["max_iter"] == 500
    assert steps[0].parameters["random_state"] == 42


def test_nmf_components_have_fresh_metadata() -> None:
    """Assign fresh default metadata to the component collection rather than copying input metadata."""

    collection = _make_collection()
    result = ra.nmf(collection, n_components=2)

    assert result.components.metadata == Metadata()


def test_nmf_raises_for_negative_values() -> None:
    """Reject input data containing negative intensity values."""

    axis = np.linspace(100.0, 500.0, 50)
    intensity = np.array([[1.0] * 50, [-1.0] + [1.0] * 49])
    collection = SpectrumCollection(axis=axis, intensity=intensity)

    with pytest.raises(ValueError, match="non-negative"):
        ra.nmf(collection, n_components=1)


def test_nmf_raises_for_zero_components() -> None:
    """Reject n_components=0 as a non-positive value."""

    collection = _make_collection()
    with pytest.raises(ValueError, match="positive integer"):
        ra.nmf(collection, n_components=0)


def test_nmf_raises_for_too_many_components() -> None:
    """Reject n_components greater than the number of spectra."""

    collection = _make_collection(n_spectra=5)
    with pytest.raises(ValueError, match="must not exceed"):
        ra.nmf(collection, n_components=6)


def test_nmf_raises_for_wrong_input_type() -> None:
    """Reject a single Spectrum as input since decomposition requires multiple spectra."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Expected SpectrumCollection or RamanImage"):
        ra.nmf(spectrum, n_components=1)  # type: ignore[arg-type]
