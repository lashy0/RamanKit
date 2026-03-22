from __future__ import annotations

import numpy as np
import pytest

import ramankit.analysis as ra
from ramankit import Metadata, RamanImage, Spectrum, SpectrumCollection
from ramankit.analysis._results import PCAResult
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


def test_pca_collection_returns_correct_types_and_shapes() -> None:
    """Decompose a collection and return components, scores, and variance ratio with correct shapes."""

    collection = _make_collection(n_spectra=10, n_points=50)
    result = ra.pca(collection, n_components=3)

    assert isinstance(result, PCAResult)
    assert isinstance(result.components, SpectrumCollection)
    assert result.components.n_spectra == 3
    assert result.components.n_points == 50
    assert np.array_equal(result.components.axis, collection.axis)
    assert result.components.spectral_axis_name == "raman_shift"
    assert result.components.spectral_unit == "cm^-1"
    assert result.scores.shape == (10, 3)
    assert result.explained_variance_ratio.shape == (3,)
    assert result.n_components == 3
    assert result.input_shape == (10,)


def test_pca_image_returns_spatial_scores() -> None:
    """Reshape scores to match the spatial dimensions of a RamanImage input."""

    image = _make_image(height=4, width=5, n_points=50)
    result = ra.pca(image, n_components=2)

    assert result.scores.shape == (4, 5, 2)
    assert result.input_shape == (4, 5)
    assert result.components.n_spectra == 2


def test_pca_explained_variance_sums_to_at_most_one() -> None:
    """Keep the sum of explained variance ratios within [0, 1]."""

    collection = _make_collection()
    result = ra.pca(collection, n_components=3)

    assert float(np.sum(result.explained_variance_ratio)) <= 1.0 + 1e-10


def test_pca_provenance_on_components() -> None:
    """Record a provenance step with method name and parameters on the component collection."""

    collection = _make_collection()
    result = ra.pca(collection, n_components=2)

    steps = result.components.provenance.steps
    assert len(steps) == 1
    assert steps[0].name == "pca"
    assert steps[0].parameters["n_components"] == 2


def test_pca_components_have_fresh_metadata() -> None:
    """Assign fresh default metadata to the component collection rather than copying input metadata."""

    collection = _make_collection()
    result = ra.pca(collection, n_components=2)

    assert result.components.metadata == Metadata()


def test_pca_single_component() -> None:
    """Decompose into one component and return scores of shape (n_spectra, 1)."""

    collection = _make_collection(n_spectra=5)
    result = ra.pca(collection, n_components=1)

    assert result.components.n_spectra == 1
    assert result.scores.shape == (5, 1)
    assert result.explained_variance_ratio.shape == (1,)


def test_pca_n_components_equals_n_spectra() -> None:
    """Allow n_components equal to the number of spectra as a valid full decomposition."""

    collection = _make_collection(n_spectra=5)
    result = ra.pca(collection, n_components=5)

    assert result.components.n_spectra == 5
    assert result.scores.shape == (5, 5)


def test_pca_raises_for_zero_components() -> None:
    """Reject n_components=0 as a non-positive value."""

    collection = _make_collection()
    with pytest.raises(ValueError, match="positive integer"):
        ra.pca(collection, n_components=0)


def test_pca_raises_for_too_many_components() -> None:
    """Reject n_components greater than the number of spectra."""

    collection = _make_collection(n_spectra=5)
    with pytest.raises(ValueError, match="must not exceed"):
        ra.pca(collection, n_components=6)


def test_pca_raises_for_wrong_input_type() -> None:
    """Reject a single Spectrum as input since decomposition requires multiple spectra."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Expected SpectrumCollection or RamanImage"):
        ra.pca(spectrum, n_components=1)  # type: ignore[arg-type]
