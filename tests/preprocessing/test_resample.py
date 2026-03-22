from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import RamanImage, Spectrum, SpectrumCollection


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


def test_linear_resample_apply_preserves_image_shape() -> None:
    """Axis-changing batch transforms rebuild RamanImage with its spatial shape intact."""

    image = RamanImage(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.arange(12.0).reshape(2, 2, 3),
    )

    resampled = pp.resample.Linear(target_axis=np.array([150.0, 250.0])).apply(image)

    assert isinstance(resampled, RamanImage)
    assert resampled.intensity.shape == (2, 2, 2)
    assert np.array_equal(resampled.axis, np.array([150.0, 250.0]))
    assert resampled.provenance.steps[-1].name == "resample"


def test_linear_resample_apply_preserves_descending_axis_direction() -> None:
    """Axis-changing rebuilds keep descending target-axis semantics."""

    collection = SpectrumCollection(
        axis=np.array([300.0, 200.0, 100.0]),
        intensity=np.array([[3.0, 2.0, 1.0], [6.0, 4.0, 2.0]]),
    )

    resampled = pp.resample.Linear(target_axis=np.array([250.0, 150.0])).apply(collection)

    assert isinstance(resampled, SpectrumCollection)
    assert resampled.axis_direction == "descending"
    assert resampled.intensity.shape == (2, 2)


# --- resample_to_common_axis tests ---


def test_resample_to_common_axis_aligns_two_spectra() -> None:
    """Resample two spectra with different axes onto a common grid."""

    s1 = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))
    s2 = Spectrum(axis=np.array([150.0, 250.0, 350.0]), intensity=np.array([1.5, 2.5, 3.5]))

    result = pp.resample.resample_to_common_axis([s1, s2])

    assert isinstance(result, SpectrumCollection)
    assert result.n_spectra == 2
    assert float(np.min(result.axis)) >= 150.0
    assert float(np.max(result.axis)) <= 300.0


def test_resample_to_common_axis_with_explicit_n_points() -> None:
    """Use an explicit n_points for the common grid."""

    s1 = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))
    s2 = Spectrum(axis=np.array([150.0, 250.0, 350.0]), intensity=np.array([1.5, 2.5, 3.5]))

    result = pp.resample.resample_to_common_axis([s1, s2], n_points=5)

    assert result.n_points == 5


def test_resample_to_common_axis_default_n_points_uses_median() -> None:
    """Default n_points is the median of input spectrum sizes."""

    s1 = Spectrum(axis=np.linspace(0, 10, 10), intensity=np.ones(10))
    s2 = Spectrum(axis=np.linspace(1, 9, 20), intensity=np.ones(20))

    result = pp.resample.resample_to_common_axis([s1, s2])

    assert result.n_points == int(np.median([10, 20]))


def test_resample_to_common_axis_raises_for_single_spectrum() -> None:
    """Reject fewer than two spectra."""

    s1 = Spectrum(axis=np.array([1.0, 2.0, 3.0]), intensity=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="at least two"):
        pp.resample.resample_to_common_axis([s1])


def test_resample_to_common_axis_raises_for_no_overlap() -> None:
    """Reject spectra with disjoint axis ranges."""

    s1 = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))
    s2 = Spectrum(axis=np.array([400.0, 500.0, 600.0]), intensity=np.array([4.0, 5.0, 6.0]))

    with pytest.raises(ValueError, match="overlapping"):
        pp.resample.resample_to_common_axis([s1, s2])


def test_resample_to_common_axis_raises_for_mismatched_units() -> None:
    """Reject spectra with different spectral units."""

    s1 = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectral_unit="cm^-1",
    )
    s2 = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectral_unit="nm",
    )

    with pytest.raises(ValueError, match="spectral_unit"):
        pp.resample.resample_to_common_axis([s1, s2])


def test_resample_to_common_axis_handles_descending_axes() -> None:
    """Handle spectra with descending axes correctly."""

    s1 = Spectrum(
        axis=np.array([300.0, 200.0, 100.0]),
        intensity=np.array([3.0, 2.0, 1.0]),
    )
    s2 = Spectrum(
        axis=np.array([350.0, 250.0, 150.0]),
        intensity=np.array([3.5, 2.5, 1.5]),
    )

    result = pp.resample.resample_to_common_axis([s1, s2], n_points=5)

    assert result.n_spectra == 2
    assert result.n_points == 5
    assert float(np.min(result.axis)) >= 150.0
    assert float(np.max(result.axis)) <= 300.0


def test_resample_to_common_axis_records_provenance() -> None:
    """Record a provenance step with resampling parameters."""

    s1 = Spectrum(axis=np.array([100.0, 200.0, 300.0]), intensity=np.array([1.0, 2.0, 3.0]))
    s2 = Spectrum(axis=np.array([150.0, 250.0, 350.0]), intensity=np.array([1.5, 2.5, 3.5]))

    result = pp.resample.resample_to_common_axis([s1, s2], n_points=5)

    step = result.provenance.steps[-1]
    assert step.name == "resample"
    assert step.parameters["method"] == "common_axis"
    assert step.parameters["n_spectra"] == 2
    assert step.parameters["n_points"] == 5


def test_resample_to_common_axis_preserves_spectral_semantics() -> None:
    """Preserve spectral_axis_name and spectral_unit on the output."""

    s1 = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    s2 = Spectrum(
        axis=np.array([150.0, 250.0, 350.0]),
        intensity=np.array([1.5, 2.5, 3.5]),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = pp.resample.resample_to_common_axis([s1, s2])

    assert result.spectral_axis_name == "raman_shift"
    assert result.spectral_unit == "cm^-1"


def test_resample_to_common_axis_interpolation_accuracy() -> None:
    """Verify interpolation accuracy using a known linear function."""

    # y = 2x, so interpolation should be exact for linear data.
    s1 = Spectrum(
        axis=np.array([0.0, 5.0, 10.0]),
        intensity=np.array([0.0, 10.0, 20.0]),
    )
    s2 = Spectrum(
        axis=np.array([2.0, 7.0, 12.0]),
        intensity=np.array([4.0, 14.0, 24.0]),
    )

    result = pp.resample.resample_to_common_axis([s1, s2], n_points=5)

    # Common range: [2.0, 10.0]
    expected_axis = np.linspace(2.0, 10.0, 5)
    np.testing.assert_allclose(result.axis, expected_axis)
    # y = 2x for both spectra
    np.testing.assert_allclose(result.intensity[0], 2.0 * expected_axis, atol=1e-10)
    np.testing.assert_allclose(result.intensity[1], 2.0 * expected_axis, atol=1e-10)
