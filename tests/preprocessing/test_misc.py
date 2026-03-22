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


# --- Cropper batch tests ---


def test_cropper_batch_matches_row_by_row() -> None:
    """Batch crop of a collection matches row-by-row application."""

    from tests._test_helpers import apply_collection_row_by_row

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]),
    )
    step = pp.misc.Cropper(lower_bound=200.0, upper_bound=400.0)

    batch_result = step.apply(collection)
    row_result = apply_collection_row_by_row(step, collection)

    assert np.array_equal(batch_result.axis, row_result.axis)
    assert np.allclose(batch_result.intensity, row_result.intensity)


# --- IndexCropper tests ---


def test_index_cropper_apply_slices_by_index() -> None:
    """Crop a spectrum using both start and stop indices."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    cropped = pp.misc.IndexCropper(start_index=1, stop_index=4).apply(spectrum)

    assert np.array_equal(cropped.axis, np.array([200.0, 300.0, 400.0]))
    assert np.array_equal(cropped.intensity, np.array([2.0, 3.0, 4.0]))


def test_index_cropper_apply_with_start_only() -> None:
    """Crop from a start index to the end of the spectrum."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    cropped = pp.misc.IndexCropper(start_index=3).apply(spectrum)

    assert np.array_equal(cropped.axis, np.array([400.0, 500.0]))
    assert np.array_equal(cropped.intensity, np.array([4.0, 5.0]))


def test_index_cropper_apply_with_stop_only() -> None:
    """Crop from the beginning up to a stop index."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    cropped = pp.misc.IndexCropper(stop_index=2).apply(spectrum)

    assert np.array_equal(cropped.axis, np.array([100.0, 200.0]))
    assert np.array_equal(cropped.intensity, np.array([1.0, 2.0]))


def test_index_cropper_raises_when_both_none() -> None:
    """Reject IndexCropper when neither start nor stop is provided."""

    with pytest.raises(ValueError, match="at least one"):
        pp.misc.IndexCropper()


def test_index_cropper_raises_for_start_ge_stop() -> None:
    """Reject IndexCropper when start_index >= stop_index."""

    with pytest.raises(ValueError, match="smaller"):
        pp.misc.IndexCropper(start_index=3, stop_index=2)

    with pytest.raises(ValueError, match="smaller"):
        pp.misc.IndexCropper(start_index=3, stop_index=3)


def test_index_cropper_raises_for_non_integer() -> None:
    """Reject non-integer index values."""

    with pytest.raises(TypeError, match="integer"):
        pp.misc.IndexCropper(start_index=1.5)  # type: ignore[arg-type]


def test_index_cropper_records_provenance() -> None:
    """Record crop provenance with index parameters."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
    )

    cropped = pp.misc.IndexCropper(start_index=1).apply(spectrum)

    step = cropped.provenance.steps[-1]
    assert step.name == "crop"
    assert step.parameters["method"] == "index_range"
    assert step.parameters["start_index"] == 1
    assert step.parameters["stop_index"] is None


def test_index_cropper_is_axis_transform_step() -> None:
    """IndexCropper is a subclass of AxisTransformStep."""

    assert issubclass(pp.misc.IndexCropper, pp.AxisTransformStep)


def test_index_cropper_apply_on_collection() -> None:
    """Apply IndexCropper to a SpectrumCollection."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]),
    )

    cropped = pp.misc.IndexCropper(start_index=1, stop_index=4).apply(collection)

    assert isinstance(cropped, SpectrumCollection)
    assert cropped.intensity.shape == (2, 3)
    assert np.array_equal(cropped.axis, np.array([200.0, 300.0, 400.0]))
