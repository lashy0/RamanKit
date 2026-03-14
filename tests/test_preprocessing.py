from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection


def test_asls_apply_preserves_spectrum_metadata_and_type() -> None:
    """Return a corrected spectrum while preserving metadata and provenance."""

    metadata = Metadata(sample="sample-1")
    provenance = Provenance(steps=(ProvenanceStep(name="load"),))
    axis = np.linspace(100.0, 400.0, 11)
    intensity = 0.02 * axis + np.array([0.0, 0.0, 0.1, 0.4, 1.2, 2.5, 1.2, 0.4, 0.1, 0.0, 0.0])
    spectrum = Spectrum(
        axis=axis,
        intensity=intensity,
        metadata=metadata,
        provenance=provenance,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    corrected = pp.baseline.ASLS().apply(spectrum)

    assert isinstance(corrected, Spectrum)
    assert corrected.metadata == metadata
    assert corrected.axis.shape == spectrum.axis.shape
    assert corrected.provenance.steps[-1].name == "baseline_correct"
    assert corrected.provenance.steps[-1].parameters["method"] == "asls"
    assert not np.allclose(corrected.intensity, spectrum.intensity)


def test_asls_apply_preserves_raman_image_shape() -> None:
    """Apply baseline correction across the last axis of a Raman image."""

    axis = np.linspace(100.0, 400.0, 9)
    image = RamanImage(
        axis=axis,
        intensity=np.stack([0.01 * axis + 1.0, 0.02 * axis + 2.0]).reshape(1, 2, 9),
    )

    corrected = pp.baseline.ASLS().apply(image)

    assert isinstance(corrected, RamanImage)
    assert corrected.intensity.shape == image.intensity.shape
    assert corrected.provenance.steps[-1].parameters["method"] == "asls"


def test_savgol_apply_returns_collection_with_same_axis() -> None:
    """Apply smoothing across each spectrum in a collection."""

    axis = np.linspace(100.0, 500.0, 9)
    collection = SpectrumCollection(
        axis=axis,
        intensity=np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 2.0, 4.0, 5.0, 4.0, 2.0, 1.0, 0.0, 1.0],
            ]
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    smoothed = pp.smoothing.SavGol(window_length=5, polyorder=2).apply(collection)

    assert isinstance(smoothed, SpectrumCollection)
    assert np.array_equal(smoothed.axis, collection.axis)
    assert smoothed.intensity.shape == collection.intensity.shape
    assert smoothed.provenance.steps[-1].name == "smooth"


def test_savgol_apply_raises_for_even_window_length() -> None:
    """Reject invalid Savitzky-Golay window lengths."""

    spectrum = Spectrum(axis=np.arange(5.0), intensity=np.arange(5.0))

    with pytest.raises(ValueError, match="positive odd integer"):
        pp.smoothing.SavGol(window_length=4).apply(spectrum)


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


def test_vector_apply_raises_for_zero_denominator() -> None:
    """Reject normalization when the denominator is zero."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="denominator"):
        pp.normalization.Vector().apply(spectrum)


def test_pipeline_applies_steps_in_sequence() -> None:
    """Apply configured preprocessing steps in sequence through a pipeline."""

    spectrum = Spectrum(
        axis=np.linspace(100.0, 400.0, 9),
        intensity=np.array([5.0, 6.0, 8.0, 15.0, 30.0, 16.0, 9.0, 6.0, 5.0]),
        provenance=Provenance(steps=(ProvenanceStep(name="load"),)),
    )
    pipeline = pp.Pipeline(
        [
            pp.baseline.ASLS(),
            pp.smoothing.SavGol(window_length=5, polyorder=2),
            pp.normalization.Vector(),
        ]
    )

    processed = pipeline.apply(spectrum)

    assert isinstance(processed, Spectrum)
    assert [step.name for step in processed.provenance.steps[-3:]] == [
        "baseline_correct",
        "smooth",
        "normalize",
    ]


def test_whitaker_hayes_apply_reduces_spike_without_changing_shape() -> None:
    """Reduce a strong spike while preserving the spectrum shape and metadata."""

    metadata = Metadata(sample="sample-2")
    spectrum = Spectrum(
        axis=np.linspace(100.0, 400.0, 7),
        intensity=np.array([1.0, 1.1, 1.2, 20.0, 1.2, 1.1, 1.0]),
        metadata=metadata,
    )

    despiked = pp.despike.WhitakerHayes(threshold=3.0, kernel_size=3).apply(spectrum)

    assert isinstance(despiked, Spectrum)
    assert despiked.metadata == metadata
    assert despiked.intensity.shape == spectrum.intensity.shape
    assert despiked.intensity[3] < spectrum.intensity[3]
    assert despiked.provenance.steps[-1].name == "despike"
    assert despiked.provenance.steps[-1].parameters["method"] == "whitaker_hayes"


def test_whitaker_hayes_apply_raises_for_invalid_kernel_size() -> None:
    """Reject invalid Whitaker-Hayes kernel sizes."""

    spectrum = Spectrum(axis=np.arange(5.0), intensity=np.array([1.0, 1.0, 5.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="kernel_size"):
        pp.despike.WhitakerHayes(kernel_size=4).apply(spectrum)


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


def test_pipeline_applies_axis_changing_step_and_keeps_provenance_order() -> None:
    """Apply a pipeline that includes resampling and keep provenance ordered."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
    )
    pipeline = pp.Pipeline(
        [
            pp.resample.Linear(target_axis=np.array([150.0, 250.0, 350.0, 450.0])),
            pp.normalization.Max(),
        ]
    )

    processed = pipeline.apply(spectrum)

    assert np.array_equal(processed.axis, np.array([150.0, 250.0, 350.0, 450.0]))
    assert [step.name for step in processed.provenance.steps[-2:]] == ["resample", "normalize"]
