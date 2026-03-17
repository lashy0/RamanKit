from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import RamanImage, Spectrum, SpectrumCollection
from tests._test_helpers import apply_collection_row_by_row, apply_image_pixel_by_pixel


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

def test_whittaker_apply_preserves_collection_axis_and_type() -> None:
    """Apply Whittaker smoothing across each spectrum in a collection."""

    axis = np.linspace(100.0, 500.0, 9)
    collection = SpectrumCollection(
        axis=axis,
        intensity=np.array(
            [
                [0.0, 1.0, 3.0, 8.0, 12.0, 7.0, 3.0, 1.0, 0.0],
                [1.0, 2.0, 5.0, 9.0, 13.0, 8.0, 4.0, 2.0, 1.0],
            ]
        ),
    )

    smoothed = pp.smoothing.Whittaker(lam=1e3).apply(collection)

    assert isinstance(smoothed, SpectrumCollection)
    assert np.array_equal(smoothed.axis, collection.axis)
    assert smoothed.intensity.shape == collection.intensity.shape
    assert smoothed.provenance.steps[-1].parameters["method"] == "whittaker"

def test_whittaker_apply_raises_for_non_positive_lam() -> None:
    """Reject non-positive Whittaker smoothing strengths."""

    with pytest.raises(ValueError, match="lam"):
        pp.smoothing.Whittaker(lam=0.0)

def test_gaussian_apply_preserves_raman_image_shape() -> None:
    """Apply Gaussian smoothing across the last axis of a Raman image."""

    axis = np.linspace(100.0, 400.0, 9)
    image = RamanImage(
        axis=axis,
        intensity=np.stack([0.01 * axis + 1.0, 0.02 * axis + 2.0]).reshape(1, 2, 9),
    )

    smoothed = pp.smoothing.Gaussian(sigma=1.25).apply(image)

    assert isinstance(smoothed, RamanImage)
    assert smoothed.intensity.shape == image.intensity.shape
    assert smoothed.provenance.steps[-1].parameters["method"] == "gaussian"

def test_gaussian_apply_raises_for_non_positive_sigma() -> None:
    """Reject non-positive Gaussian smoothing widths."""

    with pytest.raises(ValueError, match="sigma"):
        pp.smoothing.Gaussian(sigma=-1.0)

@pytest.mark.parametrize(
    "step",
    [pp.smoothing.Whittaker(lam=1e3), pp.smoothing.Gaussian(sigma=1.0)],
)
def test_smoothing_steps_reduce_local_variation(step: pp.PreprocessingStep) -> None:
    """Reduce local point-to-point variation on a noisy spectrum."""

    axis = np.linspace(100.0, 400.0, 11)
    intensity = np.array([0.0, 1.5, 0.5, 2.0, 1.0, 3.5, 1.5, 2.5, 1.0, 1.5, 0.5])
    spectrum = Spectrum(axis=axis, intensity=intensity)

    smoothed = step.apply(spectrum)

    original_variation = np.sum(np.abs(np.diff(spectrum.intensity)))
    smoothed_variation = np.sum(np.abs(np.diff(smoothed.intensity)))
    assert smoothed_variation < original_variation




@pytest.mark.parametrize(
    "step",
    [
        pp.smoothing.SavGol(window_length=5, polyorder=2),
        pp.smoothing.Gaussian(sigma=1.0),
    ],
)
def test_smoothing_batch_path_matches_row_by_row_for_collection(step: pp.PreprocessingStep) -> None:
    """Batch smoothing should match per-spectrum application on collections."""

    axis = np.linspace(500.0, 100.0, 9)
    collection = SpectrumCollection(
        axis=axis,
        intensity=np.array(
            [
                [0.0, 1.0, 2.0, 4.0, 7.0, 4.0, 2.0, 1.0, 0.0],
                [1.0, 2.0, 3.0, 6.0, 8.0, 5.0, 3.0, 2.0, 1.0],
                [0.5, 1.5, 1.0, 3.0, 5.0, 3.0, 1.0, 1.5, 0.5],
            ],
            dtype=np.float64,
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    batch_result = step.apply(collection)
    expected = apply_collection_row_by_row(step, collection)

    assert isinstance(batch_result, SpectrumCollection)
    assert np.array_equal(batch_result.axis, collection.axis)
    assert batch_result.provenance.steps[-1].name == step.function_name
    assert batch_result.provenance.steps[-1].parameters["method"] == step.method_name
    assert np.allclose(batch_result.intensity, expected.intensity)


@pytest.mark.parametrize(
    "step",
    [
        pp.smoothing.SavGol(window_length=5, polyorder=2),
        pp.smoothing.Gaussian(sigma=1.0),
    ],
)
def test_smoothing_batch_path_matches_row_by_row_for_image(step: pp.PreprocessingStep) -> None:
    """Batch smoothing should match per-pixel application on Raman images."""

    axis = np.linspace(500.0, 100.0, 9)
    image = RamanImage(
        axis=axis,
        intensity=np.array(
            [
                [
                    [0.0, 1.0, 2.0, 4.0, 7.0, 4.0, 2.0, 1.0, 0.0],
                    [1.0, 2.0, 3.0, 6.0, 8.0, 5.0, 3.0, 2.0, 1.0],
                ],
                [
                    [0.5, 1.5, 1.0, 3.0, 5.0, 3.0, 1.0, 1.5, 0.5],
                    [1.0, 1.5, 2.5, 4.0, 6.0, 4.0, 2.5, 1.5, 1.0],
                ],
            ],
            dtype=np.float64,
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    batch_result = step.apply(image)
    expected = apply_image_pixel_by_pixel(step, image)

    assert isinstance(batch_result, RamanImage)
    assert np.array_equal(batch_result.axis, image.axis)
    assert batch_result.provenance.steps[-1].name == step.function_name
    assert batch_result.provenance.steps[-1].parameters["method"] == step.method_name
    assert np.allclose(batch_result.intensity, expected.intensity)


