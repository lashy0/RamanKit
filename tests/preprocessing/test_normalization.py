from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import RamanImage, Spectrum, SpectrumCollection
from tests._test_helpers import apply_collection_row_by_row, apply_image_pixel_by_pixel


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

def test_minmax_apply_scales_spectrum_to_unit_interval() -> None:
    """Scale a spectrum so its minimum is zero and its maximum is one."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[2.0, 5.0, 3.0])

    normalized = pp.normalization.MinMax().apply(spectrum)

    assert np.isclose(np.min(normalized.intensity), 0.0)
    assert np.isclose(np.max(normalized.intensity), 1.0)
    assert normalized.provenance.steps[-1].parameters["method"] == "minmax"

def test_minmax_apply_preserves_raman_image_type() -> None:
    """Normalize each image spectrum independently into the [0, 1] range."""

    axis = np.linspace(100.0, 400.0, 4)
    image = RamanImage(
        axis=axis,
        intensity=np.array([[[1.0, 3.0, 5.0, 2.0], [4.0, 6.0, 8.0, 5.0]]]),
    )

    normalized = pp.normalization.MinMax().apply(image)

    assert isinstance(normalized, RamanImage)
    assert np.isclose(np.min(normalized.intensity[0, 0]), 0.0)
    assert np.isclose(np.max(normalized.intensity[0, 0]), 1.0)
    assert np.isclose(np.min(normalized.intensity[0, 1]), 0.0)
    assert np.isclose(np.max(normalized.intensity[0, 1]), 1.0)

def test_minmax_apply_raises_for_zero_denominator() -> None:
    """Reject min-max normalization for constant spectra."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="denominator"):
        pp.normalization.MinMax().apply(spectrum)

def test_vector_apply_raises_for_zero_denominator() -> None:
    """Reject normalization when the denominator is zero."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="denominator"):
        pp.normalization.Vector().apply(spectrum)




@pytest.mark.parametrize(
    "step",
    [
        pp.normalization.Vector(),
        pp.normalization.Area(),
        pp.normalization.Max(),
        pp.normalization.MinMax(),
    ],
)
def test_normalization_batch_path_matches_row_by_row_for_collection(
    step: pp.PreprocessingStep,
) -> None:
    """Batch normalization should match per-spectrum application on collections."""

    axis = np.array([400.0, 300.0, 200.0, 100.0], dtype=np.float64)
    collection = SpectrumCollection(
        axis=axis,
        intensity=np.array(
            [
                [1.0, 3.0, 5.0, 2.0],
                [2.0, 4.0, 6.0, 3.0],
                [1.5, 2.5, 4.5, 2.5],
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
        pp.normalization.Vector(),
        pp.normalization.Area(),
        pp.normalization.Max(),
        pp.normalization.MinMax(),
    ],
)
def test_normalization_batch_path_matches_row_by_row_for_image(step: pp.PreprocessingStep) -> None:
    """Batch normalization should match per-pixel application on Raman images."""

    axis = np.array([400.0, 300.0, 200.0, 100.0], dtype=np.float64)
    image = RamanImage(
        axis=axis,
        intensity=np.array(
            [
                [[1.0, 3.0, 5.0, 2.0], [2.0, 4.0, 6.0, 3.0]],
                [[1.5, 2.5, 4.5, 2.5], [3.0, 5.0, 7.0, 4.0]],
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


