from __future__ import annotations

import numpy as np
import pytest

import ramankit.metrics as rm
from ramankit import RamanImage, Spectrum, SpectrumCollection


def test_cosine_similarity_returns_one_for_identical_spectra() -> None:
    """Return one for identical spectra."""

    spectrum = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 2.0, 3.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.cosine_similarity(spectrum, spectrum.copy())

    assert np.isclose(result, 1.0)


def test_pearson_correlation_is_invariant_to_additive_offset() -> None:
    """Keep Pearson correlation at one under additive offsets."""

    left = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 2.0, 4.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    right = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[6.0, 7.0, 9.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.pearson_correlation(left, right)

    assert np.isclose(result, 1.0)


def test_mse_returns_zero_for_identical_spectra() -> None:
    """Return zero mean squared error for identical spectra."""

    spectrum = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 2.0, 3.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.mse(spectrum, spectrum.copy())

    assert np.isclose(result, 0.0)


def test_similarity_metrics_return_per_spectrum_values_for_collection() -> None:
    """Return one value per spectrum for collection inputs."""

    left = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    right = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    cosine = rm.cosine_similarity(left, right)
    pearson = rm.pearson_correlation(left, right)
    error = rm.mse(left, right)

    assert cosine.shape == (2,)
    assert pearson.shape == (2,)
    assert error.shape == (2,)
    assert np.isclose(cosine[0], rm.cosine_similarity(left[0], right[0]))
    assert np.isclose(pearson[1], rm.pearson_correlation(left[1], right[1]))
    assert np.isclose(error[1], rm.mse(left[1], right[1]))


def test_similarity_metrics_return_per_pixel_values_for_image() -> None:
    """Return one value per pixel for Raman image inputs."""

    left = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    right = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[[[1.0, 2.0, 3.0], [2.5, 3.0, 4.5]], [[2.0, 3.0, 1.0], [4.0, 5.0, 7.0]]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    cosine = rm.cosine_similarity(left, right)
    error = rm.mse(left, right)

    assert cosine.shape == (2, 2)
    assert error.shape == (2, 2)
    assert np.isclose(cosine[0, 0], rm.cosine_similarity(left.pixel(0, 0), right.pixel(0, 0)))
    assert np.isclose(error[1, 1], rm.mse(left.pixel(1, 1), right.pixel(1, 1)))


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (
            Spectrum(
                axis=[100.0, 200.0],
                intensity=[1.0, 2.0],
                spectral_axis_name="raman_shift",
                spectral_unit="cm^-1",
            ),
            Spectrum(
                axis=[100.0, 250.0],
                intensity=[1.0, 2.0],
                spectral_axis_name="raman_shift",
                spectral_unit="cm^-1",
            ),
            "axes must match exactly",
        ),
        (
            Spectrum(
                axis=[100.0, 200.0],
                intensity=[1.0, 2.0],
                spectral_axis_name="raman_shift",
                spectral_unit="cm^-1",
            ),
            Spectrum(
                axis=[100.0, 200.0],
                intensity=[1.0, 2.0],
                spectral_axis_name="wavelength",
                spectral_unit="cm^-1",
            ),
            "axis names must match exactly",
        ),
        (
            Spectrum(
                axis=[100.0, 200.0],
                intensity=[1.0, 2.0],
                spectral_axis_name="raman_shift",
                spectral_unit="cm^-1",
            ),
            SpectrumCollection(
                axis=[100.0, 200.0],
                intensity=[[1.0, 2.0]],
                spectral_axis_name="raman_shift",
                spectral_unit="cm^-1",
            ),
            "same container type",
        ),
    ],
)
def test_similarity_metrics_validate_compatible_inputs(
    left: Spectrum | SpectrumCollection,
    right: Spectrum | SpectrumCollection,
    message: str,
) -> None:
    """Reject incompatible inputs for pairwise spectral metrics."""

    with pytest.raises(ValueError, match=message):
        rm.cosine_similarity(left, right)
