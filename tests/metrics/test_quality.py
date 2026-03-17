from __future__ import annotations

import numpy as np
import pytest

import ramankit.metrics as rm
from ramankit import RamanImage, Spectrum, SpectrumCollection


def test_snr_returns_positive_value_for_spectrum() -> None:
    """Return a finite positive SNR for one spectrum with a clear noise window."""

    axis = np.linspace(100.0, 400.0, 7)
    spectrum = Spectrum(
        axis=axis,
        intensity=[0.0, 0.1, 2.0, 5.0, 2.0, 0.1, 0.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.snr(spectrum, noise_region=(100.0, 150.0), signal_region=(200.0, 300.0))

    assert result > 0.0
    assert np.isfinite(result)


def test_snr_returns_zero_when_noise_standard_deviation_is_zero() -> None:
    """Return zero SNR when the noise region is perfectly flat."""

    spectrum = Spectrum(
        axis=[100.0, 200.0, 300.0, 400.0],
        intensity=[1.0, 1.0, 5.0, 1.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.snr(spectrum, noise_region=(100.0, 200.0))

    assert np.isclose(result, 0.0)


def test_band_area_returns_absolute_area_for_descending_axis() -> None:
    """Keep band area positive even when the spectral axis is descending."""

    ascending = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 1.0, 1.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    descending = Spectrum(
        axis=[300.0, 200.0, 100.0],
        intensity=[1.0, 1.0, 1.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    ascending_area = rm.band_area(ascending, region=(100.0, 300.0))
    descending_area = rm.band_area(descending, region=(100.0, 300.0))

    assert np.isclose(ascending_area, 200.0)
    assert np.isclose(descending_area, 200.0)


def test_band_area_supports_simpson_method() -> None:
    """Support Simpson integration in addition to trapezoid."""

    spectrum = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 1.0, 1.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    result = rm.band_area(spectrum, region=(100.0, 300.0), method="simpson")

    assert np.isclose(result, 200.0)


def test_quality_metrics_return_per_item_values_for_collection() -> None:
    """Return one quality-metric value per spectrum in a collection."""

    axis = np.linspace(100.0, 400.0, 7)
    collection = SpectrumCollection(
        axis=axis,
        intensity=[
            [0.0, 0.1, 2.0, 5.0, 2.0, 0.1, 0.0],
            [0.0, 0.2, 3.0, 6.0, 3.0, 0.2, 0.0],
        ],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    snr = rm.snr(collection, noise_region=(100.0, 150.0), signal_region=(200.0, 300.0))
    area = rm.band_area(collection, region=(150.0, 350.0))

    assert snr.shape == (2,)
    assert area.shape == (2,)
    assert np.isclose(
        snr[0],
        rm.snr(collection[0], noise_region=(100.0, 150.0), signal_region=(200.0, 300.0)),
    )
    assert np.isclose(area[1], rm.band_area(collection[1], region=(150.0, 350.0)))


def test_quality_metrics_return_per_pixel_values_for_image() -> None:
    """Return one quality-metric value per image pixel."""

    axis = np.linspace(100.0, 400.0, 7)
    image = RamanImage(
        axis=axis,
        intensity=[
            [
                [0.0, 0.1, 2.0, 5.0, 2.0, 0.1, 0.0],
                [0.0, 0.2, 3.0, 6.0, 3.0, 0.2, 0.0],
            ],
            [
                [0.0, 0.15, 2.5, 5.5, 2.5, 0.15, 0.0],
                [0.0, 0.05, 1.8, 4.8, 1.8, 0.05, 0.0],
            ],
        ],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    snr = rm.snr(image, noise_region=(100.0, 150.0), signal_region=(200.0, 300.0))
    area = rm.band_area(image, region=(150.0, 350.0))

    assert snr.shape == (2, 2)
    assert area.shape == (2, 2)
    assert np.isclose(
        snr[0, 1],
        rm.snr(image.pixel(0, 1), noise_region=(100.0, 150.0), signal_region=(200.0, 300.0)),
    )
    assert np.isclose(area[1, 0], rm.band_area(image.pixel(1, 0), region=(150.0, 350.0)))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: rm.snr(
                Spectrum(
                    axis=[100.0, 200.0, 300.0],
                    intensity=[1.0, 2.0, 3.0],
                    spectral_axis_name="raman_shift",
                    spectral_unit="cm^-1",
                ),
                noise_region=(500.0, 600.0),
            ),
            "No data points found in noise_region",
        ),
        (
            lambda: rm.band_area(
                Spectrum(
                    axis=[100.0, 200.0, 300.0],
                    intensity=[1.0, 2.0, 3.0],
                    spectral_axis_name="raman_shift",
                    spectral_unit="cm^-1",
                ),
                region=(100.0, 300.0),
                method="bad",
            ),
            "Unsupported band area method",
        ),
    ],
)
def test_quality_metrics_validate_regions_and_methods(factory, message: str) -> None:
    """Reject empty regions and unsupported area methods."""

    with pytest.raises(ValueError, match=message):
        factory()

