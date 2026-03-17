"""Regression tests for synthetic Raman spectrum generation helpers."""

from __future__ import annotations

import numpy as np
import pytest

import ramankit.synthetic as rsyn
from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection
from tests._synthetic_helpers import gaussian, voigt


def test_generate_spectrum_returns_spectrum_with_metadata_and_provenance() -> None:
    """Generate one synthetic spectrum without losing explicit axis metadata."""

    axis = np.linspace(100.0, 140.0, 5)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=3.0, center=120.0, width=4.0),),
        baseline=rsyn.LinearBaseline(offset=0.5, slope=0.1),
    )
    metadata = Metadata(sample="synthetic-demo")
    provenance = Provenance(steps=(ProvenanceStep(name="seed_input"),))

    spectrum = rsyn.generate_spectrum(
        axis,
        config,
        metadata=metadata,
        provenance=provenance,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    expected = gaussian(axis, amplitude=3.0, center=120.0, width=4.0) + (
        0.5 + 0.1 * (axis - 100.0)
    )

    assert isinstance(spectrum, Spectrum)
    assert np.allclose(spectrum.intensity, expected)
    assert spectrum.metadata == metadata
    assert spectrum.provenance.source == "synthetic"
    assert spectrum.provenance.steps[0].name == "seed_input"
    assert spectrum.provenance.steps[-1].name == "generate_spectrum"
    assert spectrum.spectral_axis_name == "raman_shift"
    assert spectrum.spectral_unit == "cm^-1"


def test_generate_spectrum_supports_polynomial_baseline() -> None:
    """Generate one synthetic spectrum with a polynomial baseline."""

    axis = np.linspace(100.0, 140.0, 5)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=4.0),),
        baseline=rsyn.PolynomialBaseline(coefficients=(0.5, 0.1, 0.01)),
    )

    spectrum = rsyn.generate_spectrum(axis, config)

    shifted_axis = axis - axis[0]
    expected = gaussian(axis, amplitude=2.0, center=120.0, width=4.0) + (
        0.5 + 0.1 * shifted_axis + 0.01 * shifted_axis**2
    )

    assert np.allclose(spectrum.intensity, expected)


def test_generate_spectrum_supports_exponential_baseline() -> None:
    """Generate one synthetic spectrum with an exponential baseline."""

    axis = np.linspace(100.0, 140.0, 5)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=4.0),),
        baseline=rsyn.ExponentialBaseline(amplitude=0.4, rate=0.05, offset=0.2),
    )

    spectrum = rsyn.generate_spectrum(axis, config)

    shifted_axis = axis - axis[0]
    expected = gaussian(axis, amplitude=2.0, center=120.0, width=4.0) + (
        0.2 + 0.4 * np.exp(0.05 * shifted_axis)
    )

    assert np.allclose(spectrum.intensity, expected)


def test_generate_spectrum_supports_voigt_components() -> None:
    """Generate one synthetic spectrum with a Voigt peak profile."""

    axis = np.linspace(150.0, 170.0, 11)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(
            rsyn.PeakComponent(
                model="voigt",
                amplitude=5.0,
                center=160.0,
                sigma=1.5,
                gamma=2.0,
            ),
        ),
    )

    spectrum = rsyn.generate_spectrum(axis, config)

    expected = voigt(axis, amplitude=5.0, center=160.0, sigma=1.5, gamma=2.0)

    assert np.allclose(spectrum.intensity, expected)


def test_generate_collection_returns_collection_with_shared_axis() -> None:
    """Generate one collection from multiple independent spectrum configurations."""

    axis = np.linspace(100.0, 140.0, 5)
    configs = [
        rsyn.SyntheticSpectrumConfig(
            peaks=(rsyn.PeakComponent(amplitude=2.0, center=110.0, width=3.0),),
        ),
        rsyn.SyntheticSpectrumConfig(
            peaks=(rsyn.PeakComponent(model="lorentzian", amplitude=4.0, center=130.0, width=2.5),),
        ),
    ]

    collection = rsyn.generate_collection(axis, configs)

    assert isinstance(collection, SpectrumCollection)
    assert collection.intensity.shape == (2, 5)
    assert collection.provenance.source == "synthetic"
    assert collection.provenance.steps[-1].name == "generate_collection"


def test_generate_image_returns_raman_image_with_expected_shape() -> None:
    """Generate one Raman image from a 2D grid of spectrum configurations."""

    axis = np.linspace(100.0, 160.0, 7)
    configs = [
        [
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=3.0),),
            ),
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=1.5, center=140.0, width=4.0),),
            ),
        ],
        [
            rsyn.SyntheticSpectrumConfig(
                peaks=(
                    rsyn.PeakComponent(
                        model="lorentzian",
                        amplitude=3.0,
                        center=125.0,
                        width=2.0,
                    ),
                ),
            ),
            rsyn.SyntheticSpectrumConfig(
                peaks=(
                    rsyn.PeakComponent(
                        model="voigt",
                        amplitude=5.0,
                        center=145.0,
                        sigma=1.2,
                        gamma=1.6,
                    ),
                ),
            ),
        ],
    ]

    image = rsyn.generate_image(axis, configs)

    assert isinstance(image, RamanImage)
    assert image.intensity.shape == (2, 2, 7)
    assert image.provenance.source == "synthetic"
    assert image.provenance.steps[-1].name == "generate_image"


def test_generate_spectrum_noise_is_reproducible_from_seed() -> None:
    """Use the per-config noise seed to keep synthetic generation reproducible."""

    axis = np.linspace(100.0, 140.0, 9)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=3.0),),
        noise=rsyn.GaussianNoise(sigma=0.2, seed=17),
    )

    left = rsyn.generate_spectrum(axis, config)
    right = rsyn.generate_spectrum(axis, config)

    assert np.array_equal(left.intensity, right.intensity)


def test_generate_spectrum_noise_uses_shared_generator_when_provided() -> None:
    """Consume a caller-provided RNG stream when one is supplied explicitly."""

    axis = np.linspace(100.0, 140.0, 9)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=3.0),),
        noise=rsyn.GaussianNoise(sigma=0.2),
    )

    left = rsyn.generate_spectrum(axis, config, rng=np.random.default_rng(123))
    right = rsyn.generate_spectrum(axis, config, rng=np.random.default_rng(123))

    assert np.array_equal(left.intensity, right.intensity)


def test_peak_component_raises_for_missing_model_parameters() -> None:
    """Reject peak definitions that omit required model-specific parameters."""

    with pytest.raises(ValueError, match="width"):
        rsyn.PeakComponent(amplitude=2.0, center=120.0)

    with pytest.raises(ValueError, match="sigma and PeakComponent.gamma"):
        rsyn.PeakComponent(model="voigt", amplitude=2.0, center=120.0)


def test_polynomial_baseline_raises_for_invalid_coefficients() -> None:
    """Reject empty or non-finite polynomial baseline coefficients."""

    with pytest.raises(ValueError, match="at least one term"):
        rsyn.PolynomialBaseline(coefficients=())

    with pytest.raises(ValueError, match="finite values"):
        rsyn.PolynomialBaseline(coefficients=(1.0, np.inf))


def test_exponential_baseline_raises_for_non_finite_parameters() -> None:
    """Reject non-finite exponential baseline parameters."""

    with pytest.raises(ValueError, match="amplitude"):
        rsyn.ExponentialBaseline(amplitude=np.nan, rate=0.1)

    with pytest.raises(ValueError, match="rate"):
        rsyn.ExponentialBaseline(amplitude=1.0, rate=np.inf)

    with pytest.raises(ValueError, match="offset"):
        rsyn.ExponentialBaseline(amplitude=1.0, rate=0.1, offset=np.nan)


def test_generate_image_raises_for_ragged_config_rows() -> None:
    """Reject image config grids whose rows have inconsistent widths."""

    axis = np.linspace(100.0, 140.0, 5)
    config = rsyn.SyntheticSpectrumConfig(
        peaks=(rsyn.PeakComponent(amplitude=2.0, center=120.0, width=3.0),),
    )

    with pytest.raises(ValueError, match="same width"):
        rsyn.generate_image(axis, [[config, config], [config]])
