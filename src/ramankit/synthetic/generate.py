"""Generate synthetic Raman spectra, collections, and images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import voigt_profile  # type: ignore[import-untyped]

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import (
    Metadata,
    Provenance,
    ProvenanceStep,
    ensure_metadata,
    ensure_provenance,
)
from ramankit.core.spectrum import Spectrum

type PeakModel = Literal["gaussian", "lorentzian", "voigt"]
type Array1D = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PeakComponent:
    """Describe one parametric peak component for synthetic generation."""

    amplitude: float
    center: float
    model: PeakModel = "gaussian"
    width: float | None = None
    sigma: float | None = None
    gamma: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude):
            raise ValueError("Expected PeakComponent.amplitude to be finite.")
        if not np.isfinite(self.center):
            raise ValueError("Expected PeakComponent.center to be finite.")

        if self.model in {"gaussian", "lorentzian"}:
            if self.width is None:
                raise ValueError(f"Expected PeakComponent.width for model '{self.model}'.")
            if not np.isfinite(self.width) or self.width <= 0.0:
                raise ValueError("Expected PeakComponent.width to be a positive finite value.")
            if self.sigma is not None or self.gamma is not None:
                raise ValueError(
                    f"PeakComponent model '{self.model}' does not use sigma/gamma parameters."
                )
            return

        if self.sigma is None or self.gamma is None:
            raise ValueError(
                "Expected PeakComponent.sigma and PeakComponent.gamma for model 'voigt'."
            )
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("Expected PeakComponent.sigma to be a positive finite value.")
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("Expected PeakComponent.gamma to be a positive finite value.")
        if self.width is not None:
            raise ValueError("PeakComponent model 'voigt' does not use width.")


@dataclass(frozen=True, slots=True)
class LinearBaseline:
    """Describe a linear baseline added to a synthetic spectrum."""

    offset: float = 0.0
    slope: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.offset):
            raise ValueError("Expected LinearBaseline.offset to be finite.")
        if not np.isfinite(self.slope):
            raise ValueError("Expected LinearBaseline.slope to be finite.")


@dataclass(frozen=True, slots=True)
class PolynomialBaseline:
    """Describe a polynomial baseline added to a synthetic spectrum."""

    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if not self.coefficients:
            raise ValueError(
                "Expected PolynomialBaseline.coefficients to contain at least one term."
            )

        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "Expected PolynomialBaseline.coefficients to contain only finite values."
            )


@dataclass(frozen=True, slots=True)
class ExponentialBaseline:
    """Describe an exponential baseline added to a synthetic spectrum."""

    amplitude: float
    rate: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude):
            raise ValueError("Expected ExponentialBaseline.amplitude to be finite.")
        if not np.isfinite(self.rate):
            raise ValueError("Expected ExponentialBaseline.rate to be finite.")
        if not np.isfinite(self.offset):
            raise ValueError("Expected ExponentialBaseline.offset to be finite.")


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    """Describe additive Gaussian noise for synthetic generation."""

    sigma: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("Expected GaussianNoise.sigma to be a non-negative finite value.")


@dataclass(frozen=True, slots=True)
class SyntheticSpectrumConfig:
    """Bundle the components needed to generate one synthetic spectrum."""

    peaks: tuple[PeakComponent, ...]
    baseline: LinearBaseline | PolynomialBaseline | ExponentialBaseline | None = None
    noise: GaussianNoise | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "peaks", tuple(self.peaks))
        if not self.peaks:
            raise ValueError("Expected SyntheticSpectrumConfig.peaks to contain at least one peak.")


def generate_spectrum(
    axis: npt.ArrayLike,
    config: SyntheticSpectrumConfig,
    *,
    metadata: Metadata | None = None,
    provenance: Provenance | None = None,
    spectral_axis_name: str | None = None,
    spectral_unit: str | None = None,
    rng: np.random.Generator | None = None,
) -> Spectrum:
    """Generate one synthetic spectrum from an explicit axis and configuration."""

    spectrum_axis = np.asarray(axis, dtype=np.float64)
    intensity = _generate_intensity(spectrum_axis, config, rng=rng)
    spectrum_provenance = _append_generation_step(
        ensure_provenance(provenance).with_source("synthetic"),
        name="generate_spectrum",
        parameters={"peak_count": len(config.peaks)},
    )
    return Spectrum(
        axis=spectrum_axis,
        intensity=intensity,
        metadata=ensure_metadata(metadata),
        provenance=spectrum_provenance,
        spectral_axis_name=spectral_axis_name,
        spectral_unit=spectral_unit,
    )


def generate_collection(
    axis: npt.ArrayLike,
    configs: Sequence[SyntheticSpectrumConfig],
    *,
    metadata: Metadata | None = None,
    provenance: Provenance | None = None,
    spectral_axis_name: str | None = None,
    spectral_unit: str | None = None,
    rng: np.random.Generator | None = None,
) -> SpectrumCollection:
    """Generate a synthetic collection from a sequence of spectrum configurations."""

    if not configs:
        raise ValueError("generate_collection requires at least one SyntheticSpectrumConfig.")

    collection_axis = np.asarray(axis, dtype=np.float64)
    intensities = np.stack(
        [_generate_intensity(collection_axis, config, rng=rng) for config in configs],
        axis=0,
    )
    collection_provenance = _append_generation_step(
        ensure_provenance(provenance).with_source("synthetic"),
        name="generate_collection",
        parameters={"count": len(configs)},
    )
    return SpectrumCollection(
        axis=collection_axis,
        intensity=intensities,
        metadata=ensure_metadata(metadata),
        provenance=collection_provenance,
        spectral_axis_name=spectral_axis_name,
        spectral_unit=spectral_unit,
    )


def generate_image(
    axis: npt.ArrayLike,
    configs: Sequence[Sequence[SyntheticSpectrumConfig]],
    *,
    metadata: Metadata | None = None,
    provenance: Provenance | None = None,
    spectral_axis_name: str | None = None,
    spectral_unit: str | None = None,
    rng: np.random.Generator | None = None,
) -> RamanImage:
    """Generate a synthetic Raman image from a 2D grid of spectrum configurations."""

    if not configs or not configs[0]:
        raise ValueError("generate_image requires a non-empty 2D grid of SyntheticSpectrumConfig.")

    height = len(configs)
    width = len(configs[0])
    for row in configs:
        if len(row) != width:
            raise ValueError("Expected every generate_image config row to have the same width.")

    image_axis = np.asarray(axis, dtype=np.float64)
    intensities = np.stack(
        [
            np.stack(
                [_generate_intensity(image_axis, config, rng=rng) for config in row],
                axis=0,
            )
            for row in configs
        ],
        axis=0,
    )
    image_provenance = _append_generation_step(
        ensure_provenance(provenance).with_source("synthetic"),
        name="generate_image",
        parameters={"height": height, "width": width},
    )
    return RamanImage(
        axis=image_axis,
        intensity=intensities,
        metadata=ensure_metadata(metadata),
        provenance=image_provenance,
        spectral_axis_name=spectral_axis_name,
        spectral_unit=spectral_unit,
    )


def _generate_intensity(
    axis: Array1D,
    config: SyntheticSpectrumConfig,
    *,
    rng: np.random.Generator | None,
) -> Array1D:
    intensity = np.zeros_like(axis, dtype=np.float64)
    for peak in config.peaks:
        intensity += _evaluate_peak(axis, peak)

    if config.baseline is not None:
        intensity += _evaluate_baseline(axis, config.baseline)

    if config.noise is not None:
        intensity += _sample_noise(axis.shape, config.noise, rng=rng)

    return intensity


def _evaluate_peak(axis: Array1D, peak: PeakComponent) -> Array1D:
    if peak.model == "gaussian":
        assert peak.width is not None
        return peak.amplitude * np.exp(-0.5 * ((axis - peak.center) / peak.width) ** 2)
    if peak.model == "lorentzian":
        assert peak.width is not None
        return peak.amplitude / (1.0 + ((axis - peak.center) / peak.width) ** 2)

    assert peak.sigma is not None and peak.gamma is not None
    return peak.amplitude * voigt_profile(axis - peak.center, peak.sigma, peak.gamma)


def _evaluate_baseline(
    axis: Array1D,
    baseline: LinearBaseline | PolynomialBaseline | ExponentialBaseline,
) -> Array1D:
    shifted_axis = axis - float(axis[0])
    if isinstance(baseline, LinearBaseline):
        return np.asarray(baseline.offset + baseline.slope * shifted_axis, dtype=np.float64)
    if isinstance(baseline, PolynomialBaseline):
        return np.asarray(
            np.polynomial.polynomial.polyval(shifted_axis, baseline.coefficients),
            dtype=np.float64,
        )
    return np.asarray(
        baseline.offset + baseline.amplitude * np.exp(baseline.rate * shifted_axis),
        dtype=np.float64,
    )


def _sample_noise(
    shape: tuple[int, ...],
    noise: GaussianNoise,
    *,
    rng: np.random.Generator | None,
) -> Array1D:
    if noise.sigma == 0.0:
        return np.zeros(shape, dtype=np.float64)

    generator = rng if rng is not None else np.random.default_rng(noise.seed)
    return np.asarray(generator.normal(loc=0.0, scale=noise.sigma, size=shape), dtype=np.float64)


def _append_generation_step(
    provenance: Provenance,
    *,
    name: str,
    parameters: dict[str, object],
) -> Provenance:
    return provenance.append(
        ProvenanceStep(
            name=name,
            parameters=parameters,
            description="Generated synthetic Raman data from explicit peak and noise parameters.",
        )
    )


__all__ = [
    "ExponentialBaseline",
    "GaussianNoise",
    "LinearBaseline",
    "PeakComponent",
    "PeakModel",
    "PolynomialBaseline",
    "SyntheticSpectrumConfig",
    "generate_collection",
    "generate_image",
    "generate_spectrum",
]
