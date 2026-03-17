from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
import ramankit.synthetic as rsyn
from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection
from tests._test_helpers import apply_collection_row_by_row, apply_image_pixel_by_pixel

BASELINE_STEPS = [
    ("asls", pp.baseline.ASLS),
    ("iasls", pp.baseline.IASLS),
    ("airpls", pp.baseline.AIRPLS),
    ("arpls", pp.baseline.ARPLS),
    ("drpls", pp.baseline.DRPLS),
    ("iarpls", pp.baseline.IARPLS),
    ("aspls", pp.baseline.ASPLS),
    ("poly", pp.baseline.Poly),
    ("modpoly", pp.baseline.ModPoly),
    ("penalized_poly", pp.baseline.PenalisedPoly),
    ("imodpoly", pp.baseline.IModPoly),
    ("goldindec", pp.baseline.Goldindec),
    ("irsqr", pp.baseline.IRSQR),
    ("corner_cutting", pp.baseline.CornerCutting),
    ("fabc", pp.baseline.FABC),
]

SYNTHETIC_BASELINE_CASES = [
    (
        "polynomial",
        rsyn.SyntheticSpectrumConfig(
            peaks=(
                rsyn.PeakComponent(amplitude=3.0, center=210.0, width=12.0),
                rsyn.PeakComponent(amplitude=2.2, center=300.0, width=16.0),
            ),
            baseline=rsyn.PolynomialBaseline(coefficients=(0.6, 3e-3, 1e-5)),
        ),
    ),
    (
        "exponential",
        rsyn.SyntheticSpectrumConfig(
            peaks=(
                rsyn.PeakComponent(amplitude=3.0, center=210.0, width=12.0),
                rsyn.PeakComponent(amplitude=2.2, center=300.0, width=16.0),
            ),
            baseline=rsyn.ExponentialBaseline(amplitude=0.5, rate=4e-3, offset=0.6),
        ),
    ),
]

@pytest.mark.parametrize(("method", "step_cls"), BASELINE_STEPS)
@pytest.mark.parametrize(("case_name", "config"), SYNTHETIC_BASELINE_CASES)
def test_baseline_steps_apply_preserve_spectrum_metadata_and_type(
    method: str,
    step_cls: type[pp.PreprocessingStep],
    case_name: str,
    config: rsyn.SyntheticSpectrumConfig,
) -> None:
    """Return a corrected spectrum while preserving metadata and provenance."""

    metadata = Metadata(sample="sample-1")
    provenance = Provenance(steps=(ProvenanceStep(name="load"),))
    axis = np.linspace(100.0, 400.0, 121)
    spectrum = rsyn.generate_spectrum(
        axis=axis,
        config=config,
        metadata=metadata,
        provenance=provenance,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    corrected = step_cls().apply(spectrum)

    assert isinstance(corrected, Spectrum)
    assert corrected.metadata == metadata
    assert corrected.axis.shape == spectrum.axis.shape
    assert corrected.provenance.steps[-1].name == "baseline_correct"
    assert corrected.provenance.steps[-1].parameters["method"] == method
    assert np.all(np.isfinite(corrected.intensity)), case_name
    assert not np.allclose(corrected.intensity, spectrum.intensity)

@pytest.mark.parametrize(
    "step",
    [
        pp.baseline.ASLS(),
        pp.baseline.ModPoly(),
        pp.baseline.IRSQR(),
        pp.baseline.FABC(),
    ],
)
def test_selected_baseline_steps_apply_preserve_raman_image_shape(
    step: pp.PreprocessingStep,
) -> None:
    """Apply representative baseline methods across the last image axis."""

    axis = np.linspace(100.0, 400.0, 9)
    image = RamanImage(
        axis=axis,
        intensity=np.stack([0.01 * axis + 1.0, 0.02 * axis + 2.0]).reshape(1, 2, 9),
    )

    corrected = step.apply(image)

    assert isinstance(corrected, RamanImage)
    assert corrected.intensity.shape == image.intensity.shape
    assert corrected.provenance.steps[-1].name == "baseline_correct"

@pytest.mark.parametrize(
    ("step", "expected_parameters"),
    [
        (pp.baseline.DRPLS(eta=0.25), {"eta": 0.25}),
        (pp.baseline.IASLS(lam_1=1e-3), {"lam_1": 1e-3}),
        (pp.baseline.IModPoly(num_std=2.0), {"num_std": 2.0}),
        (
            pp.baseline.FABC(num_std=2.5, pad_kwargs={"mode": "edge"}),
            {"num_std": 2.5, "pad_kwargs": {"mode": "edge"}},
        ),
    ],
)
def test_baseline_steps_record_configured_parameters(
    step: pp.PreprocessingStep,
    expected_parameters: dict[str, object],
) -> None:
    """Record configured baseline parameters in provenance."""

    spectrum = Spectrum(
        axis=np.linspace(100.0, 400.0, 11),
        intensity=np.linspace(1.0, 2.0, 11)
        + np.array([0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0]),
    )

    corrected = step.apply(spectrum)

    parameters = corrected.provenance.steps[-1].parameters
    for key, value in expected_parameters.items():
        assert parameters[key] == value

def test_baseline_steps_summarize_array_parameters_in_provenance() -> None:
    """Store array-shaped baseline parameters as compact provenance summaries."""

    spectrum = Spectrum(
        axis=np.linspace(100.0, 400.0, 11),
        intensity=np.linspace(1.0, 2.0, 11),
    )
    step = pp.baseline.ASLS(weights=np.linspace(1.0, 2.0, 11))

    corrected = step.apply(spectrum)

    assert corrected.provenance.steps[-1].parameters["weights"] == {
        "kind": "ndarray",
        "shape": [11],
        "dtype": "float64",
    }

@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: pp.baseline.ASLS(lam=-1.0), "lam"),
        (lambda: pp.baseline.ASLS(p=2.0), "p"),
        (lambda: pp.baseline.Poly(poly_order=-1), "poly_order"),
        (lambda: pp.baseline.IRSQR(quantile=5.0), "quantile"),
        (lambda: pp.baseline.FABC(num_std=0.0), "num_std"),
        (lambda: pp.baseline.PenalisedPoly(cost_function="bad"), "cost_function"),
    ],
)
def test_baseline_steps_validate_scalar_parameters(
    factory,
    message: str,
) -> None:
    """Raise RamanKit errors for invalid scalar baseline parameters."""

    with pytest.raises(ValueError, match=message):
        factory()

def test_baseline_steps_validate_array_parameter_shapes() -> None:
    """Reject axis-shaped parameters whose length does not match the axis."""

    spectrum = Spectrum(
        axis=np.linspace(100.0, 400.0, 11),
        intensity=np.linspace(1.0, 2.0, 11),
    )

    with pytest.raises(ValueError, match="weights"):
        pp.baseline.ASLS(weights=np.ones(5)).apply(spectrum)

    with pytest.raises(ValueError, match="alpha"):
        pp.baseline.ASPLS(alpha=np.ones(5)).apply(spectrum)


@pytest.mark.parametrize(
    "step",
    [
        pp.baseline.ASLS(lam=1e5, p=1e-2, max_iter=20),
        pp.baseline.AIRPLS(lam=1e5, max_iter=20),
        pp.baseline.Poly(poly_order=2),
    ],
)
def test_baseline_batch_path_matches_row_by_row_for_collection(step: pp.PreprocessingStep) -> None:
    """Batch baseline correction should match per-spectrum application on collections."""

    axis = np.linspace(100.0, 400.0, 81)
    positions = np.linspace(0.0, 1.0, axis.shape[0])
    collection = SpectrumCollection(
        axis=axis,
        intensity=np.array(
            [
                0.4 + 0.5 * positions + 3.0 * np.exp(-((axis - 220.0) ** 2) / (2 * 10.0**2)),
                0.6 + 0.3 * positions + 2.0 * np.exp(-((axis - 300.0) ** 2) / (2 * 14.0**2)),
                0.5
                + 0.4 * positions
                + 1.5 * np.exp(-((axis - 180.0) ** 2) / (2 * 8.0**2))
                + 1.2 * np.exp(-((axis - 330.0) ** 2) / (2 * 12.0**2)),
            ],
            dtype=np.float64,
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    corrected = step.apply(collection)
    expected = apply_collection_row_by_row(step, collection)

    assert isinstance(corrected, SpectrumCollection)
    assert corrected.intensity.shape == collection.intensity.shape
    assert np.array_equal(corrected.axis, collection.axis)
    assert corrected.provenance.steps[-1].name == "baseline_correct"
    assert corrected.provenance.steps[-1].parameters["method"] == step.method_name
    assert np.allclose(corrected.intensity, expected.intensity)


@pytest.mark.parametrize(
    "step",
    [
        pp.baseline.ASLS(lam=1e5, p=1e-2, max_iter=20),
        pp.baseline.AIRPLS(lam=1e5, max_iter=20),
        pp.baseline.Poly(poly_order=2),
    ],
)
def test_baseline_batch_path_matches_row_by_row_for_image(step: pp.PreprocessingStep) -> None:
    """Batch baseline correction should match per-pixel application on Raman images."""

    axis = np.linspace(100.0, 400.0, 81)
    positions = np.linspace(0.0, 1.0, axis.shape[0])
    image = RamanImage(
        axis=axis,
        intensity=np.array(
            [
                [
                    0.4 + 0.5 * positions + 3.0 * np.exp(-((axis - 220.0) ** 2) / (2 * 10.0**2)),
                    0.6 + 0.3 * positions + 2.0 * np.exp(-((axis - 300.0) ** 2) / (2 * 14.0**2)),
                ],
                [
                    0.5
                    + 0.4 * positions
                    + 1.5 * np.exp(-((axis - 180.0) ** 2) / (2 * 8.0**2))
                    + 1.2 * np.exp(-((axis - 330.0) ** 2) / (2 * 12.0**2)),
                    0.45
                    + 0.35 * positions
                    + 2.3 * np.exp(-((axis - 260.0) ** 2) / (2 * 11.0**2)),
                ],
            ],
            dtype=np.float64,
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    corrected = step.apply(image)
    expected = apply_image_pixel_by_pixel(step, image)

    assert isinstance(corrected, RamanImage)
    assert corrected.intensity.shape == image.intensity.shape
    assert np.array_equal(corrected.axis, image.axis)
    assert corrected.provenance.steps[-1].name == "baseline_correct"
    assert corrected.provenance.steps[-1].parameters["method"] == step.method_name
    assert np.allclose(corrected.intensity, expected.intensity)
