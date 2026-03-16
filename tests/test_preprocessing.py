from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection

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


@pytest.mark.parametrize(("method", "step_cls"), BASELINE_STEPS)
def test_baseline_steps_apply_preserve_spectrum_metadata_and_type(
    method: str,
    step_cls: type[pp.PreprocessingStep],
) -> None:
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

    corrected = step_cls().apply(spectrum)

    assert isinstance(corrected, Spectrum)
    assert corrected.metadata == metadata
    assert corrected.axis.shape == spectrum.axis.shape
    assert corrected.provenance.steps[-1].name == "baseline_correct"
    assert corrected.provenance.steps[-1].parameters["method"] == method
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




