from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Provenance, ProvenanceStep, Spectrum, SpectrumCollection
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class _ShiftAxis(pp.AxisTransformStep):
    function_name = "shift_axis"
    method_name = "constant"

    shift: float

    def parameters(self) -> dict[str, object]:
        return {"shift": self.shift}

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        return axis + self.shift, intensity


@dataclass(frozen=True, slots=True)
class _VaryAxisPerSpectrum(pp.AxisTransformStep):
    function_name = "vary_axis"
    method_name = "row_offset"

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        return axis + float(intensity[0]), intensity


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


def test_axis_changing_steps_use_explicit_axis_transform_base_class() -> None:
    """Expose axis-changing built-ins through the explicit AxisTransformStep type."""

    assert issubclass(pp.resample.Linear, pp.AxisTransformStep)
    assert issubclass(pp.misc.Cropper, pp.AxisTransformStep)
    assert pp.resample.Linear.apply is pp.AxisTransformStep.apply
    assert pp.misc.Cropper.apply is pp.AxisTransformStep.apply


def test_axis_preserving_step_keeps_axis_exactly() -> None:
    """Preserve the spectral axis exactly for ordinary preprocessing steps."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 3.0, 2.0]),
    )

    processed = pp.normalization.Max().apply(spectrum)

    assert np.array_equal(processed.axis, spectrum.axis)
    assert processed.axis_direction == spectrum.axis_direction


def test_pipeline_accepts_mixed_axis_transform_and_axis_preserving_steps() -> None:
    """Compose explicit axis transforms with ordinary preprocessing steps."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 3.0, 2.0]),
    )
    pipeline = pp.Pipeline(
        [
            _ShiftAxis(shift=25.0),
            pp.normalization.Max(),
        ]
    )

    processed = pipeline.apply(spectrum)

    assert np.array_equal(processed.axis, np.array([125.0, 225.0, 325.0]))
    assert np.allclose(processed.intensity, np.array([1.0 / 3.0, 1.0, 2.0 / 3.0]))
    assert [step.name for step in processed.provenance.steps[-2:]] == ["shift_axis", "normalize"]


def test_axis_transform_row_fallback_requires_shared_axis_for_collection() -> None:
    """Reject row-wise axis transforms that produce different axes per spectrum."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    with pytest.raises(ValueError, match="same axis for every spectrum"):
        _VaryAxisPerSpectrum().apply(collection)


# --- SpectrumCollection pipeline tests ---


def test_pipeline_applies_axis_preserving_steps_to_collection() -> None:
    """Apply axis-preserving preprocessing steps to a SpectrumCollection."""

    collection = SpectrumCollection(
        axis=np.linspace(100.0, 400.0, 9),
        intensity=np.array([
            [5.0, 6.0, 8.0, 15.0, 30.0, 16.0, 9.0, 6.0, 5.0],
            [10.0, 12.0, 16.0, 30.0, 60.0, 32.0, 18.0, 12.0, 10.0],
        ]),
    )
    pipeline = pp.Pipeline([pp.normalization.Vector()])

    processed = pipeline.apply(collection)

    assert isinstance(processed, SpectrumCollection)
    assert processed.intensity.shape == (2, 9)
    assert np.array_equal(processed.axis, collection.axis)
    assert processed.provenance.steps[-1].name == "normalize"


def test_pipeline_applies_axis_changing_step_to_collection() -> None:
    """Apply an axis-changing step to a SpectrumCollection."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([[1.0, 2.0, 3.0, 2.0, 1.0], [2.0, 4.0, 6.0, 4.0, 2.0]]),
    )
    pipeline = pp.Pipeline([
        pp.misc.Cropper(lower_bound=200.0, upper_bound=400.0),
    ])

    processed = pipeline.apply(collection)

    assert isinstance(processed, SpectrumCollection)
    assert np.array_equal(processed.axis, np.array([200.0, 300.0, 400.0]))
    assert processed.intensity.shape == (2, 3)
    assert processed.provenance.steps[-1].name == "crop"


def test_pipeline_applies_mixed_steps_to_collection() -> None:
    """Apply a mix of axis-changing and axis-preserving steps to a collection."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 3.0, 2.0], [2.0, 6.0, 4.0]]),
    )
    pipeline = pp.Pipeline([
        _ShiftAxis(shift=25.0),
        pp.normalization.Max(),
    ])

    processed = pipeline.apply(collection)

    assert isinstance(processed, SpectrumCollection)
    assert np.array_equal(processed.axis, np.array([125.0, 225.0, 325.0]))
    assert np.allclose(processed.intensity[0], np.array([1.0 / 3.0, 1.0, 2.0 / 3.0]))
    assert [s.name for s in processed.provenance.steps[-2:]] == [
        "shift_axis",
        "normalize",
    ]


def test_pipeline_preserves_collection_metadata_and_semantics() -> None:
    """Preserve metadata, spectral_axis_name, and spectral_unit through pipeline."""

    from ramankit import Metadata

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 3.0, 2.0]]),
        metadata=Metadata(sample="polymer"),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    pipeline = pp.Pipeline([pp.normalization.Max()])

    processed = pipeline.apply(collection)

    assert processed.metadata.sample == "polymer"
    assert processed.spectral_axis_name == "raman_shift"
    assert processed.spectral_unit == "cm^-1"


def test_pipeline_crop_on_collection_reduces_shape() -> None:
    """Crop reduces the number of spectral points across all spectra."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        intensity=np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]),
    )
    pipeline = pp.Pipeline([pp.misc.Cropper(lower_bound=200.0, upper_bound=400.0)])

    processed = pipeline.apply(collection)

    assert processed.intensity.shape == (3, 3)
    assert np.array_equal(processed.axis, np.array([200.0, 300.0, 400.0]))
