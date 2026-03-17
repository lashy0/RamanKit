from __future__ import annotations

import numpy as np

import ramankit.preprocessing as pp
from ramankit import Provenance, ProvenanceStep, Spectrum


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
