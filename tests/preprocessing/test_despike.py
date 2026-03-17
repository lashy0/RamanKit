from __future__ import annotations

import numpy as np
import pytest

import ramankit.preprocessing as pp
from ramankit import Metadata, Spectrum


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
