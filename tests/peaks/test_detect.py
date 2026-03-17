from __future__ import annotations

import numpy as np
import pytest

import ramankit.peaks.detect as rpd
from ramankit import RamanImage, Spectrum, SpectrumCollection
from tests._synthetic_helpers import gaussian


def test_find_peaks_returns_typed_result_with_positions() -> None:
    """Detect multiple peaks and expose them in spectral coordinates."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = (
        gaussian(axis, amplitude=5.0, center=130.0, width=4.0, offset=0.2)
        + gaussian(axis, amplitude=3.5, center=230.0, width=6.0)
    )
    spectrum = Spectrum(axis=axis, intensity=intensity)

    result = rpd.find_peaks(spectrum, prominence=0.5, width=1.0, distance=200.0)

    assert isinstance(result, rpd.PeakDetectionResult)
    assert len(result) == 2
    assert np.allclose(result.positions, np.array([130.0, 230.0]), atol=0.5)
    assert np.all(result.heights > 0.0)
    assert result[0].prominence is not None
    assert result[0].width is not None

def test_find_peaks_returns_empty_result_when_no_peaks_exist() -> None:
    """Return an empty typed result when no local maxima satisfy the filters."""

    spectrum = Spectrum(axis=np.linspace(100.0, 300.0, 201), intensity=np.zeros(201))

    result = rpd.find_peaks(spectrum, prominence=1.0)

    assert isinstance(result, rpd.PeakDetectionResult)
    assert len(result) == 0
    assert result.indices.size == 0

def test_find_peaks_batch_returns_one_result_per_collection_spectrum() -> None:
    """Detect peaks for every spectrum in a collection in collection order."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = np.vstack(
        [
            gaussian(axis, amplitude=4.0, center=140.0, width=5.0, offset=0.1),
            gaussian(axis, amplitude=5.0, center=220.0, width=7.0, offset=0.2),
        ]
    )
    collection = SpectrumCollection(axis=axis, intensity=intensity)

    result = rpd.find_peaks_batch(collection, prominence=0.5, width=1.0)

    assert len(result) == 2
    assert result[0][0].position == pytest.approx(140.0, abs=0.5)
    assert result[1][0].position == pytest.approx(220.0, abs=0.5)

def test_find_peaks_batch_flattens_raman_image_in_row_major_order() -> None:
    """Detect peaks for every Raman image pixel in the flatten order used by RamanImage."""

    axis = np.linspace(100.0, 300.0, 1001)
    intensity = np.stack(
        [
            np.stack(
                [
                    gaussian(axis, amplitude=4.0, center=130.0, width=5.0, offset=0.1),
                    np.zeros_like(axis),
                ]
            ),
            np.stack(
                [
                    gaussian(axis, amplitude=5.0, center=210.0, width=6.0, offset=0.2),
                    gaussian(axis, amplitude=3.5, center=250.0, width=4.0, offset=0.2),
                ]
            ),
        ]
    )
    image = RamanImage(axis=axis, intensity=intensity)

    result = rpd.find_peaks_batch(image, prominence=0.5, width=1.0)

    assert len(result) == 4
    assert result[0][0].position == pytest.approx(130.0, abs=0.5)
    assert len(result[1]) == 0
    assert result[2][0].position == pytest.approx(210.0, abs=0.5)
    assert result[3][0].position == pytest.approx(250.0, abs=0.5)
