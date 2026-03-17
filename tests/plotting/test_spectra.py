from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import ramankit.plotting.spectra as rps
from ramankit import Spectrum, SpectrumCollection


def test_plot_spectrum_returns_figure_and_axes() -> None:
    """Plot one spectrum as a single matplotlib line."""

    spectrum = Spectrum(
        axis=[100.0, 200.0, 300.0],
        intensity=[1.0, 2.0, 3.0],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    figure, axes = rps.plot_spectrum(spectrum)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.lines) == 1
    assert axes.get_xlabel() == "raman_shift (cm^-1)"
    assert axes.get_ylabel() == "Intensity"
    plt.close(figure)

def test_plot_collection_returns_one_line_per_spectrum() -> None:
    """Overlay all collection spectra on one shared axes instance."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    figure, axes = rps.plot_collection(collection)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.lines) == collection.n_spectra
    plt.close(figure)

def test_plot_collection_raises_for_label_mismatch() -> None:
    """Reject labels whose count does not match the collection size."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )

    with pytest.raises(ValueError, match="Expected labels to match"):
        rps.plot_collection(collection, labels=["only-one-label"])
