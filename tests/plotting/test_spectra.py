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


def test_plot_collection_stacked_returns_one_line_per_spectrum() -> None:
    """Each spectrum in the collection produces exactly one plotted line."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    figure, axes = rps.plot_collection_stacked(collection)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.lines) == collection.n_spectra
    plt.close(figure)


def test_plot_collection_stacked_auto_offset_separates_spectra() -> None:
    """Auto-offset places each spectrum above the one before it."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )

    figure, axes = rps.plot_collection_stacked(collection)

    first_ydata = axes.lines[0].get_ydata()
    second_ydata = axes.lines[1].get_ydata()
    assert float(second_ydata.min()) >= float(first_ydata.max())
    plt.close(figure)


def test_plot_collection_stacked_explicit_offset_is_applied() -> None:
    """An explicit offset value shifts each spectrum by that exact amount."""

    import numpy as np

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
    )

    figure, axes = rps.plot_collection_stacked(collection, offset=10.0)

    first_ydata = axes.lines[0].get_ydata()
    second_ydata = axes.lines[1].get_ydata()
    np.testing.assert_allclose(second_ydata - first_ydata, 10.0)
    plt.close(figure)


def test_plot_collection_stacked_single_spectrum() -> None:
    """A single-spectrum collection plots one unshifted line."""

    import numpy as np

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0]],
    )

    figure, axes = rps.plot_collection_stacked(collection)

    assert len(axes.lines) == 1
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, 2.0, 3.0])
    plt.close(figure)


def test_plot_collection_stacked_raises_for_label_mismatch() -> None:
    """Reject labels whose count does not match the collection size."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )

    with pytest.raises(ValueError, match="Expected labels to match"):
        rps.plot_collection_stacked(collection, labels=["only-one-label"])


def test_plot_collection_stacked_raises_for_negative_offset() -> None:
    """Reject a negative offset."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )

    with pytest.raises(ValueError, match="offset must be non-negative"):
        rps.plot_collection_stacked(collection, offset=-5.0)


def test_plot_collection_stacked_with_labels_produces_legend() -> None:
    """Providing labels attaches a legend to the axes."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )

    figure, axes = rps.plot_collection_stacked(collection, labels=["sample-A", "sample-B"])

    assert axes.get_legend() is not None
    plt.close(figure)


def test_plot_collection_stacked_accepts_existing_axes() -> None:
    """When ax is provided, the function draws into it rather than creating one."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    )
    external_figure, external_axes = plt.subplots()

    figure, axes = rps.plot_collection_stacked(collection, ax=external_axes)

    assert axes is external_axes
    assert figure is external_figure
    plt.close(figure)


def test_plot_collection_stacked_all_zero_intensities() -> None:
    """All-zero intensities produce an auto-offset of zero without error."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    figure, axes = rps.plot_collection_stacked(collection)

    assert len(axes.lines) == 2
    plt.close(figure)


def test_plot_collection_stacked_applies_axis_labels() -> None:
    """Axis labels are derived from collection metadata when not overridden."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    figure, axes = rps.plot_collection_stacked(collection)

    assert axes.get_xlabel() == "raman_shift (cm^-1)"
    assert axes.get_ylabel() == "Intensity"
    plt.close(figure)
