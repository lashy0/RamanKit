from __future__ import annotations

from collections.abc import Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ramankit.core.collection import SpectrumCollection
from ramankit.core.spectrum import Spectrum
from ramankit.plotting._utils import apply_axis_labels, resolve_axes


def plot_spectrum(
    spectrum: Spectrum,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    color: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Intensity",
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one spectrum against its spectral axis."""

    figure, axes = resolve_axes(ax)
    axes.plot(spectrum.axis, spectrum.intensity, label=label, color=color)
    apply_axis_labels(axes, spectrum.spectral_axis_name, spectrum.spectral_unit, xlabel, ylabel)
    if title is not None:
        axes.set_title(title)
    if label is not None:
        axes.legend()
    if show:
        plt.show()
    return figure, axes


def plot_collection(
    collection: SpectrumCollection,
    *,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    alpha: float = 0.7,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Intensity",
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Overlay all spectra from one collection on a shared axes."""

    if labels is not None and len(labels) != collection.n_spectra:
        raise ValueError(
            "Expected labels to match the number of spectra in the collection; "
            f"got {len(labels)} labels for {collection.n_spectra} spectra."
        )

    figure, axes = resolve_axes(ax)
    for index, intensity in enumerate(collection.intensity):
        line_label = None if labels is None else labels[index]
        axes.plot(collection.axis, intensity, alpha=alpha, label=line_label)

    apply_axis_labels(
        axes,
        collection.spectral_axis_name,
        collection.spectral_unit,
        xlabel,
        ylabel,
    )
    if title is not None:
        axes.set_title(title)
    if labels is not None:
        axes.legend()
    if show:
        plt.show()
    return figure, axes
