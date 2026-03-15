from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ramankit.core.collection import SpectrumCollection
from ramankit.core.spectrum import Spectrum


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

    figure, axes = _resolve_axes(ax)
    axes.plot(spectrum.axis, spectrum.intensity, label=label, color=color)
    _apply_axis_labels(axes, spectrum.spectral_axis_name, spectrum.spectral_unit, xlabel, ylabel)
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

    figure, axes = _resolve_axes(ax)
    for index, intensity in enumerate(collection.intensity):
        line_label = None if labels is None else labels[index]
        axes.plot(collection.axis, intensity, alpha=alpha, label=line_label)

    _apply_axis_labels(
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


def _resolve_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return cast(Figure, ax.figure), ax
    figure, axes = plt.subplots()
    return figure, axes


def _apply_axis_labels(
    axes: Axes,
    spectral_axis_name: str | None,
    spectral_unit: str | None,
    xlabel: str | None,
    ylabel: str,
) -> None:
    axes.set_xlabel(_format_spectral_axis_label(spectral_axis_name, spectral_unit, xlabel))
    axes.set_ylabel(ylabel)


def _format_spectral_axis_label(
    spectral_axis_name: str | None,
    spectral_unit: str | None,
    xlabel: str | None,
) -> str:
    if xlabel is not None:
        return xlabel
    if spectral_axis_name is None and spectral_unit is None:
        return "Spectral axis"
    if spectral_axis_name is None:
        return f"Spectral axis ({spectral_unit})"
    if spectral_unit is None:
        return spectral_axis_name
    return f"{spectral_axis_name} ({spectral_unit})"
