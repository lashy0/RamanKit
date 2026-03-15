from __future__ import annotations

from typing import cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def resolve_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return a figure/axes pair, creating one when needed."""

    if ax is not None:
        return cast(Figure, ax.figure), ax
    figure, axes = plt.subplots()
    return figure, axes


def apply_axis_labels(
    axes: Axes,
    spectral_axis_name: str | None,
    spectral_unit: str | None,
    xlabel: str | None,
    ylabel: str,
) -> None:
    """Apply consistent spectral-axis and intensity labels to axes."""

    axes.set_xlabel(format_spectral_axis_label(spectral_axis_name, spectral_unit, xlabel))
    axes.set_ylabel(ylabel)


def format_spectral_axis_label(
    spectral_axis_name: str | None,
    spectral_unit: str | None,
    xlabel: str | None,
) -> str:
    """Return a human-readable label for the spectral axis."""

    if xlabel is not None:
        return xlabel
    if spectral_axis_name is None and spectral_unit is None:
        return "Spectral axis"
    if spectral_axis_name is None:
        return f"Spectral axis ({spectral_unit})"
    if spectral_unit is None:
        return spectral_axis_name
    return f"{spectral_axis_name} ({spectral_unit})"
