from __future__ import annotations

from typing import cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ramankit.core.image import RamanImage


def plot_image_band(
    image: RamanImage,
    *,
    index: int | None = None,
    shift: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one spatial Raman map extracted from a hyperspectral image."""

    if (index is None) == (shift is None):
        raise ValueError("Expected exactly one of index or shift to be provided.")

    band_index = _resolve_band_index(image, index=index, shift=shift)
    figure, axes = _resolve_axes(ax)
    artist = axes.imshow(image.intensity[:, :, band_index], cmap=cmap)
    axes.set_xlabel("Column")
    axes.set_ylabel("Row")
    if title is not None:
        axes.set_title(title)
    if colorbar:
        figure.colorbar(artist, ax=axes)
    if show:
        plt.show()
    return figure, axes


def _resolve_band_index(
    image: RamanImage,
    *,
    index: int | None,
    shift: float | None,
) -> int:
    if index is not None:
        if index < 0 or index >= image.n_points:
            raise ValueError(
                f"Expected index to be between 0 and {image.n_points - 1}; got {index}."
            )
        return index

    assert shift is not None
    return int(abs(image.axis - shift).argmin())


def _resolve_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return cast(Figure, ax.figure), ax
    figure, axes = plt.subplots()
    return figure, axes
