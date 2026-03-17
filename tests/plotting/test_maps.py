from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import ramankit.plotting.maps as rpm
from ramankit import RamanImage


def test_plot_image_band_by_index_returns_axes_image() -> None:
    """Plot one Raman image slice selected by its spectral index."""

    image = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]],
        ],
    )

    figure, axes = rpm.plot_image_band(image, index=1)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert len(axes.images) == 1
    plotted = axes.images[0].get_array()
    assert np.array_equal(plotted, np.array([[2.0, 4.0], [6.0, 8.0]]))
    plt.close(figure)

def test_plot_image_band_by_shift_uses_nearest_band() -> None:
    """Select the closest spectral coordinate when plotting by shift value."""

    image = RamanImage(
        axis=[100.0, 200.0, 310.0],
        intensity=[
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]],
        ],
    )

    figure, axes = rpm.plot_image_band(image, shift=295.0)

    plotted = axes.images[0].get_array()
    assert np.array_equal(plotted, np.array([[3.0, 5.0], [7.0, 9.0]]))
    plt.close(figure)

def test_plot_image_band_raises_when_selector_is_ambiguous() -> None:
    """Reject missing or conflicting band selectors."""

    image = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]],
    )

    with pytest.raises(ValueError, match="exactly one of index or shift"):
        rpm.plot_image_band(image)

    with pytest.raises(ValueError, match="exactly one of index or shift"):
        rpm.plot_image_band(image, index=1, shift=200.0)

def test_plot_image_band_raises_for_invalid_index() -> None:
    """Reject spectral indices that fall outside the image axis length."""

    image = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]],
    )

    with pytest.raises(ValueError, match="Expected index to be between"):
        rpm.plot_image_band(image, index=3)
