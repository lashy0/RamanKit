from __future__ import annotations

import numpy as np
import pytest

from ramankit import RamanImage, SpectrumCollection
from ramankit.core._nd import coerce_spectral_nd, flatten_spectral_rows, restore_spectral_rows


def test_coerce_spectral_nd_raises_for_last_dimension_mismatch() -> None:
    """The ND helper validates the spectral axis against the last intensity dimension."""

    with pytest.raises(ValueError, match="last dimension"):
        coerce_spectral_nd(
            axis=[100.0, 200.0],
            intensity=np.ones((2, 3)),
            ndim=2,
            container_name="SpectrumCollection",
        )


def test_flatten_spectral_rows_for_collection_returns_batch_rows() -> None:
    """Collections flatten to one row per spectrum while preserving leading shape."""

    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    batch = flatten_spectral_rows(collection)

    assert batch.leading_shape == (2,)
    assert batch.rows.shape == (2, 3)
    np.testing.assert_array_equal(batch.rows, collection.intensity)


def test_flatten_spectral_rows_for_image_returns_pixel_rows() -> None:
    """Images flatten all spatial pixels into a shared batch matrix."""

    image = RamanImage(
        axis=[100.0, 200.0],
        intensity=np.arange(12.0).reshape(2, 3, 2),
    )

    batch = flatten_spectral_rows(image)

    assert batch.leading_shape == (2, 3)
    assert batch.rows.shape == (6, 2)
    np.testing.assert_array_equal(batch.rows, image.intensity.reshape(6, 2))


def test_restore_spectral_rows_rebuilds_collection_shape() -> None:
    """The ND helper restores flattened rows back to a collection-shaped array."""

    rows = np.arange(6.0).reshape(2, 3)

    restored = restore_spectral_rows(rows, leading_shape=(2,))

    assert restored.shape == (2, 3)
    np.testing.assert_array_equal(restored, rows)


def test_restore_spectral_rows_rebuilds_image_shape() -> None:
    """The ND helper restores flattened rows back to an image-shaped array."""

    rows = np.arange(12.0).reshape(6, 2)

    restored = restore_spectral_rows(rows, leading_shape=(2, 3))

    assert restored.shape == (2, 3, 2)
    np.testing.assert_array_equal(restored, rows.reshape(2, 3, 2))
