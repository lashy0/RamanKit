from __future__ import annotations

import numpy as np
import pytest

from ramankit import Metadata, RamanImage, Spectrum


def test_raman_image_raises_for_axis_length_mismatch() -> None:
    """Reject images whose spectral axis length mismatches the cube depth."""

    with pytest.raises(ValueError, match="to match"):
        RamanImage(
            axis=[100.0, 200.0, 300.0],
            intensity=np.ones((2, 2, 2)),
        )

def test_raman_image_flatten_preserves_metadata_and_shape() -> None:
    """Flatten images without losing metadata or provenance continuity."""

    metadata = Metadata(sample="sample-1", extras={"laser_nm": 785})
    image = RamanImage(
        axis=[100.0, 200.0],
        intensity=np.arange(12.0).reshape(2, 3, 2),
        metadata=metadata,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )

    flattened = image.flatten()

    assert flattened.intensity.shape == (6, 2)
    assert flattened.metadata == metadata
    assert flattened.provenance.steps[-1].name == "flatten_image"


def test_image_spatial_shape() -> None:
    """spatial_shape returns (height, width)."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.ones((3, 4, 2)))
    assert image.spatial_shape == (3, 4)


def test_image_n_points() -> None:
    """n_points returns the spectral axis length."""
    image = RamanImage(axis=[1.0, 2.0, 3.0], intensity=np.ones((2, 2, 3)))
    assert image.n_points == 3


def test_image_n_pixels() -> None:
    """n_pixels returns height * width."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.ones((3, 4, 2)))
    assert image.n_pixels == 12


def test_image_pixel_returns_spectrum() -> None:
    """pixel() extracts the correct spectrum at (row, col)."""
    data = np.arange(24.0).reshape(2, 3, 4)
    image = RamanImage(axis=[1.0, 2.0, 3.0, 4.0], intensity=data)
    spec = image.pixel(1, 2)
    assert isinstance(spec, Spectrum)
    np.testing.assert_array_equal(spec.intensity, data[1, 2])


def test_image_pixel_preserves_metadata() -> None:
    """pixel() inherits metadata, spectral_axis_name, and spectral_unit."""
    meta = Metadata(sample="img-1")
    image = RamanImage(
        axis=[1.0, 2.0],
        intensity=np.ones((2, 2, 2)),
        metadata=meta,
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    spec = image.pixel(0, 0)
    assert spec.metadata == meta
    assert spec.spectral_axis_name == "raman_shift"
    assert spec.spectral_unit == "cm^-1"


def test_image_copy_is_independent() -> None:
    """Mutating the original array does not affect the copy."""
    raw = np.ones((2, 2, 3))
    image = RamanImage(axis=[1.0, 2.0, 3.0], intensity=raw)
    copied = image.copy()
    raw[0, 0, 0] = -999.0
    assert copied.intensity[0, 0, 0] == 1.0


def test_image_mean_returns_spectrum() -> None:
    """mean() reduces to the spatial mean with provenance."""
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    image = RamanImage(axis=[1.0, 2.0], intensity=data)
    result = image.mean()
    assert isinstance(result, Spectrum)
    np.testing.assert_allclose(result.intensity, [4.0, 5.0])
    assert result.provenance.steps[-1].name == "mean"


def test_image_sum_returns_spectrum() -> None:
    """sum() reduces to the spatial sum with provenance."""
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    image = RamanImage(axis=[1.0, 2.0], intensity=data)
    result = image.sum()
    assert isinstance(result, Spectrum)
    np.testing.assert_allclose(result.intensity, [16.0, 20.0])
    assert result.provenance.steps[-1].name == "sum"


def test_image_std_returns_spectrum() -> None:
    """std() reduces to the spatial std with provenance."""
    data = np.arange(8.0).reshape(2, 2, 2)
    image = RamanImage(axis=[1.0, 2.0], intensity=data)
    result = image.std()
    assert isinstance(result, Spectrum)
    np.testing.assert_allclose(result.intensity, np.std(data, axis=(0, 1)))
    assert result.provenance.steps[-1].name == "std"


def test_image_add_scalar() -> None:
    """Adding a scalar shifts all intensities."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.ones((2, 2, 2)))
    np.testing.assert_allclose(image.add(10.0).intensity, 11.0)


def test_image_subtract_image() -> None:
    """Subtracting a matching image gives element-wise difference."""
    a = RamanImage(axis=[1.0, 2.0], intensity=np.full((2, 2, 2), 5.0))
    b = RamanImage(axis=[1.0, 2.0], intensity=np.full((2, 2, 2), 3.0))
    np.testing.assert_allclose(a.subtract(b).intensity, 2.0)


def test_image_multiply_scalar() -> None:
    """Multiplying by a scalar scales all intensities."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.full((2, 2, 2), 3.0))
    np.testing.assert_allclose(image.multiply(2.0).intensity, 6.0)


def test_image_divide_scalar() -> None:
    """Dividing by a scalar scales all intensities."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.full((2, 2, 2), 8.0))
    np.testing.assert_allclose(image.divide(4.0).intensity, 2.0)


def test_image_dunder_operators() -> None:
    """Dunder operators +, -, *, / produce correct results."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.full((2, 2, 2), 10.0))
    np.testing.assert_allclose((image + 1.0).intensity, 11.0)
    np.testing.assert_allclose((image - 1.0).intensity, 9.0)
    np.testing.assert_allclose((image * 2.0).intensity, 20.0)
    np.testing.assert_allclose((image / 2.0).intensity, 5.0)


def test_image_arithmetic_preserves_metadata() -> None:
    """Arithmetic preserves metadata, spectral_axis_name, and spectral_unit."""
    image = RamanImage(
        axis=[1.0, 2.0],
        intensity=np.ones((2, 2, 2)),
        metadata=Metadata(sample="s1"),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    result = image + 1.0
    assert result.metadata == image.metadata
    assert result.spectral_axis_name == "raman_shift"
    assert result.spectral_unit == "cm^-1"


def test_image_descending_axis() -> None:
    """A descending axis is accepted and direction is recorded."""
    image = RamanImage(axis=[300.0, 200.0, 100.0], intensity=np.ones((2, 2, 3)))
    assert image.axis_direction == "descending"


def test_image_default_metadata_and_provenance() -> None:
    """Default metadata and provenance are non-None when not provided."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.ones((2, 2, 2)))
    assert image.metadata is not None
    assert image.provenance is not None
    assert image.metadata.sample is None
    assert image.provenance.steps == ()


def test_image_flatten_spatial_shape_in_provenance() -> None:
    """Flatten provenance parameters include height and width."""
    image = RamanImage(axis=[1.0, 2.0], intensity=np.ones((3, 4, 2)))
    params = image.flatten().provenance.steps[-1].parameters
    assert params["height"] == 3
    assert params["width"] == 4
