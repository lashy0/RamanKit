from __future__ import annotations

import numpy as np
import pytest

from ramankit import Metadata, RamanImage


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
