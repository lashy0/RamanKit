from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from ramankit.core._nd import flatten_spectral_rows
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import Metadata, Provenance, ProvenanceStep


def validate_and_flatten(
    data: SpectrumCollection | RamanImage,
    n_components: int,
) -> tuple[npt.NDArray[np.float64], tuple[int, ...]]:
    """Validate inputs and flatten spectral data to a 2-D matrix.

    Returns
    -------
    rows
        Array of shape ``(n_samples, n_points)``.
    leading_shape
        Non-spectral dimensions of the input: ``(n_spectra,)`` for a
        collection, ``(height, width)`` for an image.
    """
    if not isinstance(data, (SpectrumCollection, RamanImage)):
        raise TypeError(f"Expected SpectrumCollection or RamanImage, got {type(data).__name__}.")
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"Expected n_components to be a positive integer; got {n_components!r}.")

    batch = flatten_spectral_rows(data)
    rows = np.asarray(batch.rows, dtype=np.float64)
    leading_shape = batch.leading_shape

    n_samples = rows.shape[0]
    if n_components > n_samples:
        raise ValueError(
            f"n_components ({n_components}) must not exceed the number of spectra ({n_samples})."
        )
    n_features = rows.shape[1]
    if n_components > n_features:
        raise ValueError(
            f"n_components ({n_components}) must not exceed the number of "
            f"spectral points ({n_features})."
        )
    return rows, leading_shape


def reshape_scores(
    scores_2d: npt.NDArray[np.float64],
    leading_shape: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Reshape flat scores ``(n_samples, k)`` back to ``(*leading_shape, k)``."""
    return np.asarray(
        scores_2d.reshape(*leading_shape, scores_2d.shape[-1]),
        dtype=np.float64,
    )


def build_components(
    data: SpectrumCollection | RamanImage,
    components_2d: npt.NDArray[np.float64],
    method_name: str,
    parameters: Mapping[str, object],
) -> SpectrumCollection:
    """Build a SpectrumCollection of component spectra with provenance."""
    step = ProvenanceStep(name=method_name, parameters=parameters)
    provenance = Provenance().append(step)

    return SpectrumCollection(
        axis=data.axis,
        intensity=np.asarray(components_2d, dtype=np.float64),
        spectral_axis_name=data.spectral_axis_name,
        spectral_unit=data.spectral_unit,
        metadata=Metadata(),
        provenance=provenance,
    )
