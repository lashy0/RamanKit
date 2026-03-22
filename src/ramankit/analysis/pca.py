from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA  # type: ignore[import-untyped]

from ramankit.analysis._common import build_components, reshape_scores, validate_and_flatten
from ramankit.analysis._results import PCAResult
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage


def pca(
    data: SpectrumCollection | RamanImage,
    *,
    n_components: int,
) -> PCAResult:
    """Decompose spectral data using principal component analysis.

    Parameters
    ----------
    data:
        Spectral data to decompose.
    n_components:
        Number of principal components to extract.  Must be at least 1
        and at most ``min(n_spectra, n_points)``.

    Returns
    -------
    PCAResult
        Components as a :class:`SpectrumCollection` sharing the input
        spectral axis, scores shaped to match the input geometry, and
        the explained variance ratio for each component.
    """
    rows, leading_shape = validate_and_flatten(data, n_components)

    model = SklearnPCA(n_components=n_components)
    scores_2d = model.fit_transform(rows)

    return PCAResult(
        components=build_components(
            data,
            model.components_,
            "pca",
            {"n_components": n_components},
        ),
        scores=reshape_scores(scores_2d, leading_shape),
        explained_variance_ratio=np.asarray(
            model.explained_variance_ratio_,
            dtype=np.float64,
        ),
        n_components=n_components,
        input_shape=leading_shape,
    )
