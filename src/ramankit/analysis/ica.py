from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA as SklearnFastICA  # type: ignore[import-untyped]

from ramankit.analysis._common import build_components, reshape_scores, validate_and_flatten
from ramankit.analysis._results import ICAResult
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage


def ica(
    data: SpectrumCollection | RamanImage,
    *,
    n_components: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> ICAResult:
    """Decompose spectral data using independent component analysis.

    Parameters
    ----------
    data:
        Spectral data to decompose.
    n_components:
        Number of independent components to extract.
    max_iter:
        Maximum number of iterations.
    tol:
        Convergence tolerance for FastICA.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    ICAResult
        Components, scores, and the mixing matrix.
    """
    rows, leading_shape = validate_and_flatten(data, n_components)

    model = SklearnFastICA(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    scores_2d = model.fit_transform(rows)

    parameters: dict[str, object] = {
        "n_components": n_components,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state,
    }

    return ICAResult(
        components=build_components(data, model.components_, "ica", parameters),
        scores=reshape_scores(scores_2d, leading_shape),
        mixing_matrix=np.asarray(model.mixing_, dtype=np.float64),
        n_components=n_components,
        input_shape=leading_shape,
    )
