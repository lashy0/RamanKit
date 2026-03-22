from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF as SklearnNMF  # type: ignore[import-untyped]

from ramankit.analysis._common import build_components, reshape_scores, validate_and_flatten
from ramankit.analysis._results import NMFResult
from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage


def nmf(
    data: SpectrumCollection | RamanImage,
    *,
    n_components: int,
    init: str | None = None,
    max_iter: int = 200,
    random_state: int | None = None,
) -> NMFResult:
    """Decompose spectral data using non-negative matrix factorization.

    Parameters
    ----------
    data:
        Spectral data to decompose.  All intensity values must be
        non-negative.
    n_components:
        Number of components to extract.
    init:
        Initialization method forwarded to sklearn NMF.  When ``None``
        sklearn chooses automatically.
    max_iter:
        Maximum number of solver iterations.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    NMFResult
        Components, scores, and the reconstruction error.

    Raises
    ------
    ValueError
        If *data* contains negative intensity values.
    """
    rows, leading_shape = validate_and_flatten(data, n_components)

    if np.any(rows < 0):
        raise ValueError("NMF requires non-negative intensity values.")

    model = SklearnNMF(
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
    )
    scores_2d = model.fit_transform(rows)

    parameters: dict[str, object] = {
        "n_components": n_components,
        "init": init,
        "max_iter": max_iter,
        "random_state": random_state,
    }

    return NMFResult(
        components=build_components(data, model.components_, "nmf", parameters),
        scores=reshape_scores(scores_2d, leading_shape),
        reconstruction_error=float(model.reconstruction_err_),
        n_components=n_components,
        input_shape=leading_shape,
    )
