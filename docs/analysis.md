# Analysis

`ramankit.analysis` provides spectral decomposition methods for `SpectrumCollection`
and `RamanImage` data.

## PCA

Principal component analysis finds orthogonal directions of maximum variance.

```python
import ramankit.analysis as ra

result = ra.pca(collection, n_components=3)
```

The result exposes:

- `result.components` — a `SpectrumCollection` of `n_components` component spectra
  sharing the input spectral axis
- `result.scores` — mixing weights shaped `(n_spectra, k)` for a collection or
  `(height, width, k)` for a Raman image
- `result.explained_variance_ratio` — fraction of total variance explained by each
  component, shape `(k,)`
- `result.n_components` — number of components extracted
- `result.input_shape` — non-spectral shape of the input, `(n_spectra,)` or `(height, width)`

PCA is useful for noise reduction, identifying dominant spectral patterns, and
choosing how many components to use for downstream methods.

## NMF

Non-negative matrix factorization constrains components and scores to be
non-negative, which often produces physically interpretable pure-component spectra.

```python
result = ra.nmf(collection, n_components=3)
result = ra.nmf(collection, n_components=3, init="nndsvd", max_iter=500, random_state=42)
```

**All intensity values must be non-negative.** A `ValueError` is raised when the
input contains negative values. Apply baseline correction and verify there are no
negative residuals before calling `nmf`.

The result exposes the same fields as `PCAResult` plus:

- `result.reconstruction_error` — scalar residual `‖X − WH‖` after fitting

Parameters forwarded to sklearn:

| Parameter | Default | Meaning |
|---|---|---|
| `init` | `None` | Initialisation method; `None` lets sklearn choose |
| `max_iter` | `200` | Maximum solver iterations |
| `random_state` | `None` | Seed for reproducibility |

## ICA

Independent component analysis maximises statistical independence of the recovered
components. It uses sklearn's FastICA algorithm.

```python
result = ra.ica(collection, n_components=3, random_state=42)
result = ra.ica(image, n_components=3, max_iter=500, tol=1e-3, random_state=0)
```

The result exposes the same fields as `PCAResult` plus:

- `result.mixing_matrix` — the mixing matrix of shape `(n_points, k)` that maps
  components back to the original spectral space

Parameters forwarded to sklearn:

| Parameter | Default | Meaning |
|---|---|---|
| `max_iter` | `200` | Maximum iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `random_state` | `None` | Seed for reproducibility |

## Working with RamanImage

All three methods accept a `RamanImage` directly. Scores are returned with the
spatial shape preserved so maps can be visualised immediately.

```python
import ramankit.analysis as ra
import ramankit.plotting.spectra as rps
import ramankit.plotting.maps as rpm
import matplotlib.pyplot as plt

result = ra.nmf(image, n_components=3)

# Plot component spectra
rps.plot_collection(result.components, labels=["C1", "C2", "C3"])

# Plot spatial distribution map for each component
fig, axes = plt.subplots(1, 3)
for i, ax in enumerate(axes):
    ax.imshow(result.scores[:, :, i], cmap="viridis")
    ax.set_title(f"Component {i + 1}")
plt.show()
```

## Provenance

Every result records the decomposition method and parameters in the provenance of
the returned component collection.

```python
result = ra.pca(collection, n_components=2)
step = result.components.provenance.steps[0]
print(step.name)        # "pca"
print(step.parameters)  # {"n_components": 2}
```

## Validation

All three functions share the same validation contract:

- `n_components` must be a positive integer
- `n_components` must not exceed `min(n_spectra, n_points)`
- Input must be `SpectrumCollection` or `RamanImage` — single `Spectrum` is rejected
- NMF additionally requires all intensity values to be non-negative
