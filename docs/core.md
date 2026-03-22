# Core models

RamanKit has three core data containers:

- `Spectrum` for one spectrum
- `SpectrumCollection` for a batch of spectra sharing one spectral axis
- `RamanImage` for a hyperspectral Raman image with shape `(height, width, n_points)`

## Shared principles

All core containers:

- validate that spectral axes are one-dimensional and strictly monotonic
- preserve metadata and provenance explicitly
- avoid hidden in-place mutation
- keep spectral axis names and units explicit when provided

## `Spectrum`

Use `Spectrum` for a single axis/intensity pair.

```python
from ramankit import Spectrum

spectrum = Spectrum(
    axis=[100.0, 200.0, 300.0],
    intensity=[3.0, 5.0, 4.0],
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)
```

## `SpectrumCollection`

Use `SpectrumCollection` when multiple spectra share the same spectral axis.

```python
from ramankit import SpectrumCollection

collection = SpectrumCollection(
    axis=[100.0, 200.0, 300.0],
    intensity=[
        [3.0, 5.0, 4.0],
        [2.0, 4.0, 6.0],
    ],
)
```

Collection reductions return a `Spectrum`:

```python
mean_spectrum = collection.mean()
std_spectrum = collection.std()
```

Collections can also be exported to NumPy explicitly for external analysis or
machine-learning workflows:

```python
export = collection.to_numpy()

export.intensity  # shape: (n_spectra, n_points)
export.axis       # shape: (n_points,)
```

Set `copy=False` to share memory with the underlying collection arrays instead
of creating detached copies.

## `RamanImage`

Use `RamanImage` for hyperspectral cubes.

```python
from ramankit import RamanImage

image = RamanImage(
    axis=[100.0, 200.0, 300.0],
    intensity=[
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
    ],
)
```

Image helpers include:

- `image.pixel(row, column)`
- `image.flatten()`
- `image.mean()`
- `image.std()`

## Metadata and provenance

`Metadata` stores scientific context such as sample or instrument information.

`Provenance` stores append-only transformation history. Operations and preprocessing steps append structured `ProvenanceStep` records instead of silently mutating the input history.
