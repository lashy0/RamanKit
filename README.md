# RamanKit

[![CI](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml/badge.svg)](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml)

RamanKit is a Python library for Raman spectroscopy data processing and analysis. It provides a typed, scientifically explicit API for spectra, collections, hyperspectral Raman images, preprocessing pipelines, and extensible I/O contracts.

## Package layout

```text
src/ramankit/
├─ __init__.py          # public package exports
├─ core/                # spectral domain models and metadata
├─ io/                  # generic I/O contracts and persistence backends
├─ peaks/               # peak analysis modules
├─ pipelines/           # reusable preprocessing pipelines
├─ plotting/            # plotting helpers
└─ preprocessing/       # preprocessing steps and built-in operations
```

## Installation

```bash
uv sync
```

Or, for editable local development:

```bash
pip install -e .
```

## Quick start

### Core objects

```python
import numpy as np

from ramankit import Spectrum, SpectrumCollection

spectrum = Spectrum(
    axis=np.array([100.0, 200.0, 300.0]),
    intensity=np.array([3.0, 5.0, 4.0]),
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)

collection = SpectrumCollection(
    axis=np.array([100.0, 200.0, 300.0]),
    intensity=np.array([
        [3.0, 5.0, 4.0],
        [2.0, 4.0, 6.0],
    ]),
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)

mean_spectrum = collection.mean()
```

### Preprocessing

```python
import numpy as np
import ramankit.preprocessing as pp

from ramankit import Spectrum

spectrum = Spectrum(
    axis=np.linspace(100.0, 1800.0, 9),
    intensity=np.array([5.0, 6.0, 8.0, 15.0, 30.0, 16.0, 9.0, 6.0, 5.0]),
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)

pipeline = pp.Pipeline(
    [
        pp.despike.WhitakerHayes(),
        pp.baseline.ASLS(),
        pp.smoothing.SavGol(window_length=5, polyorder=2),
        pp.normalization.Vector(),
    ]
)

processed = pipeline.apply(spectrum)
```

### NPZ persistence

```python
from ramankit import Spectrum

spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
spectrum.save("spectrum.npz")
loaded = Spectrum.load("spectrum.npz")
```

### Generic I/O contracts

```python
from pathlib import Path

from ramankit import Spectrum
from ramankit.io import BaseLoader


class MySpectrumLoader(BaseLoader[Spectrum]):
    def load(self, path: str | Path) -> Spectrum:
        raise NotImplementedError
```

## Documentation

```text
docs/
├─ getting-started.md  # installation, first objects, and the basic workflow
├─ core.md             # core container semantics, metadata, and provenance
├─ preprocessing.md    # preprocessing steps, pipelines, and built-in operations
├─ io.md               # generic I/O contracts and built-in NPZ persistence
└─ api-overview.md     # public imports and package layout
```

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
