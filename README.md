# RamanKit

[![CI](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml/badge.svg)](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml)

RamanKit is a Python library for Raman spectroscopy data processing and analysis. The current focus is a typed, scientifically explicit core API for spectra, collections, hyperspectral Raman images, preprocessing pipelines, and extensible I/O contracts.

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

from ramankit import Spectrum
import ramankit.preprocessing as pp

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

Additional guides live in `docs/`:

- `docs/getting-started.md`
- `docs/core.md`
- `docs/preprocessing.md`
- `docs/io.md`
- `docs/api-overview.md`

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
