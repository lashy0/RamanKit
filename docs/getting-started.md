# Getting started

RamanKit is organized around three layers:

- core domain models for spectral data
- preprocessing steps applied through `.apply(...)`
- generic and built-in I/O for persistence and format adapters

## Installation

```bash
uv sync
```

## Core workflow

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

summary = collection.mean()
```

## Preprocessing workflow

```python
import ramankit.preprocessing as pp

pipeline = pp.Pipeline(
    [
        pp.baseline.ASLS(),
        pp.smoothing.SavGol(window_length=5, polyorder=2),
        pp.normalization.Vector(),
    ]
)

processed = pipeline.apply(spectrum)
```

## Built-in persistence workflow

```python
from ramankit import Spectrum

spectrum.save("spectrum.npz")
loaded = Spectrum.load("spectrum.npz")
```

## I/O extension workflow

Vendor or text readers can be built on top of `ramankit.io`.

```python
from pathlib import Path

from ramankit import Spectrum
from ramankit.io import BaseLoader


class MySpectrumLoader(BaseLoader[Spectrum]):
    def load(self, path: str | Path) -> Spectrum:
        raise NotImplementedError
```

## Next pages

- `core.md` for container semantics
- `preprocessing.md` for step-based preprocessing
- `io.md` for generic I/O extension points and NPZ persistence
- `plotting.md` for module-level plotting helpers
- `peaks.md` for peak detection and fitting
- `api-overview.md` for public imports
