# RamanKit

[![CI](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml/badge.svg)](https://github.com/lashy0/RamanKit/actions/workflows/ci.yml)

RamanKit is a Python library for Raman spectroscopy data processing and analysis. It provides a typed, scientifically explicit API for spectra, collections, hyperspectral Raman images, preprocessing pipelines, plotting helpers, peak analysis, and extensible I/O contracts.

## Package layout

```text
src/ramankit/
├─ __init__.py          # public package exports
├─ core/                # spectral domain models and metadata
├─ io/                  # generic I/O contracts and persistence backends
├─ peaks/               # peak analysis modules
├─ pipelines/           # reusable preprocessing pipelines
├─ plotting/            # plotting helpers
├─ preprocessing/       # preprocessing steps and built-in operations
└─ synthetic/           # synthetic spectra, collections, and image generation
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

Baseline correction includes least-squares (`ASLS`, `ARPLS`, `IARPLS`, `ASPLS`),
polynomial (`Poly`, `ModPoly`, `PenalisedPoly`, `IModPoly`), and additional
methods such as `Goldindec`, `IRSQR`, `CornerCutting`, and `FABC`.
Smoothing includes `SavGol`, `Whittaker`, and `Gaussian`. Normalization includes `Vector`, `Area`, `Max`, and `MinMax`.

### Peak analysis

```python
import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf

result = rpd.find_peaks(spectrum, prominence=0.5, width=1.0)
peak = result[0]
batch_results = rpd.find_peaks_batch(collection, prominence=0.5, width=1.0)
fit_result = rpf.fit_peak(spectrum, peak, window=(900.0, 1100.0), model="voigt")
multi_fit_result = rpf.fit_peaks(spectrum, peaks=result[:2], window=(900.0, 1100.0), model="voigt")
```

### Synthetic data

```python
import numpy as np
import ramankit.synthetic as rsyn

spectrum = rsyn.generate_spectrum(
    axis=np.linspace(100.0, 300.0, 201),
    config=rsyn.SyntheticSpectrumConfig(
        peaks=(
            rsyn.PeakComponent(amplitude=5.0, center=160.0, width=6.0),
            rsyn.PeakComponent(model="lorentzian", amplitude=3.0, center=220.0, width=5.0),
        ),
        baseline=rsyn.PolynomialBaseline(coefficients=(0.2, 0.001, 2e-6)),
        noise=rsyn.GaussianNoise(sigma=0.05, seed=7),
    ),
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)
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

### Plotting

```python
import ramankit.plotting.maps as rpm
import ramankit.plotting.peaks as rpp
import ramankit.plotting.spectra as rps

from ramankit import RamanImage, Spectrum

spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
figure, axes = rps.plot_spectrum(spectrum)

image = RamanImage(
    axis=[100.0, 200.0, 300.0],
    intensity=[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]],
)
map_figure, map_axes = rpm.plot_image_band(image, index=1)
peak_figure, peak_axes = rpp.plot_detected_peaks(spectrum, result)
fit_figure, fit_axes = rpp.plot_peak_fit(spectrum, peak, fit_result)
```

## Documentation

```text
docs/
├─ getting-started.md  # installation, first objects, and the basic workflow
├─ core.md             # core container semantics, metadata, and provenance
├─ preprocessing.md    # preprocessing steps, pipelines, and built-in operations
├─ io.md               # generic I/O contracts and built-in NPZ persistence
├─ peaks.md            # peak detection, single-peak fitting, and multi-peak fitting
├─ plotting.md         # spectral, peak, and map plotting helpers
├─ synthetic.md        # synthetic spectrum, collection, and image generation
└─ api-overview.md     # public imports and package layout
```

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```






