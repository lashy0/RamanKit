# API overview

## Top-level package

```python
from ramankit import (
    Metadata,
    NumpyExport,
    Provenance,
    ProvenanceStep,
    RamanImage,
    Spectrum,
    SpectrumCollection,
    synthetic,
)
```

The top-level package intentionally exports the core domain models plus the metrics and synthetic namespaces.

## Core package

```python
from ramankit.core import NumpyExport, RamanImage, Spectrum, SpectrumCollection
```

Use this when you want explicit access to the core domain layer.

## Preprocessing package

```python
import ramankit.preprocessing as pp
```

Use this package for step-based preprocessing and pipelines.

The preprocessing namespace exposes two explicit step categories:

- `pp.PreprocessingStep` for transforms that preserve the spectral axis exactly
- `pp.AxisTransformStep` for transforms that may change both axis and intensity

## Peaks package

```python
import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
```

Use this package for peak detection, batch peak detection, and peak fitting on spectra.

## I/O package

```python
from ramankit.io import BaseLoader, BaseSaver, LoaderRegistry, load
```

Use this package when implementing custom readers and writers or when loading
data through the built-in registry.

## Analysis package

```python
import ramankit.analysis as ra
```

Use this package for spectral decomposition of `SpectrumCollection` and `RamanImage`
data. Available methods: `pca`, `nmf`, `ica`.

## Metrics package

```python
import ramankit.metrics as rm
```

Use this package for spectral similarity (`cosine_similarity`,
`pearson_correlation`, `mse`) and general quality metrics (`snr`, `band_area`)
across spectra, collections, and Raman images.

## Plotting package

```python
import ramankit.plotting.maps as rpm
import ramankit.plotting.peaks as rpp
import ramankit.plotting.spectra as rps
```

Use this package for module-level plotting helpers for spectra, peaks, and Raman maps.

## Pipelines package

```python
from ramankit.pipelines import AxisTransformStep, Pipeline, PreprocessingStep
```

This package contains the reusable pipeline abstraction plus the explicit base
classes for axis-preserving and axis-changing preprocessing steps.

## Synthetic package

```python
import ramankit.synthetic as rsyn
```

Use this package for synthetic spectra, collections, and Raman image generation from explicit peak, baseline, and noise parameters.
