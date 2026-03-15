# API overview

## Top-level package

```python
from ramankit import (
    Metadata,
    Provenance,
    ProvenanceStep,
    RamanImage,
    Spectrum,
    SpectrumCollection,
)
```

The top-level package intentionally exports only the core domain models.

## Core package

```python
from ramankit.core import Spectrum, SpectrumCollection, RamanImage
```

Use this when you want explicit access to the core domain layer.

## Preprocessing package

```python
import ramankit.preprocessing as pp
```

Use this package for step-based preprocessing and pipelines.

## Peaks package

```python
import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
```

Use this package for peak detection and single-peak fitting on spectra.

## I/O package

```python
from ramankit.io import BaseLoader, BaseSaver
```

Use this package when implementing custom readers and writers.

## Plotting package

```python
import ramankit.plotting.maps as rpm
import ramankit.plotting.peaks as rpp
import ramankit.plotting.spectra as rps
```

Use this package for module-level plotting helpers for spectra, peaks, and Raman maps.

## Pipelines package

```python
from ramankit.pipelines import Pipeline, PreprocessingStep
```

This package contains the reusable pipeline and preprocessing-step abstractions behind the preprocessing namespace.
