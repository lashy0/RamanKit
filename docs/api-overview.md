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

## I/O package

```python
from ramankit.io import BaseLoader, BaseSaver
```

Use this package when implementing custom readers and writers.

## Pipelines package

```python
from ramankit.pipelines import Pipeline, PreprocessingStep
```

This package contains the reusable pipeline and preprocessing-step abstractions behind the preprocessing namespace.
