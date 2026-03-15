# Preprocessing

RamanKit uses a step-based preprocessing API inspired by scientific workflow composition.

## Core concepts

Every preprocessing step:

- is a configured object
- is applied with `.apply(data)`
- returns a new object of the same container type
- preserves metadata
- appends one provenance step

## Import style

```python
import ramankit.preprocessing as pp
```

## Built-in steps

### Baseline

- `pp.baseline.ASLS`

### Despike

- `pp.despike.WhitakerHayes`

### Smoothing

- `pp.smoothing.SavGol`

### Normalization

- `pp.normalization.Vector`
- `pp.normalization.Area`
- `pp.normalization.Max`

### Miscellaneous

- `pp.misc.Cropper`
- `pp.misc.BackgroundSubtractor`

### Resampling

- `pp.resample.Linear`

## Pipelines

Use `pp.Pipeline` to apply multiple steps in sequence.

```python
import ramankit.preprocessing as pp

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

## Scientific notes

- preprocessing always works along the spectral axis
- resampling is explicit; arithmetic and reductions do not auto-align axes
- metadata is preserved and provenance is extended, not replaced
