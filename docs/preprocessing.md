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

Least-squares methods:

- `pp.baseline.ASLS`
- `pp.baseline.IASLS`
- `pp.baseline.AIRPLS`
- `pp.baseline.ARPLS`
- `pp.baseline.DRPLS`
- `pp.baseline.IARPLS`
- `pp.baseline.ASPLS`

Polynomial methods:

- `pp.baseline.Poly`
- `pp.baseline.ModPoly`
- `pp.baseline.PenalisedPoly`
- `pp.baseline.IModPoly`

Other methods:

- `pp.baseline.Goldindec`
- `pp.baseline.IRSQR`
- `pp.baseline.CornerCutting`
- `pp.baseline.FABC`

#### Method selection

- Start with `ARPLS` or `ASLS` for general fluorescence backgrounds.
- Use `AIRPLS` or `IARPLS` when peaks are strong and the baseline needs more robust reweighting.
- Use `ModPoly` or `IModPoly` for simple polynomial-like backgrounds without tuning `lam`.
- Use `IRSQR` or `FABC` when the background shape is difficult and a more specialized model helps.

### Despike

- `pp.despike.WhitakerHayes`

### Smoothing

- `pp.smoothing.SavGol`
- `pp.smoothing.Whittaker`
- `pp.smoothing.Gaussian`

- `SavGol` for local polynomial smoothing with explicit window control.
- `Whittaker` for one-parameter smoothness control via `lam`.
- `Gaussian` for simple sigma-based smoothing.

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
        pp.baseline.ARPLS(),
        pp.smoothing.Whittaker(lam=1e3),
        pp.normalization.Vector(),
    ]
)

processed = pipeline.apply(spectrum)
```

## Scientific notes

- preprocessing always works along the spectral axis
- resampling is explicit; arithmetic and reductions do not auto-align axes
- metadata is preserved and provenance is extended, not replaced
