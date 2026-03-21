# Preprocessing

RamanKit uses a step-based preprocessing API inspired by scientific workflow composition.

## Core concepts

Every preprocessing step:

- is a configured object
- is applied with `.apply(data)`
- returns a new object of the same container type
- preserves metadata
- appends one provenance step

RamanKit exposes two explicit step categories:

- `pp.PreprocessingStep` for transforms that preserve the spectral axis exactly
- `pp.AxisTransformStep` for transforms that may change both axis and intensity

This keeps spectral-axis changes explicit in the type of preprocessing step you implement.

## Import style

```python
import ramankit.preprocessing as pp
```

## Built-in steps

### Axis-preserving steps

#### Baseline

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

#### Despike

- `pp.despike.WhitakerHayes`

#### Smoothing

- `pp.smoothing.SavGol`
- `pp.smoothing.Whittaker`
- `pp.smoothing.Gaussian`

- `SavGol` for local polynomial smoothing with explicit window control.
- `Whittaker` for one-parameter smoothness control via `lam`.
- `Gaussian` for simple sigma-based smoothing.

#### Normalization

- `pp.normalization.Vector`
- `pp.normalization.Area`
- `pp.normalization.Max`
- `pp.normalization.MinMax`

#### Miscellaneous axis-preserving steps

- `pp.misc.BackgroundSubtractor`

### Axis-transform steps

- `pp.misc.Cropper`
- `pp.resample.Linear`

## Custom axis-transform steps

Use `pp.AxisTransformStep` when your preprocessing step returns both a new axis and matching intensity values.

```python
from dataclasses import dataclass

import ramankit.preprocessing as pp
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class ShiftAxis(pp.AxisTransformStep):
    function_name = "shift_axis"
    method_name = "constant"

    shift: float

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        return axis + self.shift, intensity
```

## Pipelines

Use `pp.Pipeline` to apply multiple steps in sequence.

```python
import ramankit.preprocessing as pp

pipeline = pp.Pipeline(
    [
        pp.misc.Cropper(lower_bound=400.0, upper_bound=1800.0),
        pp.baseline.ARPLS(),
        pp.smoothing.Whittaker(lam=1e3),
        pp.normalization.Vector(),
    ]
)

processed = pipeline.apply(spectrum)
```

## Scientific notes

- preprocessing always works along the spectral axis
- axis changes are explicit through `AxisTransformStep`; ordinary preprocessing steps preserve axis semantics
- resampling is explicit; arithmetic and reductions do not auto-align axes
- metadata is preserved and provenance is extended, not replaced
