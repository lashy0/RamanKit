# Peaks

RamanKit keeps peak analysis separate from preprocessing and plotting and exposes it as module-level helpers.

## Import style

```python
import ramankit.peaks.detect as rpd
import ramankit.peaks.fit as rpf
```

## Peak detection

Use `find_peaks(...)` on a single `Spectrum`.

```python
result = rpd.find_peaks(spectrum, prominence=0.5, width=1.0)
peak = result[0]
```

The detection result is typed and exposes:

- sampled indices
- spectral positions
- heights
- optional prominence and width values when SciPy computes them

## Peak fitting

Use `fit_peak(...)` to fit one detected peak inside an explicit spectral window.

```python
fit_result = rpf.fit_peak(
    spectrum,
    peak,
    window=(950.0, 1050.0),
    model="gaussian",
)
```

## Multi-peak fitting

Use `fit_peaks(...)` to fit multiple detected peaks simultaneously inside one shared window.

```python
multi_fit_result = rpf.fit_peaks(
    spectrum,
    peaks=result[:2],
    window=(950.0, 1100.0),
    model="gaussian",
)
```

The multi-peak fit result includes:

- one shared `offset`
- fitted parameters for each component peak
- one total fitted curve
- one fitted curve per peak component

Supported models in v1:

- `gaussian`
- `lorentzian`

## Notes

- peak analysis in v1 supports only `Spectrum`
- no preprocessing is performed implicitly before detection or fitting
- `fit_peak(...)` is single-peak only
- `fit_peaks(...)` uses one shared model type and one shared constant offset per fit window
