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

Use `find_peaks_batch(...)` on a `SpectrumCollection` or `RamanImage`.

```python
batch_results = rpd.find_peaks_batch(collection, prominence=0.5, width=1.0)
image_results = rpd.find_peaks_batch(image, prominence=0.5, width=1.0)
```

For `RamanImage`, the returned list is flat and follows the row-major order produced by `image.flatten()`.

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
    model="voigt",
)
```

The multi-peak fit result includes:

- one shared `offset`
- fitted parameters for each component peak
- one total fitted curve
- one fitted curve per peak component

Supported models:

- `gaussian`
- `lorentzian`
- `voigt` via `scipy.special.voigt_profile`

For `gaussian` and `lorentzian`, fit results expose `width`. For `voigt`, fit results expose `sigma` and `gamma` instead.

## Notes

- peak detection supports `Spectrum`, `SpectrumCollection`, and `RamanImage`
- peak fitting supports `Spectrum`
- no preprocessing is performed implicitly before detection or fitting
- `fit_peak(...)` is single-peak only
- `fit_peaks(...)` uses one shared model type and one shared constant offset per fit window
