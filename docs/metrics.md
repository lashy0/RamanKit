# Metrics

`ramankit.metrics` provides general spectral metrics that are independent of peak
analysis and preprocessing steps.

## Similarity

```python
import ramankit.metrics as rm

similarity = rm.cosine_similarity(left_spectrum, right_spectrum)
correlation = rm.pearson_correlation(left_spectrum, right_spectrum)
error = rm.mse(left_spectrum, right_spectrum)
```

These functions support:

- `Spectrum` -> one scalar result
- `SpectrumCollection` -> one value per spectrum
- `RamanImage` -> one value per pixel

All pairwise similarity functions require exact matches for:

- spectral axis values
- `spectral_axis_name`
- `spectral_unit`
- intensity shape

No implicit tolerance or automatic resampling is applied.

## Quality metrics

```python
snr_value = rm.snr(spectrum, noise_region=(1800.0, 1900.0), signal_region=(900.0, 1100.0))
band = rm.band_area(spectrum, region=(950.0, 1050.0), method="trapezoid")
```

- `snr` computes `signal / std(noise_region)`
- `band_area` integrates the selected region and returns an absolute area
- `band_area(method="simpson")` is also supported

For batch containers, both functions return arrays matching the collection or
image geometry.
