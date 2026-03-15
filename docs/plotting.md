# Plotting

RamanKit keeps plotting separate from the core data model and exposes plotting as module-level helpers.

## Import style

```python
import ramankit.plotting.maps as rpm
import ramankit.plotting.spectra as rps
```

## Spectra

Use `plot_spectrum(...)` for one `Spectrum` and `plot_collection(...)` for overlays of a
`SpectrumCollection`.

```python
figure, axes = rps.plot_spectrum(spectrum)
collection_figure, collection_axes = rps.plot_collection(collection)
```

Both helpers:

- return `matplotlib` `Figure` and `Axes`
- preserve the input data objects
- label the spectral axis from `spectral_axis_name` and `spectral_unit` when available

## Raman maps

Use `plot_image_band(...)` to render one spatial slice of a `RamanImage`.

```python
figure, axes = rpm.plot_image_band(image, index=10)
figure, axes = rpm.plot_image_band(image, shift=1000.0)
```

Exactly one selector must be provided:

- `index` for direct band selection
- `shift` for nearest-band selection on the current spectral axis

## Notes

- plotting uses `matplotlib`
- `show=False` by default
- no preprocessing or resampling is performed implicitly
