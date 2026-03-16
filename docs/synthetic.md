# Synthetic data

Use `ramankit.synthetic` to generate synthetic `Spectrum`, `SpectrumCollection`, and `RamanImage` objects from an explicit spectral axis and parametric components.

## Imports

```python
import numpy as np
import ramankit.synthetic as rsyn
```

## One spectrum

```python
spectrum = rsyn.generate_spectrum(
    axis=np.linspace(100.0, 300.0, 201),
    config=rsyn.SyntheticSpectrumConfig(
        peaks=(
            rsyn.PeakComponent(amplitude=5.0, center=160.0, width=6.0),
            rsyn.PeakComponent(model="voigt", amplitude=8.0, center=220.0, sigma=2.0, gamma=3.0),
        ),
        baseline=rsyn.LinearBaseline(offset=0.2, slope=0.001),
        noise=rsyn.GaussianNoise(sigma=0.05, seed=7),
    ),
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)
```

## Collection and image

```python
collection = rsyn.generate_collection(
    axis=np.linspace(100.0, 300.0, 201),
    configs=[
        rsyn.SyntheticSpectrumConfig(
            peaks=(rsyn.PeakComponent(amplitude=5.0, center=160.0, width=6.0),),
        ),
        rsyn.SyntheticSpectrumConfig(
            peaks=(rsyn.PeakComponent(model="lorentzian", amplitude=3.0, center=220.0, width=5.0),),
        ),
    ],
)

image = rsyn.generate_image(
    axis=np.linspace(100.0, 300.0, 201),
    configs=[
        [
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=5.0, center=160.0, width=6.0),),
            ),
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=4.0, center=180.0, width=5.0),),
            ),
        ],
        [
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=3.0, center=200.0, width=4.0),),
            ),
            rsyn.SyntheticSpectrumConfig(
                peaks=(rsyn.PeakComponent(amplitude=2.0, center=220.0, width=7.0),),
            ),
        ],
    ],
)
```

## Notes

- The spectral axis is always provided explicitly by the caller.
- Peak models currently support `gaussian`, `lorentzian`, and `voigt`.
- `LinearBaseline` adds `offset + slope * (axis - axis[0])`.
- `GaussianNoise` is additive and reproducible when `seed` is set.
- Generated data is marked with `provenance.source == "synthetic"`.
