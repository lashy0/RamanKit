# I/O abstractions

RamanKit exposes generic loader and saver contracts, a built-in loader registry, and one built-in NPZ persistence format for round-tripping existing containers.

## Public API

```python
from ramankit.io import BaseLoader, BaseSaver, LoaderRegistry, load
from ramankit.io.npz import NPZLoader, NPZSaver
```

## Generic contracts

`BaseLoader[T]` defines one method:

```python
def load(self, path: str | Path) -> T:
    ...
```

Loaders may also declare:

```python
format_name: str
supported_suffixes: tuple[str, ...]

def can_load(self, path: str | Path) -> bool:
    ...
```

`can_load()` must stay cheap and deterministic. It may inspect suffixes,
directory layout, or a small fixed-size header read, but it must not perform
deep parsing.

`BaseSaver[T]` defines one method:

```python
def save(self, data: T, path: str | Path) -> None:
    ...
```

## Registry-based loading

Top-level loading goes through the built-in registry:

```python
from ramankit.io import load

loaded = load("spectrum.npz", format="npz")
```

Auto-detection is conservative:

- explicit `format=` always wins
- otherwise suffix matching runs first
- optional `can_load()` runs only when no suffix matched
- ambiguous matches raise `ValueError`
- no match raises `ValueError`

## Built-in NPZ format

The preferred NPZ workflow is:

```python
from ramankit import Spectrum
from ramankit.io import load
from ramankit.io.npz import NPZSaver

spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
NPZSaver().save(spectrum, "spectrum.npz")
loaded = load("spectrum.npz", format="npz")
```

Low-level access stays available through `ramankit.io.npz`:

```python
from ramankit.io.npz import NPZLoader, NPZSaver
```

## CSV / TSV loader

The built-in CSV loader handles comma, tab, and semicolon-delimited files:

```python
from ramankit.io import load

# Auto-detected by .csv or .tsv suffix
spectrum = load("data.csv")
spectrum = load("data.tsv")

# Explicit format for .txt files
spectrum = load("data.txt", format="csv")
```

For non-standard layouts, configure the loader directly:

```python
from ramankit.io.csv import CSVLoader

loader = CSVLoader(
    axis_column="Raman Shift",
    intensity_column="Intensity",
    delimiter=";",
    skip_rows=2,
    decimal=",",
    spectral_axis_name="raman_shift",
    spectral_unit="cm^-1",
)
spectrum = loader.load("data.csv")
```

| Parameter | Default | Meaning |
|---|---|---|
| `axis_column` | `0` | Column for spectral axis (name or index) |
| `intensity_column` | `1` | Column for intensity (name or index) |
| `delimiter` | `None` | Field separator; `None` auto-detects (tab, semicolon, comma) |
| `skip_rows` | `0` | Leading rows to skip before header/data |
| `encoding` | `"utf-8"` | Text encoding |
| `decimal` | `"."` | Decimal separator (`"."` or `","`) |
| `spectral_axis_name` | `None` | Spectral axis name set on the returned Spectrum |
| `spectral_unit` | `None` | Spectral unit set on the returned Spectrum |

## Example custom loader

```python
from pathlib import Path

from ramankit import Spectrum
from ramankit.io import BaseLoader


class MySpectrumLoader(BaseLoader[Spectrum]):
    format_name = "my_spectrum"
    supported_suffixes = (".txt",)

    def can_load(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() == ".txt"

    def load(self, path: str | Path) -> Spectrum:
        raise NotImplementedError
```

## Design intent

- generic contracts and the registry are the extension points for future formats
- NPZ is the built-in persistence backend
- read-side and write-side responsibilities remain separate
- metadata and provenance are serialized explicitly without `pickle`
