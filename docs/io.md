# I/O abstractions

RamanKit exposes generic extension points for file and directory I/O and ships one built-in persistence format for round-tripping existing containers.

## Public API

```python
from ramankit.io import BaseLoader, BaseSaver
from ramankit.io.npz import NPZLoader, NPZSaver
```

## Generic contracts

`BaseLoader[T]` defines one method:

```python
def load(self, path: str | Path) -> T:
    ...
```

`BaseSaver[T]` defines one method:

```python
def save(self, data: T, path: str | Path) -> None:
    ...
```

## Built-in NPZ format

All core containers provide convenience methods for the built-in NPZ round-trip format.

```python
from ramankit import Spectrum

spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
spectrum.save("spectrum.npz")
loaded = Spectrum.load("spectrum.npz")
```

Low-level access stays available through `ramankit.io.npz`:

```python
from ramankit.io.npz import NPZLoader, NPZSaver
```

## Example custom loader

```python
from pathlib import Path

from ramankit import Spectrum
from ramankit.io import BaseLoader


class MySpectrumLoader(BaseLoader[Spectrum]):
    def load(self, path: str | Path) -> Spectrum:
        raise NotImplementedError
```

## Design intent

- generic contracts are the extension points for future formats
- NPZ is the built-in persistence backend
- read-side and write-side responsibilities remain separate
- metadata and provenance are serialized explicitly without `pickle`
