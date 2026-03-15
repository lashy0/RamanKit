# I/O abstractions

RamanKit currently exposes generic extension points for file and directory I/O, but does not yet ship concrete readers or writers.

## Public API

```python
from ramankit.io import BaseLoader, BaseSaver
```

## Loader contract

`BaseLoader[T]` defines one method:

```python
def load(self, path: str | Path) -> T:
    ...
```

`T` can be any supported core container:

- `Spectrum`
- `SpectrumCollection`
- `RamanImage`

## Saver contract

`BaseSaver[T]` defines one method:

```python
def save(self, data: T, path: str | Path) -> None:
    ...
```

## Example loader

```python
from pathlib import Path

from ramankit import Spectrum
from ramankit.io import BaseLoader


class MySpectrumLoader(BaseLoader[Spectrum]):
    def load(self, path: str | Path) -> Spectrum:
        raise NotImplementedError
```

## Design intent

The generic contracts exist so that concrete format support can be added without tying the public API to one vendor or one file layout too early.

This also keeps read-side and write-side responsibilities separate.
