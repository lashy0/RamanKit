from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ramankit.io.base import SpectralContainer


class _LoaderLike(Protocol):
    @property
    def format_name(self) -> str | None: ...

    @property
    def supported_suffixes(self) -> tuple[str, ...]: ...

    def load(self, path: str | Path) -> SpectralContainer: ...

    def can_load(self, path: str | Path) -> bool: ...


class LoaderRegistry:
    """Store and resolve RamanKit loaders with deterministic detection rules."""

    def __init__(self) -> None:
        self._loaders: dict[str, _LoaderLike] = {}

    def register(self, loader: _LoaderLike) -> None:
        """Register one loader instance by its stable format key."""

        format_name = (loader.format_name or "").strip().lower()
        if not format_name:
            raise ValueError("Registered loaders must define a non-empty format_name.")
        if format_name in self._loaders:
            raise ValueError(f"LoaderRegistry already has a loader registered for '{format_name}'.")
        self._loaders[format_name] = loader

    def get(self, format_name: str) -> _LoaderLike:
        """Return the registered loader for an explicit format key."""

        normalized = format_name.strip().lower()
        try:
            return self._loaders[normalized]
        except KeyError as error:
            available = ", ".join(sorted(self._loaders)) or "<none>"
            raise ValueError(
                f"Unsupported I/O format '{format_name}'. Registered formats: {available}."
            ) from error

    def load(self, path: str | Path, *, format: str | None = None) -> SpectralContainer:
        """Load one RamanKit container from a path with deterministic resolution."""

        candidate = Path(path)
        if format is not None:
            return self.get(format).load(candidate)

        extension_matches = self._match_suffix(candidate)
        if len(extension_matches) > 1:
            raise ValueError(
                f"Multiple loaders match suffix '{candidate.suffix.lower()}': "
                f"{', '.join(self._format_names(extension_matches))}."
            )
        if len(extension_matches) == 1:
            return extension_matches[0].load(candidate)

        sniff_matches = self._match_can_load(candidate)
        if len(sniff_matches) > 1:
            raise ValueError(
                f"Multiple loaders matched '{candidate}': "
                f"{', '.join(self._format_names(sniff_matches))}."
            )
        if len(sniff_matches) == 1:
            return sniff_matches[0].load(candidate)

        raise ValueError(
            f"No registered loader matched '{candidate}'. Provide format=... to load explicitly."
        )

    def _match_suffix(self, path: Path) -> list[_LoaderLike]:
        suffix = path.suffix.lower()
        if not suffix:
            return []
        return [
            loader
            for loader in self._loaders.values()
            if suffix in {item.lower() for item in loader.supported_suffixes}
        ]

    def _match_can_load(self, path: Path) -> list[_LoaderLike]:
        return [loader for loader in self._loaders.values() if loader.can_load(path)]

    def _format_names(self, loaders: list[_LoaderLike]) -> list[str]:
        return [loader.format_name or "<unknown>" for loader in loaders]


def _build_builtin_registry() -> LoaderRegistry:
    from ramankit.io.bwtek import BWTekLoader
    from ramankit.io.npz import NPZLoader

    registry = LoaderRegistry()
    registry.register(NPZLoader())
    registry.register(BWTekLoader())
    return registry


BUILTIN_LOADER_REGISTRY = _build_builtin_registry()


def load(path: str | Path, *, format: str | None = None) -> SpectralContainer:
    """Load one RamanKit container through the built-in loader registry."""

    return BUILTIN_LOADER_REGISTRY.load(path, format=format)


def _save_npz(data: SpectralContainer, path: str | Path) -> None:
    """Persist one container through the built-in NPZ saver."""

    from ramankit.io.npz import NPZSaver

    NPZSaver().save(data, path)
