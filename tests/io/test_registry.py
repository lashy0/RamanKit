from __future__ import annotations

from pathlib import Path

import pytest

from ramankit import Spectrum
from ramankit.io import BaseLoader, LoaderRegistry


def _spectrum() -> Spectrum:
    return Spectrum(axis=[100.0, 200.0], intensity=[1.0, 2.0])


class ExplicitLoader(BaseLoader[Spectrum]):
    format_name = "explicit"
    supported_suffixes = (".exp",)

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self, path: str | Path) -> Spectrum:
        self.load_calls += 1
        return _spectrum()


class ExtensionLoader(BaseLoader[Spectrum]):
    format_name = "text"
    supported_suffixes = (".txt",)

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self, path: str | Path) -> Spectrum:
        self.load_calls += 1
        return _spectrum()


class SniffLoader(BaseLoader[Spectrum]):
    format_name = "sniff"
    supported_suffixes = ()

    def __init__(self, *, matches: bool = True) -> None:
        self.matches = matches
        self.can_load_calls = 0
        self.load_calls = 0

    def can_load(self, path: str | Path) -> bool:
        self.can_load_calls += 1
        return self.matches

    def load(self, path: str | Path) -> Spectrum:
        self.load_calls += 1
        return _spectrum()


def test_registry_explicit_format_uses_only_requested_loader() -> None:
    """Explicit format bypasses suffix and sniff detection."""

    registry = LoaderRegistry()
    explicit = ExplicitLoader()
    registry.register(explicit)

    loaded = registry.load("sample.unknown", format="explicit")

    assert isinstance(loaded, Spectrum)
    assert explicit.load_calls == 1


def test_registry_uses_extension_match_before_can_load() -> None:
    """Suffix matches short-circuit optional sniff hooks."""

    registry = LoaderRegistry()
    extension_loader = ExtensionLoader()
    sniff_loader = SniffLoader(matches=True)
    registry.register(extension_loader)
    registry.register(sniff_loader)

    loaded = registry.load("sample.txt")

    assert isinstance(loaded, Spectrum)
    assert extension_loader.load_calls == 1
    assert sniff_loader.can_load_calls == 0


def test_registry_raises_for_ambiguous_suffix_match() -> None:
    """Auto-detection fails when more than one loader owns the same suffix."""

    registry = LoaderRegistry()
    registry.register(ExtensionLoader())

    class AnotherTextLoader(BaseLoader[Spectrum]):
        format_name = "text_two"
        supported_suffixes = (".txt",)

        def load(self, path: str | Path) -> Spectrum:
            return _spectrum()

    registry.register(AnotherTextLoader())

    with pytest.raises(ValueError, match="Multiple loaders match suffix"):
        registry.load("sample.txt")


def test_registry_raises_for_ambiguous_can_load_match() -> None:
    """Sniff-based detection fails when more than one loader matches."""

    registry = LoaderRegistry()
    registry.register(SniffLoader(matches=True))

    class OtherSniffLoader(SniffLoader):
        format_name = "sniff_two"

    registry.register(OtherSniffLoader(matches=True))

    with pytest.raises(ValueError, match="Multiple loaders matched"):
        registry.load("sample.unknown")


def test_registry_raises_when_no_loader_matches() -> None:
    """Auto-detection raises a clear error when no loader matches."""

    registry = LoaderRegistry()
    registry.register(SniffLoader(matches=False))

    with pytest.raises(ValueError, match="No registered loader matched"):
        registry.load("sample.unknown")
