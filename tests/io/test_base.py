from __future__ import annotations

from pathlib import Path

import pytest

from ramankit import RamanImage, Spectrum, SpectrumCollection
from ramankit.io import BaseLoader, BaseSaver


class DummySpectrumLoader(BaseLoader[Spectrum]):
    """Return deterministic spectra for generic loader tests."""

    def load(self, path: str | Path) -> Spectrum:
        """Return a spectrum whose intensity depends on the requested path."""

        axis = [100.0, 200.0, 300.0]
        if str(path) == "first":
            return Spectrum(axis=axis, intensity=[1.0, 2.0, 3.0])
        if str(path) == "second":
            return Spectrum(axis=axis, intensity=[2.0, 3.0, 4.0])
        return Spectrum(axis=[100.0, 250.0, 300.0], intensity=[1.0, 2.0, 3.0])

class DummyCollectionLoader(BaseLoader[SpectrumCollection]):
    """Return a deterministic spectrum collection for loader tests."""

    def load(self, path: str | Path) -> SpectrumCollection:
        """Return one collection for the requested path."""

        return SpectrumCollection(
            axis=[100.0, 200.0, 300.0],
            intensity=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        )

class DummyImageLoader(BaseLoader[RamanImage]):
    """Return a deterministic Raman image for loader tests."""

    def load(self, path: str | Path) -> RamanImage:
        """Return one image for the requested path."""

        return RamanImage(
            axis=[100.0, 200.0, 300.0],
            intensity=[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]],
        )

class DummySpectrumSaver(BaseSaver[Spectrum]):
    """Capture spectrum save calls for saver tests."""

    def __init__(self) -> None:
        self.saved: tuple[Spectrum, str | Path] | None = None

    def save(self, data: Spectrum, path: str | Path) -> None:
        """Store the save request for later assertions."""

        self.saved = (data, path)

class DummyCollectionSaver(BaseSaver[SpectrumCollection]):
    """Capture collection save calls for saver tests."""

    def __init__(self) -> None:
        self.saved: tuple[SpectrumCollection, str | Path] | None = None

    def save(self, data: SpectrumCollection, path: str | Path) -> None:
        """Store the save request for later assertions."""

        self.saved = (data, path)

class DummyImageSaver(BaseSaver[RamanImage]):
    """Capture image save calls for saver tests."""

    def __init__(self) -> None:
        self.saved: tuple[RamanImage, str | Path] | None = None

    def save(self, data: RamanImage, path: str | Path) -> None:
        """Store the save request for later assertions."""

        self.saved = (data, path)

def test_base_loader_cannot_be_instantiated_without_load_method() -> None:
    """Reject instantiation of the generic abstract loader base class."""

    with pytest.raises(TypeError):
        BaseLoader()

def test_base_saver_cannot_be_instantiated_without_save_method() -> None:
    """Reject instantiation of the generic abstract saver base class."""

    with pytest.raises(TypeError):
        BaseSaver()

def test_dummy_spectrum_loader_returns_spectrum() -> None:
    """Load one spectrum through the generic spectrum loader contract."""

    spectrum = DummySpectrumLoader().load("first")

    assert isinstance(spectrum, Spectrum)
    assert spectrum.n_points == 3

def test_dummy_collection_loader_returns_collection() -> None:
    """Load one collection through the generic collection loader contract."""

    collection = DummyCollectionLoader().load("collection-path")

    assert isinstance(collection, SpectrumCollection)
    assert collection.n_spectra == 2

def test_dummy_image_loader_returns_image() -> None:
    """Load one image through the generic image loader contract."""

    image = DummyImageLoader().load("image-path")

    assert isinstance(image, RamanImage)
    assert image.n_pixels == 2

def test_spectrum_saver_captures_save_call() -> None:
    """Save one spectrum through the generic saver method."""

    saver = DummySpectrumSaver()
    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])

    saver.save(spectrum, "spectrum.txt")

    assert saver.saved == (spectrum, "spectrum.txt")

def test_collection_saver_captures_save_call() -> None:
    """Save one collection through the generic saver method."""

    saver = DummyCollectionSaver()
    collection = SpectrumCollection(
        axis=[100.0, 200.0, 300.0],
        intensity=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    )

    saver.save(collection, "collection.npz")

    assert saver.saved == (collection, "collection.npz")

def test_image_saver_captures_save_call() -> None:
    """Save one image through the generic saver method."""

    saver = DummyImageSaver()
    image = RamanImage(
        axis=[100.0, 200.0, 300.0],
        intensity=[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]],
    )

    saver.save(image, "image.npz")

    assert saver.saved == (image, "image.npz")
