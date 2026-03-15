from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection
from ramankit.io import BaseLoader, BaseSaver
from ramankit.io.npz import NPZLoader, NPZSaver


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


def _test_path(name: str) -> Path:
    """Return a stable repository-local path for I/O tests."""

    root = Path(".cache/test_io")
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.exists():
        path.unlink()
    return path


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


def test_spectrum_npz_round_trip_via_container_methods() -> None:
    """Round-trip a spectrum through the built-in NPZ persistence API."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        metadata=Metadata(sample="sample-1", instrument="inst-1", extras={"laser": 785}),
        provenance=Provenance(
            source="synthetic",
            steps=(ProvenanceStep(name="load", parameters={"kind": "synthetic"}),),
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("spectrum.npz")

    spectrum.save(path)
    loaded = Spectrum.load(path)

    assert np.array_equal(loaded.axis, spectrum.axis)
    assert np.array_equal(loaded.intensity, spectrum.intensity)
    assert loaded.metadata == spectrum.metadata
    assert loaded.provenance == spectrum.provenance
    assert loaded.spectral_axis_name == spectrum.spectral_axis_name
    assert loaded.spectral_unit == spectrum.spectral_unit


def test_collection_npz_round_trip_via_container_methods() -> None:
    """Round-trip a spectrum collection through the built-in NPZ persistence API."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        metadata=Metadata(sample="batch-1"),
        provenance=Provenance(steps=(ProvenanceStep(name="stack"),)),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("collection.npz")

    collection.save(path)
    loaded = SpectrumCollection.load(path)

    assert np.array_equal(loaded.axis, collection.axis)
    assert np.array_equal(loaded.intensity, collection.intensity)
    assert loaded.metadata == collection.metadata
    assert loaded.provenance == collection.provenance


def test_image_npz_round_trip_via_container_methods() -> None:
    """Round-trip a Raman image through the built-in NPZ persistence API."""

    image = RamanImage(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]),
        metadata=Metadata(sample="image-1"),
        provenance=Provenance(steps=(ProvenanceStep(name="import"),)),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("image.npz")

    image.save(path)
    loaded = RamanImage.load(path)

    assert np.array_equal(loaded.axis, image.axis)
    assert np.array_equal(loaded.intensity, image.intensity)
    assert loaded.metadata == image.metadata
    assert loaded.provenance == image.provenance


def test_npz_backend_round_trip_works_directly() -> None:
    """Round-trip a spectrum through the low-level NPZ saver and loader."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
    path = _test_path("low-level.npz")

    NPZSaver().save(spectrum, path)
    loaded = NPZLoader().load(path)

    assert isinstance(loaded, Spectrum)
    assert np.array_equal(loaded.intensity, spectrum.intensity)


def test_container_load_raises_for_type_mismatch() -> None:
    """Reject loading an NPZ file through the wrong container class."""

    image = RamanImage(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]),
    )
    path = _test_path("wrong-type.npz")

    image.save(path)

    with pytest.raises(ValueError, match="Expected Spectrum"):
        Spectrum.load(path)


def test_npz_loader_raises_for_unknown_container_type() -> None:
    """Reject NPZ archives with an unsupported container type."""

    path = _test_path("unknown-type.npz")
    np.savez(
        path,
        container_type=np.array("Unknown"),
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectral_axis_name_json=np.array(json.dumps(None)),
        spectral_unit_json=np.array(json.dumps(None)),
        metadata_json=np.array(json.dumps({})),
        provenance_json=np.array(json.dumps({"source": None, "steps": [], "extras": {}})),
    )

    with pytest.raises(ValueError, match="Unsupported container_type"):
        NPZLoader().load(path)


def test_npz_loader_raises_for_missing_required_keys() -> None:
    """Reject NPZ archives that miss required serialization keys."""

    path = _test_path("missing-key.npz")
    np.savez(path, container_type=np.array("Spectrum"), axis=np.array([100.0, 200.0, 300.0]))

    with pytest.raises(ValueError, match="missing required keys"):
        NPZLoader().load(path)


def test_npz_loader_raises_for_invalid_metadata_json() -> None:
    """Reject NPZ archives with invalid metadata JSON payloads."""

    path = _test_path("invalid-json.npz")
    np.savez(
        path,
        container_type=np.array("Spectrum"),
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectral_axis_name_json=np.array(json.dumps(None)),
        spectral_unit_json=np.array(json.dumps(None)),
        metadata_json=np.array("{"),
        provenance_json=np.array(json.dumps({"source": None, "steps": [], "extras": {}})),
    )

    with pytest.raises(ValueError, match="metadata_json"):
        NPZLoader().load(path)
