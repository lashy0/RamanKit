from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ramankit import Metadata, Provenance, ProvenanceStep, RamanImage, Spectrum, SpectrumCollection
from ramankit.io import load as io_load
from ramankit.io.npz import NPZLoader, NPZSaver


def _test_path(name: str) -> Path:
    """Return a stable repository-local path for I/O tests."""

    root = Path(".cache/test_io")
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.exists():
        path.unlink()
    return path

def test_spectrum_npz_round_trip_via_public_io_load() -> None:
    """Round-trip a spectrum through NPZ save and public registry-based load."""

    spectrum = Spectrum(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        metadata=Metadata(
            sample="sample-1",
            instrument="inst-1",
            laser_wavelength=785.0,
            exposure_time=0.5,
            accumulations=3,
            acquisition_datetime="2025-02-19 12:06:04",
            operator="operator-1",
            extras={"laser": 785},
            raw_vendor_metadata={"source_format": "synthetic"},
        ),
        provenance=Provenance(
            source="synthetic",
            steps=(ProvenanceStep(name="load", parameters={"kind": "synthetic"}),),
        ),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("spectrum.npz")

    NPZSaver().save(spectrum, path)
    loaded = io_load(path, format="npz")

    assert isinstance(loaded, Spectrum)
    assert np.array_equal(loaded.axis, spectrum.axis)
    assert np.array_equal(loaded.intensity, spectrum.intensity)
    assert loaded.metadata == spectrum.metadata
    assert loaded.provenance.steps[:-1] == spectrum.provenance.steps
    assert loaded.provenance.steps[-1].name == "load_npz"
    assert loaded.spectral_axis_name == spectrum.spectral_axis_name
    assert loaded.spectral_unit == spectrum.spectral_unit

def test_collection_npz_round_trip_via_public_io_load() -> None:
    """Round-trip a spectrum collection through NPZ save and public load."""

    collection = SpectrumCollection(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        metadata=Metadata(sample="batch-1"),
        provenance=Provenance(steps=(ProvenanceStep(name="stack"),)),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("collection.npz")

    NPZSaver().save(collection, path)
    loaded = io_load(path)

    assert isinstance(loaded, SpectrumCollection)
    assert np.array_equal(loaded.axis, collection.axis)
    assert np.array_equal(loaded.intensity, collection.intensity)
    assert loaded.metadata == collection.metadata
    assert loaded.provenance.steps[:-1] == collection.provenance.steps
    assert loaded.provenance.steps[-1].name == "load_npz"

def test_image_npz_round_trip_via_public_io_load() -> None:
    """Round-trip a Raman image through NPZ save and public load."""

    image = RamanImage(
        axis=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]),
        metadata=Metadata(sample="image-1"),
        provenance=Provenance(steps=(ProvenanceStep(name="import"),)),
        spectral_axis_name="raman_shift",
        spectral_unit="cm^-1",
    )
    path = _test_path("image.npz")

    NPZSaver().save(image, path)
    loaded = io_load(path)

    assert isinstance(loaded, RamanImage)
    assert np.array_equal(loaded.axis, image.axis)
    assert np.array_equal(loaded.intensity, image.intensity)
    assert loaded.metadata == image.metadata
    assert loaded.provenance.steps[:-1] == image.provenance.steps
    assert loaded.provenance.steps[-1].name == "load_npz"

def test_npz_backend_round_trip_works_directly() -> None:
    """Round-trip a spectrum through the low-level NPZ saver and loader."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
    path = _test_path("low-level.npz")

    NPZSaver().save(spectrum, path)
    loaded = NPZLoader().load(path)

    assert isinstance(loaded, Spectrum)
    assert np.array_equal(loaded.intensity, spectrum.intensity)
    assert loaded.provenance.steps[-1].name == "load_npz"


def test_public_io_load_auto_detects_npz_suffix() -> None:
    """Top-level loading auto-detects NPZ archives from the suffix."""

    spectrum = Spectrum(axis=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
    path = _test_path("autodetect.npz")

    NPZSaver().save(spectrum, path)
    loaded = io_load(path)

    assert isinstance(loaded, Spectrum)


def test_core_containers_do_not_expose_load_or_save_methods() -> None:
    """Core containers no longer expose built-in NPZ convenience methods."""

    assert not hasattr(Spectrum, "load")
    assert not hasattr(Spectrum, "save")
    assert not hasattr(SpectrumCollection, "load")
    assert not hasattr(SpectrumCollection, "save")
    assert not hasattr(RamanImage, "load")
    assert not hasattr(RamanImage, "save")

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
