from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

import numpy as np

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.metadata import Metadata, Provenance, ProvenanceStep
from ramankit.core.spectrum import Spectrum

SpectralContainer = Spectrum | SpectrumCollection | RamanImage


class NPZSaver:
    """Persist RamanKit spectral containers in the internal NPZ format."""

    def save(self, data: SpectralContainer, path: str | Path) -> None:
        """Save one spectral container to an NPZ archive."""

        target = Path(path)
        np.savez(
            target,
            container_type=np.array(type(data).__name__),
            axis=data.axis,
            intensity=data.intensity,
            spectral_axis_name_json=np.array(json.dumps(data.spectral_axis_name)),
            spectral_unit_json=np.array(json.dumps(data.spectral_unit)),
            metadata_json=np.array(json.dumps(_metadata_to_dict(data.metadata))),
            provenance_json=np.array(json.dumps(_provenance_to_dict(data.provenance))),
        )


class NPZLoader:
    """Load RamanKit spectral containers from the internal NPZ format."""

    def load(self, path: str | Path) -> SpectralContainer:
        """Load one spectral container from an NPZ archive."""

        with np.load(Path(path), allow_pickle=False) as archive:
            required_keys = {
                "container_type",
                "axis",
                "intensity",
                "spectral_axis_name_json",
                "spectral_unit_json",
                "metadata_json",
                "provenance_json",
            }
            missing_keys = sorted(required_keys.difference(archive.files))
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(f"NPZ archive is missing required keys: {missing}.")

            container_type = str(archive["container_type"].item())
            axis = np.asarray(archive["axis"], dtype=np.float64)
            intensity = np.asarray(archive["intensity"], dtype=np.float64)
            spectral_axis_name = _decode_optional_string_field(
                archive,
                "spectral_axis_name_json",
            )
            spectral_unit = _decode_optional_string_field(archive, "spectral_unit_json")
            metadata = _metadata_from_dict(_decode_json_field(archive, "metadata_json"))
            provenance = _provenance_from_dict(
                _decode_json_field(archive, "provenance_json")
            )

        match container_type:
            case "Spectrum":
                return Spectrum(
                    axis=axis,
                    intensity=intensity,
                    metadata=metadata,
                    provenance=provenance,
                    spectral_axis_name=spectral_axis_name,
                    spectral_unit=spectral_unit,
                )
            case "SpectrumCollection":
                return SpectrumCollection(
                    axis=axis,
                    intensity=intensity,
                    metadata=metadata,
                    provenance=provenance,
                    spectral_axis_name=spectral_axis_name,
                    spectral_unit=spectral_unit,
                )
            case "RamanImage":
                return RamanImage(
                    axis=axis,
                    intensity=intensity,
                    metadata=metadata,
                    provenance=provenance,
                    spectral_axis_name=spectral_axis_name,
                    spectral_unit=spectral_unit,
                )
            case _:
                raise ValueError(f"Unsupported container_type '{container_type}' in NPZ archive.")


def _metadata_to_dict(metadata: Metadata) -> dict[str, object]:
    return metadata.to_dict()


def _metadata_from_dict(payload: object) -> Metadata:
    if not isinstance(payload, dict):
        raise ValueError("Expected metadata payload to be a JSON object.")

    sample = payload.get("sample")
    instrument = payload.get("instrument")
    acquisition = payload.get("acquisition")
    extras = {
        key: value
        for key, value in payload.items()
        if key not in {"sample", "instrument", "acquisition"}
    }
    return Metadata(
        sample=sample if isinstance(sample, str) else None,
        instrument=instrument if isinstance(instrument, str) else None,
        acquisition=acquisition if isinstance(acquisition, str) else None,
        extras=extras,
    )


def _provenance_to_dict(provenance: Provenance) -> dict[str, object]:
    return {
        "source": provenance.source,
        "steps": [
            {
                "name": step.name,
                "parameters": dict(step.parameters),
                "description": step.description,
            }
            for step in provenance.steps
        ],
        "extras": dict(provenance.extras),
    }


def _provenance_from_dict(payload: object) -> Provenance:
    if not isinstance(payload, dict):
        raise ValueError("Expected provenance payload to be a JSON object.")

    source = payload.get("source")
    steps_payload = payload.get("steps", [])
    extras = payload.get("extras", {})

    if not isinstance(steps_payload, list):
        raise ValueError("Expected provenance steps to be a JSON list.")
    if not isinstance(extras, dict):
        raise ValueError("Expected provenance extras to be a JSON object.")

    steps = tuple(_provenance_step_from_dict(step_payload) for step_payload in steps_payload)
    return Provenance(
        source=source if isinstance(source, str) else None,
        steps=steps,
        extras=extras,
    )


def _provenance_step_from_dict(payload: object) -> ProvenanceStep:
    if not isinstance(payload, dict):
        raise ValueError("Expected provenance step payload to be a JSON object.")

    name = payload.get("name")
    parameters = payload.get("parameters", {})
    description = payload.get("description")

    if not isinstance(name, str):
        raise ValueError("Expected provenance step name to be a string.")
    if not isinstance(parameters, dict):
        raise ValueError("Expected provenance step parameters to be a JSON object.")
    if description is not None and not isinstance(description, str):
        raise ValueError("Expected provenance step description to be a string or null.")

    return ProvenanceStep(name=name, parameters=parameters, description=description)


def _decode_optional_string_field(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> str | None:
    payload = _decode_json_field(archive, key)
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise ValueError(f"Expected '{key}' to store a JSON string or null.")
    return payload


def _decode_json_field(archive: np.lib.npyio.NpzFile, key: str) -> object:
    try:
        return json.loads(str(archive[key].item()))
    except (JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON payload stored in '{key}'.") from error
