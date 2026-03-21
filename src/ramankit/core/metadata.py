from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _freeze_mapping(values: Mapping[str, object] | None) -> Mapping[str, object]:
    if values is None:
        return MappingProxyType({})
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class Metadata:
    """Store scientific metadata attached to spectral data."""

    sample: str | None = None
    instrument: str | None = None
    acquisition: str | None = None
    laser_wavelength: float | None = None
    grating: str | None = None
    exposure_time: float | None = None
    accumulations: int | None = None
    objective: str | None = None
    acquisition_datetime: str | None = None
    operator: str | None = None
    extras: Mapping[str, object] = field(default_factory=dict)
    raw_vendor_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _freeze_mapping(self.extras))
        object.__setattr__(self, "raw_vendor_metadata", _freeze_mapping(self.raw_vendor_metadata))

    def to_dict(self) -> dict[str, object]:
        """Return metadata as a plain dictionary."""

        data: dict[str, object] = {}
        if self.sample is not None:
            data["sample"] = self.sample
        if self.instrument is not None:
            data["instrument"] = self.instrument
        if self.acquisition is not None:
            data["acquisition"] = self.acquisition
        if self.laser_wavelength is not None:
            data["laser_wavelength"] = self.laser_wavelength
        if self.grating is not None:
            data["grating"] = self.grating
        if self.exposure_time is not None:
            data["exposure_time"] = self.exposure_time
        if self.accumulations is not None:
            data["accumulations"] = self.accumulations
        if self.objective is not None:
            data["objective"] = self.objective
        if self.acquisition_datetime is not None:
            data["acquisition_datetime"] = self.acquisition_datetime
        if self.operator is not None:
            data["operator"] = self.operator
        if self.extras:
            data["extras"] = dict(self.extras)
        if self.raw_vendor_metadata:
            data["raw_vendor_metadata"] = dict(self.raw_vendor_metadata)
        return data


@dataclass(frozen=True, slots=True)
class ProvenanceStep:
    """Store one structured processing or transformation step."""

    name: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Expected ProvenanceStep.name to be a non-empty string.")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class Provenance:
    """Store append-only provenance for spectral data objects."""

    source: str | None = None
    steps: tuple[ProvenanceStep, ...] = ()
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "extras", _freeze_mapping(self.extras))

    def append(self, step: ProvenanceStep) -> Provenance:
        """Return a new provenance object with one appended step."""

        return Provenance(
            source=self.source,
            steps=(*self.steps, step),
            extras=self.extras,
        )

    def with_source(self, source: str | None) -> Provenance:
        """Return a new provenance object with an updated source."""

        return Provenance(source=source, steps=self.steps, extras=self.extras)


EMPTY_METADATA = Metadata()
EMPTY_PROVENANCE = Provenance()


def ensure_metadata(metadata: Metadata | None) -> Metadata:
    """Return metadata or the shared empty metadata object."""

    return metadata if metadata is not None else EMPTY_METADATA


def ensure_provenance(provenance: Provenance | None) -> Provenance:
    """Return provenance or the shared empty provenance object."""

    return provenance if provenance is not None else EMPTY_PROVENANCE
