from __future__ import annotations

from collections.abc import Mapping

from ramankit.core.metadata import ProvenanceStep


def build_load_provenance_step(
    step_name: str,
    *,
    format_name: str,
    file_type: str,
    path: str,
    vendor: str | None = None,
    extra_parameters: Mapping[str, object] | None = None,
    description: str | None = None,
) -> ProvenanceStep:
    """Build a standard loader provenance step."""

    parameters: dict[str, object] = {
        "format": format_name,
        "file_type": file_type,
        "path": path,
    }
    if vendor is not None:
        parameters["vendor"] = vendor
    if extra_parameters is not None:
        parameters.update(extra_parameters)
    return ProvenanceStep(name=step_name, parameters=parameters, description=description)
