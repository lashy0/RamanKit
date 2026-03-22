from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ramankit.core._validation import NumericArray, coerce_axis
from ramankit.core.collection import SpectrumCollection
from ramankit.core.metadata import ProvenanceStep, ensure_provenance
from ramankit.core.spectrum import Spectrum
from ramankit.pipelines.pipeline import AxisTransformStep
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class Linear(AxisTransformStep):
    """Linear interpolation onto an explicitly provided spectral axis."""

    function_name = "resample"
    method_name = "linear"

    target_axis: npt.ArrayLike

    def __post_init__(self) -> None:
        axis, _ = coerce_axis(self.target_axis)
        object.__setattr__(self, "target_axis", axis)

    @property
    def target_axis_array(self) -> NumericArray:
        """Return the validated target axis as a NumPy array."""

        return np.array(self.target_axis, dtype=np.float64, copy=True)

    def _transform_with_axis(self, intensity: Array1D, axis: Array1D) -> tuple[Array1D, Array1D]:
        target_axis = self.target_axis_array
        source_min = float(np.min(axis))
        source_max = float(np.max(axis))
        target_min = float(np.min(target_axis))
        target_max = float(np.max(target_axis))
        if target_min < source_min or target_max > source_max:
            raise ValueError("Expected target_axis to stay within the source axis range.")

        source_axis = axis
        source_intensity = intensity
        if source_axis[0] > source_axis[-1]:
            source_axis = source_axis[::-1]
            source_intensity = source_intensity[::-1]

        target_for_interp = target_axis if target_axis[0] < target_axis[-1] else target_axis[::-1]
        resampled = np.interp(target_for_interp, source_axis, source_intensity)
        if target_axis[0] > target_axis[-1]:
            resampled = resampled[::-1]
        return target_axis, np.asarray(resampled, dtype=np.float64)


def resample_to_common_axis(
    spectra: list[Spectrum] | tuple[Spectrum, ...],
    *,
    n_points: int | None = None,
) -> SpectrumCollection:
    """Resample spectra with different axes onto a common uniform grid.

    Finds the overlapping axis range across all input spectra, creates a
    uniform grid within that range, and linearly interpolates each spectrum
    onto the common grid.

    Args:
        spectra: Two or more spectra with potentially different spectral axes.
        n_points: Number of points in the common grid.  If ``None``, uses the
            median number of points across the input spectra.

    Returns:
        A :class:`SpectrumCollection` with all spectra on the same axis.

    Raises:
        ValueError: If fewer than two spectra are provided.
        ValueError: If the overlapping axis range is empty.
        ValueError: If spectra have incompatible spectral semantics.
    """

    if len(spectra) < 2:
        raise ValueError(
            "Expected at least two spectra for common-axis resampling."
        )

    # Validate compatible spectral semantics.
    ref_name = spectra[0].spectral_axis_name
    ref_unit = spectra[0].spectral_unit
    for i, s in enumerate(spectra[1:], start=1):
        if s.spectral_axis_name != ref_name:
            raise ValueError(
                f"Spectrum {i} has spectral_axis_name={s.spectral_axis_name!r}, "
                f"expected {ref_name!r}."
            )
        if s.spectral_unit != ref_unit:
            raise ValueError(
                f"Spectrum {i} has spectral_unit={s.spectral_unit!r}, "
                f"expected {ref_unit!r}."
            )

    # Find overlapping axis range.
    common_min = float(max(float(np.min(s.axis)) for s in spectra))
    common_max = float(min(float(np.max(s.axis)) for s in spectra))
    if common_min >= common_max:
        raise ValueError(
            "No overlapping axis range found across the input spectra."
        )

    # Determine grid size.
    if n_points is None:
        n_points = int(np.median([s.n_points for s in spectra]))
    if n_points < 2:
        raise ValueError("Expected n_points to be at least 2.")

    common_axis = np.linspace(common_min, common_max, n_points)

    # Interpolate each spectrum onto the common axis.
    rows: list[npt.NDArray[np.float64]] = []
    for s in spectra:
        source_axis = np.asarray(s.axis, dtype=np.float64)
        source_intensity = np.asarray(s.intensity, dtype=np.float64)
        if source_axis[0] > source_axis[-1]:
            source_axis = source_axis[::-1]
            source_intensity = source_intensity[::-1]
        resampled = np.interp(common_axis, source_axis, source_intensity)
        rows.append(resampled)

    intensity = np.stack(rows, axis=0)

    provenance = ensure_provenance(None).append(
        ProvenanceStep(
            name="resample",
            parameters={
                "method": "common_axis",
                "n_spectra": len(spectra),
                "n_points": n_points,
                "common_min": common_min,
                "common_max": common_max,
            },
        )
    )

    return SpectrumCollection(
        axis=common_axis,
        intensity=intensity,
        provenance=provenance,
        spectral_axis_name=ref_name,
        spectral_unit=ref_unit,
    )
