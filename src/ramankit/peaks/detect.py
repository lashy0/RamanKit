from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks as scipy_find_peaks  # type: ignore[import-untyped]

from ramankit.core.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class DetectedPeak:
    """Store one detected peak in spectral-axis coordinates."""

    index: int
    position: float
    height: float
    prominence: float | None = None
    width: float | None = None
    left_base: int | None = None
    right_base: int | None = None


@dataclass(frozen=True, slots=True)
class PeakDetectionResult:
    """Store the typed output of one peak-detection pass."""

    peaks: tuple[DetectedPeak, ...]

    def __len__(self) -> int:
        """Return the number of detected peaks."""

        return len(self.peaks)

    def __iter__(self) -> Iterator[DetectedPeak]:
        """Iterate over detected peaks in spectral-axis order."""

        return iter(self.peaks)

    def __getitem__(self, item: int) -> DetectedPeak:
        """Return one detected peak by index."""

        return self.peaks[item]

    @property
    def indices(self) -> npt.NDArray[np.int64]:
        """Return detected peak indices along the sampled spectrum."""

        return np.asarray([peak.index for peak in self.peaks], dtype=np.int64)

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Return detected peak positions in spectral-axis coordinates."""

        return np.asarray([peak.position for peak in self.peaks], dtype=np.float64)

    @property
    def heights(self) -> npt.NDArray[np.float64]:
        """Return peak heights sampled from the original intensity array."""

        return np.asarray([peak.height for peak in self.peaks], dtype=np.float64)


def find_peaks(
    spectrum: Spectrum,
    *,
    height: float | tuple[float, float] | None = None,
    prominence: float | tuple[float, float] | None = None,
    width: float | tuple[float, float] | None = None,
    distance: float | None = None,
    threshold: float | tuple[float, float] | None = None,
) -> PeakDetectionResult:
    """Detect peaks in one Raman spectrum using SciPy peak finding."""

    indices, properties = scipy_find_peaks(
        spectrum.intensity,
        height=height,
        prominence=prominence,
        width=width,
        distance=distance,
        threshold=threshold,
    )
    peak_indices = np.asarray(indices, dtype=np.int64)
    heights = spectrum.intensity[peak_indices].astype(np.float64, copy=False)
    positions = spectrum.axis[peak_indices].astype(np.float64, copy=False)

    prominences = _optional_float_array(properties, "prominences")
    left_bases = _optional_int_array(properties, "left_bases")
    right_bases = _optional_int_array(properties, "right_bases")
    widths = _optional_widths(properties, spectrum.axis)

    peaks = tuple(
        DetectedPeak(
            index=int(peak_indices[index]),
            position=float(positions[index]),
            height=float(heights[index]),
            prominence=None if prominences is None else float(prominences[index]),
            width=None if widths is None else float(widths[index]),
            left_base=None if left_bases is None else int(left_bases[index]),
            right_base=None if right_bases is None else int(right_bases[index]),
        )
        for index in range(peak_indices.size)
    )
    return PeakDetectionResult(peaks=peaks)


def _optional_float_array(
    properties: dict[str, npt.ArrayLike],
    key: str,
) -> npt.NDArray[np.float64] | None:
    values = properties.get(key)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _optional_int_array(
    properties: dict[str, npt.ArrayLike],
    key: str,
) -> npt.NDArray[np.int64] | None:
    values = properties.get(key)
    if values is None:
        return None
    return np.asarray(values, dtype=np.int64)


def _optional_widths(
    properties: dict[str, npt.ArrayLike],
    axis: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    left_ips = properties.get("left_ips")
    right_ips = properties.get("right_ips")
    if left_ips is None or right_ips is None:
        return None

    left_positions = _interpolate_axis_positions(axis, np.asarray(left_ips, dtype=np.float64))
    right_positions = _interpolate_axis_positions(axis, np.asarray(right_ips, dtype=np.float64))
    return np.abs(right_positions - left_positions)


def _interpolate_axis_positions(
    axis: npt.NDArray[np.float64],
    sample_positions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    sample_indices = np.arange(axis.shape[0], dtype=np.float64)
    return np.interp(sample_positions, sample_indices, axis)
