from __future__ import annotations

from pathlib import Path

import numpy as np

from ramankit.core.metadata import Metadata, Provenance
from ramankit.core.spectrum import Spectrum
from ramankit.io._provenance import build_load_provenance_step
from ramankit.io.base import BaseLoader

_HEADER_SENTINEL = "Pixel"


class BWTekLoader(BaseLoader[Spectrum]):
    """Load a single Raman spectrum from a B&W Tek TXT export (BWRam format).

    B&W Tek files use a semicolon field separator and a comma decimal separator
    (European locale).  A block of ``key;value`` metadata lines precedes a
    column-header line whose first token is ``Pixel``, followed by data rows.

    Args:
        axis_column: Name of the column to use as the spectral axis.
            Defaults to ``"Raman Shift"`` (cm⁻¹ relative to the excitation
            laser).
        intensity_column: Name of the column to use as the intensity signal.
            Defaults to ``"Dark Subtracted #1"``.
        encoding: Text encoding of the file.  Defaults to ``"utf-8"``.
    """

    format_name = "bwtek"
    supported_suffixes = (".txt",)

    def __init__(
        self,
        *,
        axis_column: str = "Raman Shift",
        intensity_column: str = "Dark Subtracted #1",
        encoding: str = "utf-8",
    ) -> None:
        self._axis_column = axis_column
        self._intensity_column = intensity_column
        self._encoding = encoding

    def load(self, path: str | Path) -> Spectrum:
        """Load one spectrum from a B&W Tek TXT export file.

        Args:
            path: Path to the ``.txt`` file produced by BWRam software.

        Returns:
            A :class:`~ramankit.core.spectrum.Spectrum` with the spectral axis
            in cm⁻¹ (``spectral_axis_name="raman_shift"``,
            ``spectral_unit="cm^-1"``).

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be decoded, the column-header line
                is missing, or a required column is absent.
        """
        path = Path(path)

        try:
            raw_text = path.read_text(encoding=self._encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"BWTekLoader: cannot decode '{path}' with encoding '{self._encoding}'."
            ) from exc

        lines = raw_text.splitlines()

        header_idx: int | None = None
        for i, line in enumerate(lines):
            first_token = line.split(";", maxsplit=1)[0].strip()
            if first_token == _HEADER_SENTINEL:
                header_idx = i
                break

        if header_idx is None:
            raise ValueError("BWTekLoader: column header line not found.")

        meta_lines = lines[:header_idx]
        header_line = lines[header_idx]
        data_lines = [ln for ln in lines[header_idx + 1:] if ln.strip()]

        raw_meta: dict[str, str] = {}
        for line in meta_lines:
            if not line.strip():
                continue
            parts = line.split(";", maxsplit=1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            raw_meta[key] = value

        instrument = raw_meta.get("model") or raw_meta.get("title")
        acquisition = (
            f"Date: {raw_meta.get('Date')}, "
            f"Integration: {raw_meta.get('intigration times(us)')} us, "
            f"Averages: {raw_meta.get('average number')}"
        )

        col_names = [c.strip() for c in header_line.split(";") if c.strip()]

        def _require_column(name: str) -> int:
            """Return the index of *name* in *col_names* or raise ValueError."""
            try:
                return col_names.index(name)
            except ValueError:
                available = ", ".join(f'"{c}"' for c in col_names)
                raise ValueError(
                    f"BWTekLoader: column '{name}' not found. "
                    f"Available columns: {available}."
                ) from None

        axis_idx = _require_column(self._axis_column)
        intensity_idx = _require_column(self._intensity_column)

        axis_values: list[float] = []
        intensity_values: list[float] = []

        for row_num, line in enumerate(data_lines, start=1):
            tokens = line.replace(",", ".").split(";")
            try:
                axis_values.append(float(tokens[axis_idx]))
                intensity_values.append(float(tokens[intensity_idx]))
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"BWTekLoader: cannot parse numeric value on data row {row_num} "
                    f"(axis column index {axis_idx}, intensity column index {intensity_idx}): "
                    f"{exc}"
                ) from exc

        axis_array = np.array(axis_values, dtype=np.float64)
        intensity_array = np.array(intensity_values, dtype=np.float64)

        metadata = Metadata(
            sample=None,
            instrument=instrument,
            acquisition=acquisition,
            laser_wavelength=_parse_locale_float(raw_meta.get("laser_wavelength")),
            grating=_optional_string(raw_meta.get("grating")),
            exposure_time=_parse_locale_float(raw_meta.get("intigration times(us)")),
            accumulations=_parse_locale_int(raw_meta.get("average number")),
            objective=_optional_string(raw_meta.get("objective")),
            acquisition_datetime=_optional_string(raw_meta.get("Date")),
            operator=_optional_string(raw_meta.get("operator")),
            raw_vendor_metadata=raw_meta,
        )

        resolved = str(path.resolve())
        provenance = Provenance(
            source=resolved,
            steps=(
                build_load_provenance_step(
                    "load_bwtek",
                    format_name=self.format_name,
                    vendor="bwtek",
                    file_type="bwram_txt",
                    path=resolved,
                    extra_parameters={
                        "axis_column": self._axis_column,
                        "intensity_column": self._intensity_column,
                        "encoding": self._encoding,
                    },
                    description="Loaded spectrum from B&W Tek TXT export.",
                ),
            ),
        )

        return Spectrum(
            axis=axis_array,
            intensity=intensity_array,
            metadata=metadata,
            provenance=provenance,
            spectral_axis_name="raman_shift",
            spectral_unit="cm^-1",
        )

    def can_load(self, path: str | Path) -> bool:
        """Return True when a small header read looks like B&W Tek text export."""

        candidate = Path(path)
        if candidate.is_dir():
            return False
        try:
            with candidate.open("r", encoding=self._encoding) as handle:
                header = handle.read(4096)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return False
        return any(
            line.split(";", maxsplit=1)[0].strip() == _HEADER_SENTINEL
            for line in header.splitlines()
        )


def _optional_string(value: str | None) -> str | None:
    return value


def _parse_locale_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value.replace(",", "."))


def _parse_locale_int(value: str | None) -> int | None:
    parsed = _parse_locale_float(value)
    if parsed is None:
        return None
    return int(parsed)
