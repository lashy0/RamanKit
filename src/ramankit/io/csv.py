"""CSV and TSV loader for tabular spectral data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ramankit.core.metadata import Metadata, Provenance
from ramankit.core.spectrum import Spectrum
from ramankit.io._provenance import build_load_provenance_step
from ramankit.io.base import BaseLoader

_AUTO_DELIMITERS = ("\t", ";", ",")


class CSVLoader(BaseLoader[Spectrum]):
    """Load a single spectrum from a delimited text file.

    Supports CSV, TSV, and semicolon-delimited files with optional header
    rows, configurable column selection, and locale-aware decimal separators.
    """

    format_name = "csv"
    supported_suffixes = (".csv", ".tsv")

    def __init__(
        self,
        *,
        axis_column: int | str = 0,
        intensity_column: int | str = 1,
        delimiter: str | None = None,
        skip_rows: int = 0,
        encoding: str = "utf-8",
        decimal: str = ".",
        spectral_axis_name: str | None = None,
        spectral_unit: str | None = None,
    ) -> None:
        self._axis_column = axis_column
        self._intensity_column = intensity_column
        self._delimiter = delimiter
        self._skip_rows = skip_rows
        self._encoding = encoding
        self._decimal = decimal
        self._spectral_axis_name = spectral_axis_name
        self._spectral_unit = spectral_unit

    def can_load(self, path: str | Path) -> bool:
        """Return True for .csv and .tsv files only."""
        suffix = Path(path).suffix.lower()
        return suffix in (".csv", ".tsv")

    def load(self, path: str | Path) -> Spectrum:
        """Load a spectrum from a delimited text file."""
        resolved = Path(path)
        lines = self._read_lines(resolved)

        if not lines:
            raise ValueError(f"File is empty or contains no data after skipping rows: {path}")

        effective_delimiter = self._detect_delimiter(lines[0])
        axis_index, intensity_index, data_start = self._resolve_columns(lines, effective_delimiter)

        axis_values: list[float] = []
        intensity_values: list[float] = []

        first_line_number = self._skip_rows + data_start + 1
        for line_number, line in enumerate(lines[data_start:], start=first_line_number):
            fields = line.split(effective_delimiter)
            if len(fields) <= max(axis_index, intensity_index):
                raise ValueError(
                    f"Row {line_number} has {len(fields)} fields, "
                    f"but column index {max(axis_index, intensity_index)} is required."
                )
            try:
                axis_values.append(float(self._normalize_decimal(fields[axis_index].strip())))
                intensity_values.append(float(self._normalize_decimal(fields[intensity_index].strip())))
            except ValueError:
                raise ValueError(
                    f"Non-numeric value at row {line_number}: "
                    f"axis={fields[axis_index].strip()!r}, "
                    f"intensity={fields[intensity_index].strip()!r}."
                ) from None

        if len(axis_values) < 2:
            raise ValueError(
                f"Expected at least 2 data points, got {len(axis_values)} in {path}."
            )

        provenance_step = build_load_provenance_step(
            step_name="load_csv",
            format_name="csv",
            file_type=resolved.suffix.lstrip("."),
            path=str(resolved),
            extra_parameters={
                "axis_column": self._axis_column,
                "intensity_column": self._intensity_column,
                "delimiter": effective_delimiter,
                "skip_rows": self._skip_rows,
                "decimal": self._decimal,
            },
        )

        return Spectrum(
            axis=np.array(axis_values, dtype=np.float64),
            intensity=np.array(intensity_values, dtype=np.float64),
            spectral_axis_name=self._spectral_axis_name,
            spectral_unit=self._spectral_unit,
            metadata=Metadata(),
            provenance=Provenance(source=str(resolved)).append(provenance_step),
        )

    def _read_lines(self, path: Path) -> list[str]:
        """Read non-empty lines from the file, skipping leading rows."""
        try:
            text = path.read_text(encoding=self._encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Cannot decode {path} with encoding {self._encoding!r}: {exc}"
            ) from None

        all_lines = text.splitlines()
        remaining = all_lines[self._skip_rows:]
        return [line for line in remaining if line.strip()]

    def _detect_delimiter(self, first_line: str) -> str:
        """Return the configured or auto-detected delimiter."""
        if self._delimiter is not None:
            return self._delimiter
        for candidate in _AUTO_DELIMITERS:
            if len(first_line.split(candidate)) >= 2:
                return candidate
        return ","

    def _resolve_columns(
        self,
        lines: list[str],
        delimiter: str,
    ) -> tuple[int, int, int]:
        """Resolve column names/indices and determine where data starts.

        Returns (axis_index, intensity_index, data_start_line).
        """
        needs_header = isinstance(self._axis_column, str) or isinstance(self._intensity_column, str)

        if needs_header:
            header_fields = [f.strip() for f in lines[0].split(delimiter)]
            axis_index = self._column_index(self._axis_column, header_fields, "axis_column")
            intensity_index = self._column_index(
                self._intensity_column, header_fields, "intensity_column",
            )
            return axis_index, intensity_index, 1

        assert isinstance(self._axis_column, int)
        assert isinstance(self._intensity_column, int)

        # Both are int — check if first line looks like a header (non-numeric)
        fields = lines[0].split(delimiter)
        try:
            float(self._normalize_decimal(fields[self._axis_column].strip()))
            # First line is data
            return self._axis_column, self._intensity_column, 0
        except (ValueError, IndexError):
            # First line is a header
            return self._axis_column, self._intensity_column, 1

    def _column_index(self, column: int | str, header: list[str], param_name: str) -> int:
        """Resolve a column specification to an integer index."""
        if isinstance(column, int):
            if column < 0 or column >= len(header):
                raise ValueError(
                    f"{param_name} index {column} is out of range for "
                    f"{len(header)} columns: {header}."
                )
            return column
        try:
            return header.index(column)
        except ValueError:
            raise ValueError(
                f"{param_name} {column!r} not found in header: {header}."
            ) from None

    def _normalize_decimal(self, value: str) -> str:
        """Replace locale decimal separator with a period."""
        if self._decimal != ".":
            return value.replace(self._decimal, ".")
        return value
