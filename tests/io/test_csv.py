from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ramankit import Spectrum
from ramankit.io import load
from ramankit.io.csv import CSVLoader

_CACHE_DIR = Path(".cache/test_csv")


def _fixture_path() -> Path:
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _write(name: str, content: str) -> Path:
    path = _fixture_path() / name
    path.write_text(content, encoding="utf-8")
    return path


def test_csv_loads_basic_comma_delimited() -> None:
    """Load a simple comma-delimited file with numeric column indices."""

    path = _write("basic.csv", "100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = CSVLoader().load(path)

    assert isinstance(spectrum, Spectrum)
    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spectrum.intensity, [1.0, 2.0, 3.0])


def test_csv_loads_tab_delimited() -> None:
    """Load a tab-delimited file using explicit delimiter."""

    path = _write("data.tsv", "100.0\t1.0\n200.0\t2.0\n300.0\t3.0\n")
    spectrum = CSVLoader(delimiter="\t").load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spectrum.intensity, [1.0, 2.0, 3.0])


def test_csv_loads_semicolon_delimited() -> None:
    """Load a semicolon-delimited file using explicit delimiter."""

    path = _write("data.csv", "100.0;1.0\n200.0;2.0\n300.0;3.0\n")
    spectrum = CSVLoader(delimiter=";").load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])


def test_csv_auto_detects_tab_delimiter() -> None:
    """Auto-detect tab as the delimiter when delimiter is None."""

    path = _write("auto.csv", "100.0\t1.0\n200.0\t2.0\n300.0\t3.0\n")
    spectrum = CSVLoader().load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spectrum.intensity, [1.0, 2.0, 3.0])


def test_csv_loads_with_named_columns() -> None:
    """Resolve columns by name from a header row."""

    content = "Wavenumber,Signal,Extra\n100.0,1.0,9.0\n200.0,2.0,8.0\n300.0,3.0,7.0\n"
    path = _write("named.csv", content)
    spectrum = CSVLoader(axis_column="Wavenumber", intensity_column="Signal").load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spectrum.intensity, [1.0, 2.0, 3.0])


def test_csv_skips_leading_rows() -> None:
    """Skip metadata lines at the top of the file before data."""

    content = "# metadata line 1\n# metadata line 2\n100.0,1.0\n200.0,2.0\n300.0,3.0\n"
    path = _write("skip.csv", content)
    spectrum = CSVLoader(skip_rows=2).load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])


def test_csv_handles_comma_decimal() -> None:
    """Parse European-style comma decimal separators."""

    path = _write("euro.csv", "100,5\t1,2\n200,5\t2,3\n300,5\t3,4\n")
    spectrum = CSVLoader(delimiter="\t", decimal=",").load(path)

    np.testing.assert_allclose(spectrum.axis, [100.5, 200.5, 300.5])
    np.testing.assert_allclose(spectrum.intensity, [1.2, 2.3, 3.4])


def test_csv_propagates_spectral_axis_metadata() -> None:
    """Set spectral_axis_name and spectral_unit on the returned Spectrum."""

    path = _write("meta.csv", "100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = CSVLoader(spectral_axis_name="raman_shift", spectral_unit="cm^-1").load(path)

    assert spectrum.spectral_axis_name == "raman_shift"
    assert spectrum.spectral_unit == "cm^-1"


def test_csv_records_provenance() -> None:
    """Record the load step with parameters in provenance."""

    path = _write("prov.csv", "100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = CSVLoader().load(path)

    assert spectrum.provenance.source == str(path)
    steps = spectrum.provenance.steps
    assert len(steps) == 1
    assert steps[0].name == "load_csv"
    assert steps[0].parameters["format"] == "csv"
    assert steps[0].parameters["axis_column"] == 0
    assert steps[0].parameters["intensity_column"] == 1


def test_csv_raises_for_unknown_column_name() -> None:
    """Reject a column name that does not exist in the header."""

    path = _write("bad_col.csv", "A,B\n1.0,2.0\n3.0,4.0\n")
    with pytest.raises(ValueError, match="not found in header"):
        CSVLoader(axis_column="Missing").load(path)


def test_csv_raises_for_non_numeric_data() -> None:
    """Reject rows containing non-numeric values in selected columns."""

    path = _write("bad_data.csv", "100.0,1.0\n200.0,abc\n")
    with pytest.raises(ValueError, match="Non-numeric value"):
        CSVLoader().load(path)


def test_csv_raises_for_empty_file() -> None:
    """Reject an empty file."""

    path = _write("empty.csv", "")
    with pytest.raises(ValueError, match="empty"):
        CSVLoader().load(path)


def test_csv_raises_for_too_few_data_points() -> None:
    """Reject a file with fewer than 2 data points."""

    path = _write("one.csv", "100.0,1.0\n")
    with pytest.raises(ValueError, match="at least 2"):
        CSVLoader().load(path)


def test_csv_registry_auto_detect() -> None:
    """Resolve .csv suffix to CSVLoader through the built-in registry."""

    path = _write("auto.csv", "100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = load(str(path))

    assert isinstance(spectrum, Spectrum)
    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])


def test_csv_registry_explicit_format_for_txt() -> None:
    """Load a .txt file explicitly as CSV format through the registry."""

    path = _write("data.txt", "100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = load(str(path), format="csv")

    assert isinstance(spectrum, Spectrum)
    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])


def test_csv_skips_header_when_columns_are_int() -> None:
    """Auto-detect and skip a header row when using integer column indices."""

    path = _write("header.csv", "wavenumber,intensity\n100.0,1.0\n200.0,2.0\n300.0,3.0\n")
    spectrum = CSVLoader().load(path)

    np.testing.assert_array_equal(spectrum.axis, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spectrum.intensity, [1.0, 2.0, 3.0])
