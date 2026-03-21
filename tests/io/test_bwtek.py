from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ramankit.core.spectrum import Spectrum
from ramankit.io import load as io_load
from ramankit.io.bwtek import BWTekLoader

_META_BLOCK = """\
File Version;BWRam4.15_10
Date;2025-02-19 12:06:04
title;BWS465-785S
model;BTC665N-785S-SYS
operator;
laser_wavelength;785,022475277658
intigration times(us);500000
average number;5
"""

_META_BLOCK_NO_MODEL = """\
File Version;BWRam4.15_10
Date;2025-02-19 12:06:04
title;BWS465-785S
laser_wavelength;785,022475277658
intigration times(us);500000
average number;5
"""

_META_BLOCK_WITH_BLANKS = """\
File Version;BWRam4.15_10

Date;2025-02-19 12:06:04

title;BWS465-785S
model;BTC665N-785S-SYS

laser_wavelength;785,022475277658
intigration times(us);500000
average number;5

"""

_HEADER_LINE = (
    "Pixel;Wavelength;Wavenumber;Raman Shift;Dark;Reference;"
    "Raw data #1;Dark Subtracted #1;%TR #1;Absorbance #1;"
)

_DATA_ROWS = """\
0;782,832473500501;12774,1251653552;-35,6363769124728;1093;65535;1100,4;7,4;0;0;
1;782,984614364625;12771,6430393907;-33,1542509480059;1101,8;65535;1104,2;2,4;0;0;
2;783,136746409959;12769,1620216288;-30,6732331860894;1092;65535;1103,6;11,6;0;0;
"""

_DATA_ROWS_BAD = """\
0;782,832473500501;12774,1251653552;-35,6363769124728;1093;65535;1100,4;7,4;0;0;
1;782,984614364625;12771,6430393907;INVALID;1101,8;65535;1104,2;2,4;0;0;
2;783,136746409959;12769,1620216288;-30,6732331860894;1092;65535;1103,6;11,6;0;0;
"""

_CACHE_DIR = Path(".cache/test_bwtek")


def _fixture_path(name: str) -> Path:
    """Return a stable cache-local path for a named fixture file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _CACHE_DIR / name
    if p.exists():
        p.unlink()
    return p


def _write_fixture(
    name: str = "spectrum.txt",
    *,
    meta: str = _META_BLOCK,
    data: str = _DATA_ROWS,
    encoding: str = "utf-8",
) -> Path:
    """Write a BWTek fixture file from blocks and return its path."""
    p = _fixture_path(name)
    p.write_text(meta + _HEADER_LINE + "\n" + data, encoding=encoding)
    return p


def test_bwtek_load_returns_spectrum() -> None:
    """Load a BWTek file and confirm the result is a Spectrum."""
    assert isinstance(BWTekLoader().load(_write_fixture()), Spectrum)


def test_bwtek_load_axis_values() -> None:
    """Raman Shift column maps to the spectral axis."""
    result = BWTekLoader().load(_write_fixture())
    expected = np.array([-35.6363769124728, -33.1542509480059, -30.6732331860894])
    np.testing.assert_allclose(result.axis, expected)


def test_bwtek_load_intensity_values() -> None:
    """Dark Subtracted #1 column maps to the intensity."""
    result = BWTekLoader().load(_write_fixture())
    np.testing.assert_allclose(result.intensity, np.array([7.4, 2.4, 11.6]))


def test_bwtek_load_row_count() -> None:
    """Axis and intensity lengths match the number of data rows."""
    result = BWTekLoader().load(_write_fixture())
    assert len(result.axis) == 3
    assert len(result.intensity) == 3


def test_bwtek_axis_dtype_is_float64() -> None:
    """The spectral axis array has dtype float64."""
    assert BWTekLoader().load(_write_fixture()).axis.dtype == np.float64


def test_bwtek_intensity_dtype_is_float64() -> None:
    """The intensity array has dtype float64."""
    assert BWTekLoader().load(_write_fixture()).intensity.dtype == np.float64


def test_bwtek_comma_decimal_separator_parsed() -> None:
    """Values using a comma decimal separator parse to correct floats."""
    result = BWTekLoader().load(_write_fixture())
    assert pytest.approx(result.axis[0], rel=1e-6) == -35.6363769124728


def test_bwtek_load_accepts_string_path() -> None:
    """load() accepts a plain str path, not only pathlib.Path."""
    path = _write_fixture("str_path.txt")
    result = BWTekLoader().load(str(path))
    assert isinstance(result, Spectrum)
    assert len(result.axis) == 3


def test_bwtek_metadata_instrument_from_model() -> None:
    """instrument is populated from the model key."""
    assert BWTekLoader().load(_write_fixture()).metadata.instrument == "BTC665N-785S-SYS"


def test_bwtek_metadata_instrument_falls_back_to_title() -> None:
    """instrument falls back to title when model key is absent."""
    path = _write_fixture("no_model.txt", meta=_META_BLOCK_NO_MODEL)
    assert BWTekLoader().load(path).metadata.instrument == "BWS465-785S"


def test_bwtek_metadata_acquisition_contains_date() -> None:
    """acquisition string includes the capture date."""
    acq = BWTekLoader().load(_write_fixture()).metadata.acquisition
    assert acq is not None
    assert "2025-02-19 12:06:04" in acq


def test_bwtek_metadata_acquisition_contains_integration_time() -> None:
    """acquisition string includes the integration time."""
    acq = BWTekLoader().load(_write_fixture()).metadata.acquisition or ""
    assert "500000" in acq


def test_bwtek_metadata_laser_wavelength_is_normalized() -> None:
    """laser_wavelength is parsed into the normalized metadata field."""

    result = BWTekLoader().load(_write_fixture())

    assert result.metadata.laser_wavelength == pytest.approx(785.022475277658)


def test_bwtek_extracted_keys_absent_from_extras() -> None:
    """B&W Tek metadata normalization does not leak raw keys into extras."""

    assert BWTekLoader().load(_write_fixture()).metadata.extras == {}


def test_bwtek_metadata_sample_is_none() -> None:
    """sample is not set by the loader."""
    assert BWTekLoader().load(_write_fixture()).metadata.sample is None


def test_bwtek_operator_is_normalized() -> None:
    """A metadata line with no value after the separator maps to empty operator string."""

    assert BWTekLoader().load(_write_fixture()).metadata.operator == ""


def test_bwtek_metadata_exposure_time_is_normalized() -> None:
    """The configured exposure/integration time is parsed into normalized metadata."""

    assert BWTekLoader().load(_write_fixture()).metadata.exposure_time == 500000.0


def test_bwtek_metadata_accumulations_is_normalized() -> None:
    """The average number field is parsed into normalized metadata."""

    assert BWTekLoader().load(_write_fixture()).metadata.accumulations == 5


def test_bwtek_metadata_acquisition_datetime_is_normalized() -> None:
    """The acquisition timestamp is preserved in a dedicated normalized field."""

    result = BWTekLoader().load(_write_fixture())

    assert result.metadata.acquisition_datetime == "2025-02-19 12:06:04"


def test_bwtek_raw_vendor_metadata_is_preserved_unchanged() -> None:
    """Original vendor metadata is preserved separately from normalized fields."""

    metadata = BWTekLoader().load(_write_fixture()).metadata

    assert metadata.raw_vendor_metadata["laser_wavelength"] == "785,022475277658"
    assert metadata.raw_vendor_metadata["operator"] == ""


def test_bwtek_blank_metadata_lines_are_skipped() -> None:
    """Blank lines interspersed in the metadata block do not cause errors."""
    result = BWTekLoader().load(_write_fixture("blanks.txt", meta=_META_BLOCK_WITH_BLANKS))
    assert result.metadata.instrument == "BTC665N-785S-SYS"
    assert len(result.axis) == 3


def test_bwtek_provenance_source_is_absolute_path() -> None:
    """provenance.source is the resolved absolute path of the file."""
    path = _write_fixture()
    assert BWTekLoader().load(path).provenance.source == str(path.resolve())


def test_bwtek_provenance_single_step() -> None:
    """Exactly one provenance step named load_bwtek is recorded."""
    steps = BWTekLoader().load(_write_fixture()).provenance.steps
    assert len(steps) == 1
    assert steps[0].name == "load_bwtek"


def test_bwtek_provenance_parameters_recorded() -> None:
    """Loader parameters are stored in the provenance step."""
    path = _write_fixture()
    params = BWTekLoader().load(path).provenance.steps[0].parameters
    assert params["format"] == "bwtek"
    assert params["vendor"] == "bwtek"
    assert params["file_type"] == "bwram_txt"
    assert params["axis_column"] == "Raman Shift"
    assert params["intensity_column"] == "Dark Subtracted #1"
    assert params["encoding"] == "utf-8"
    assert params["path"] == str(path.resolve())


def test_bwtek_custom_encoding_recorded_in_provenance() -> None:
    """Non-default encoding is stored in the provenance step parameters."""
    path = _write_fixture("latin1.txt", encoding="latin-1")
    params = BWTekLoader(encoding="latin-1").load(path).provenance.steps[0].parameters
    assert params["encoding"] == "latin-1"


def test_bwtek_spectral_axis_name() -> None:
    """spectral_axis_name is set to raman_shift."""
    assert BWTekLoader().load(_write_fixture()).spectral_axis_name == "raman_shift"


def test_bwtek_spectral_unit() -> None:
    """spectral_unit is set to cm^-1."""
    assert BWTekLoader().load(_write_fixture()).spectral_unit == "cm^-1"


def test_bwtek_public_io_load_supports_explicit_format() -> None:
    """Top-level loading supports explicit selection of the bwtek loader."""

    result = io_load(_write_fixture(), format="bwtek")

    assert isinstance(result, Spectrum)


def test_bwtek_public_io_load_auto_detects_txt_suffix() -> None:
    """Top-level loading auto-detects B&W Tek TXT files by suffix."""

    result = io_load(_write_fixture())

    assert isinstance(result, Spectrum)


def test_bwtek_custom_axis_column_wavenumber() -> None:
    """axis_column=Wavenumber loads absolute wavenumber values."""
    result = BWTekLoader(axis_column="Wavenumber").load(_write_fixture())
    expected = np.array([12774.1251653552, 12771.6430393907, 12769.1620216288])
    np.testing.assert_allclose(result.axis, expected)


def test_bwtek_custom_intensity_column_raw_data() -> None:
    """intensity_column='Raw data #1' loads unprocessed detector counts."""
    result = BWTekLoader(intensity_column="Raw data #1").load(_write_fixture())
    np.testing.assert_allclose(result.intensity, np.array([1100.4, 1104.2, 1103.6]))


def test_bwtek_raises_file_not_found() -> None:
    """FileNotFoundError is raised when the path does not exist."""
    with pytest.raises(FileNotFoundError):
        BWTekLoader().load(_CACHE_DIR / "definitely_does_not_exist.txt")


def test_bwtek_raises_when_header_line_missing() -> None:
    """ValueError is raised when no column-header line is found."""
    p = _fixture_path("no_header.txt")
    p.write_text("key1;value1\nkey2;value2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column header line not found"):
        BWTekLoader().load(p)


def test_bwtek_raises_for_unknown_axis_column() -> None:
    """ValueError names the missing column and lists available columns."""
    with pytest.raises(ValueError, match="Nonexistent") as exc_info:
        BWTekLoader(axis_column="Nonexistent").load(_write_fixture())
    assert "Raman Shift" in str(exc_info.value)


def test_bwtek_raises_for_unknown_intensity_column() -> None:
    """ValueError is raised when intensity_column is not present."""
    with pytest.raises(ValueError, match="Nonexistent"):
        BWTekLoader(intensity_column="Nonexistent").load(_write_fixture())


def test_bwtek_raises_on_unicode_decode_error() -> None:
    """ValueError is raised when the file cannot be decoded."""
    p = _fixture_path("bad_encoding.txt")
    p.write_bytes(b"\xff\xfe invalid utf-8 \x80")
    with pytest.raises(ValueError, match="cannot decode"):
        BWTekLoader(encoding="utf-8").load(p)


def test_bwtek_non_numeric_data_row_raises_with_row_number() -> None:
    """ValueError for bad data includes the 1-based row number."""
    with pytest.raises(ValueError, match=r"row 2"):
        BWTekLoader().load(_write_fixture("bad_row.txt", data=_DATA_ROWS_BAD))


def test_bwtek_empty_data_block_raises() -> None:
    """A file with header but no data rows raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        BWTekLoader().load(_write_fixture("empty_data.txt", data=""))
