from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from astropy.io import fits

from gtellcorr.data.observation.header import (
    HeaderField,
    HeaderSchema,
    parse_sexagesimal,
    read_on_site_weather,
    read_scalar,
    read_site_time,
    read_time,
)
from gtellcorr.errors import ConfigError, DataError


def test_sexagesimal() -> None:
    assert parse_sexagesimal("+32:42:04.71") == pytest.approx(32.70130833)
    assert parse_sexagesimal("-07:19:33.38") == pytest.approx(-7.32593889)
    assert parse_sexagesimal("12:30") == 12.5
    assert parse_sexagesimal("5") == 5.0


def test_read_scalar_formats_and_fallthrough() -> None:
    h = fits.Header({"A": "", "B": "x", "C": 3.5, "LON": "-07:19:33.38"})
    assert read_scalar(h, HeaderField(keys=("MISSING", "A", "B", "C"), fmt="deg")) == 3.5
    assert read_scalar(h, HeaderField(keys=("LON",), fmt="hms_hours")) == pytest.approx(
        -109.88908333
    )
    assert read_scalar(h, HeaderField()) is None
    assert read_scalar(None, HeaderField(keys=("C",), fmt="deg")) is None


def test_header_field_validation() -> None:
    with pytest.raises(ConfigError, match="fmt: required"):
        HeaderField(keys=("A",))
    with pytest.raises(ConfigError, match="exactly"):
        HeaderField(keys=("DATE",), fmt="date+time")


def test_read_time_variants() -> None:
    iso = HeaderField(keys=("DATE-OBS",), fmt="iso", jd_keys=("MJD-OBS",))
    t = read_time(fits.Header({"DATE-OBS": "2022-02-13T07:30:05Z"}), iso)
    assert t == datetime(2022, 2, 13, 7, 30, 5, tzinfo=UTC)
    split = HeaderField(keys=("DATE-OBS", "TIME-OBS"), fmt="date+time")
    t = read_time(fits.Header({"DATE-OBS": "2025-01-01", "TIME-OBS": "03:00:00"}), split)
    assert t == datetime(2025, 1, 1, 3, 0, 0, tzinfo=UTC)
    # unparsable -> Julian-date fallback (plain JD and MJD accepted)
    t = read_time(fits.Header({"DATE-OBS": "garbage", "MJD-OBS": 59623.5}), iso)
    assert t == datetime(2022, 2, 13, 12, 0, tzinfo=UTC)
    jd = HeaderField(keys=("X",), fmt="iso", jd_keys=("JD",))
    assert read_time(fits.Header({"JD": 2459624.0}), jd) == datetime(2022, 2, 13, 12, 0, tzinfo=UTC)
    assert read_time(fits.Header({}), iso) is None
    assert read_time(None, iso) is None
    bad_split = fits.Header({"DATE-OBS": "2025-13-40"})
    assert read_time(bad_split, split) is None


def test_pepsi_has_no_fallback_epoch() -> None:
    # JD-OBS is mid-exposure and JD-TDB barycentric; neither may stand in for the start
    from gtellcorr.data.observation.instrument import instruments

    time = instruments.get("LBT/PEPSI").header.time
    header = fits.Header({"JD-OBS": 2460676.627315, "JD-TDB": 2460676.629290})
    assert read_time(header, time) is None
    header.update({"DATE-OBS": "2025-01-01", "TIME-OBS": "03:00:00.500"})
    assert read_time(header, time) == datetime(2025, 1, 1, 3, 0, 0, 500000, tzinfo=UTC)


SCHEMA = HeaderSchema(
    lat=HeaderField(keys=("LAT",), fmt="deg"),
    lon=HeaderField(keys=("LON",), fmt="deg"),
    alt_m=HeaderField(keys=("ALT",), fmt="m"),
    time=HeaderField(keys=("DATE-OBS",), fmt="iso"),
    exptime=HeaderField(keys=("EXPTIME",), fmt="s"),
    airmass=HeaderField(keys=("AIRMASS",), fmt="float"),
    asm_T=HeaderField(keys=("T",), fmt="degC"),
    asm_dew=HeaderField(keys=("DEW",), fmt="degC"),
    asm_time=HeaderField(keys=("TM",), fmt="iso"),
)


def test_site_time_mid_exposure_and_field_fallback() -> None:
    h = fits.Header(
        {"LAT": 10.0, "DATE-OBS": "2022-01-01T00:00:00", "EXPTIME": 600.0, "AIRMASS": 1.2}
    )
    st = read_site_time(h, SCHEMA, mid_exposure=True, fallback=(1.0, 2.0, 3000.0))
    assert (st.lat_deg, st.lon_deg, st.alt_m) == (10.0, 2.0, 3000.0)
    assert st.sources == {"lat_deg": "header", "lon_deg": "fallback", "alt_m": "fallback"}
    assert st.time_utc == datetime(2022, 1, 1, tzinfo=UTC) + timedelta(seconds=300)
    assert st.airmass == 1.2
    start = read_site_time(h, SCHEMA, mid_exposure=False, fallback=(1.0, 2.0, 3.0))
    assert start.time_utc == datetime(2022, 1, 1, tzinfo=UTC)
    with pytest.raises(DataError, match="no lon_deg and no fallback"):
        read_site_time(h, SCHEMA, mid_exposure=True, fallback=None)


def test_on_site_weather() -> None:
    w = read_on_site_weather(
        fits.Header({"T": 10.0, "DEW": -5.0, "TM": "2022-01-01T00:00:00"}), SCHEMA
    )
    assert (w.T_K, w.dewpoint_K, w.RH_pct, w.P_hpa) == (283.15, 268.15, None, None)
    assert w.time == "2022-01-01T00:00:00"
    empty = read_on_site_weather(None, SCHEMA)
    assert empty.T_K is None and empty.time is None
