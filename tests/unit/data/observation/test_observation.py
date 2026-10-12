from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gtellcorr.data.observation.config import DataConfig, SiteCoords
from gtellcorr.data.observation.observation import Observation, read_token_file
from gtellcorr.data.observation.target import Star
from gtellcorr.errors import DataError, RegistryError


def _night(make: Callable[..., Path], tmp_path: Path) -> Path:
    for i, t in enumerate(["073005", "074032", "075100", "131533"]):
        make(
            f"neidL2_20220213T{t}.fits",
            DATE_OBS=None,
            **{"DATE-OBS": f"2022-02-13T{t[:2]}:{t[2:4]}:{t[4:]}"},
        )
        del i
    (tmp_path / "notes.txt").write_text("not a spectrum")
    return tmp_path


def test_from_directory_gathers_and_filters(make_neid: Callable[..., Path], tmp_path: Path) -> None:
    root = _night(make_neid, tmp_path)
    obs = Observation.from_directory(root, instrument="WIYN/NEID", exclude=["T131533"])
    assert [e.name for e in obs.exposures] == [
        "neidL2_20220213T073005.fits",
        "neidL2_20220213T074032.fits",
        "neidL2_20220213T075100.fits",
    ]
    e = obs.exposures[0]
    assert e.time_utc == datetime(2022, 2, 13, 7, 35, 5, tzinfo=UTC)  # + EXPTIME/2
    assert e.site.coords == (31.958092, -111.600562, 2091.0)
    assert set(e.site_sources.values()) == {"header"}
    assert (e.airmass, e.airmass_source, e.resolution) == (1.9, "fits_header", 110000.0)
    assert pytest.approx(286.55) == e.weather.T_K
    assert len(obs) == 3 and "3 exposures" in repr(obs)
    only = obs.select(include=["T0740"])
    assert [x.name for x in only.exposures] == ["neidL2_20220213T074032.fits"]
    with pytest.raises(DataError, match="leaves no exposures"):
        obs.select(include=["nothing"])


def test_include_file_and_config(make_neid: Callable[..., Path], tmp_path: Path) -> None:
    root = _night(make_neid, tmp_path)
    tokens = tmp_path / "keep.txt"
    tokens.write_text("# night 1\nT073005  # first\n\nT075100\n")
    assert read_token_file(tokens) == ("T073005", "T075100")
    cfg = DataConfig(
        directory=str(root),
        include_file=str(tokens),
        target={"name": "TOI-2076"},  # type: ignore[arg-type]
    )
    obs = Observation.from_config(cfg)
    assert len(obs) == 2 and obs.instrument.name == "WIYN/NEID"  # identified from headers
    assert obs.target == Star(name="TOI-2076")


def test_site_override_and_fallbacks(make_neid: Callable[..., Path], tmp_path: Path) -> None:
    make_neid(OBSLAT=None, OBSALT=None)
    obs = Observation.from_directory(tmp_path, instrument="NEID")
    e = obs.exposures[0]
    assert e.site_sources == {
        "lat_deg": "instrument_default",
        "lon_deg": "header",
        "alt_m": "instrument_default",
    }
    assert e.site.lat_deg == 31.95805  # WIYN
    obs = Observation.from_directory(tmp_path, instrument="NEID", fallback_site="Kitt Peak")
    assert obs.exposures[0].site.lat_deg == 31.96333
    assert obs.exposures[0].site_sources["lat_deg"] == "fallback"
    obs = Observation.from_directory(
        tmp_path, instrument="NEID", site=SiteCoords(lat_deg=1, lon_deg=2, alt_m=3)
    )
    assert obs.exposures[0].site.coords == (1.0, 2.0, 3.0)
    assert set(obs.exposures[0].site_sources.values()) == {"override"}


def test_airmass_fallback_is_flagged(
    make_neid: Callable[..., Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_neid(AIRMASS=0.5)
    with caplog.at_level(logging.WARNING, logger="gtellcorr"):
        obs = Observation.from_directory(tmp_path, instrument="NEID")
    e = obs.exposures[0]
    assert (e.airmass, e.airmass_source) == (1.5, "fallback_constant")
    assert "without a usable header airmass" in caplog.text


def test_missing_epoch_is_an_error(make_neid: Callable[..., Path], tmp_path: Path) -> None:
    make_neid(**{"DATE-OBS": None})
    with pytest.raises(DataError, match="no usable epoch"):
        Observation.from_directory(tmp_path, instrument="NEID")


def test_science_filter_and_channels(make_pepsi: Callable[..., Path], tmp_path: Path) -> None:
    make_pepsi("pepsir.20250101.001.dxt.nor")
    make_pepsi("pepsir.20250101.002.dxt.nor", IMAGETYP="bias")
    make_pepsi("pepsir.20250101.003.dxt.nor", IMAGETYP=None)  # empty card passes
    make_pepsi("pepsib.20250101.001.dxt.nor")
    obs = Observation.from_directory(tmp_path)
    assert obs.instrument.name == "LBT/PEPSI" and obs.channel == "red"
    assert [e.name for e in obs.exposures] == [
        "pepsir.20250101.001.dxt.nor",
        "pepsir.20250101.003.dxt.nor",
    ]
    assert obs.exposures[0].resolution == 130000.0
    blue = Observation.from_directory(tmp_path, instrument="PEPSI", channel="blue")
    assert [e.name for e in blue.exposures] == ["pepsib.20250101.001.dxt.nor"]


def test_read_order_by_index_name_or_exposure(
    make_neid: Callable[..., Path], tmp_path: Path
) -> None:
    root = _night(make_neid, tmp_path)
    obs = Observation.from_directory(root, instrument="NEID")
    a = obs.read_order(0, 0)
    b = obs.read_order("T073005", 0)
    c = obs.read_order(obs.exposures[0], 0)
    assert a.flux.size == b.flux.size == c.flux.size == 38
    assert obs.read_order_raw("T073005", 0).status == "raw_only"
    with pytest.raises(DataError, match="matches 4 exposures"):
        obs.read_order("neidL2", 0)


def test_directory_errors(make_neid: Callable[..., Path], tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        Observation.from_directory(tmp_path / "missing")
    with pytest.raises(DataError, match="no file matches any registered"):
        Observation.from_directory(tmp_path)
    make_neid(INSTRUME="SOMETHING")
    with pytest.raises(DataError, match="cannot identify the instrument"):
        Observation.from_directory(tmp_path)
    with pytest.raises(DataError, match="no WIYN/NEID"):
        Observation.from_directory(tmp_path, instrument="NEID", include=["zzz"])
    with pytest.raises(RegistryError):
        Observation.from_directory(tmp_path, instrument="HARPS")
