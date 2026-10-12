from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gtellcorr.data.atmosphere import (
    AtmosphereConfig,
    _context,
    api,
    atmosphere_context,
    full_profile_for_exposure,
    get_full_profile,
    get_meteo,
)
from gtellcorr.data.atmosphere.reference import ReferenceFile, latest_celestrak, record_celestrak
from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.header import OnSiteWeather
from gtellcorr.data.observation.observation import Exposure
from gtellcorr.data.observation.site import Site
from gtellcorr.errors import ConfigError, ResourceError
from gtellcorr.settings import Settings

T0 = datetime(2022, 2, 13, 7, 35, 5, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(resource_root=tmp_path / "res", cache_dir=tmp_path / "cache")


def _celestrak(tmp_path: Path) -> ReferenceFile:
    return ReferenceFile(tmp_path / "sw.csv", "c" * 64)


def test_context_is_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="need a configuration"):
        _ = _context.atm.BLEND_ZONE_WIDTH_KM
    with pytest.raises(ConfigError, match="need a configuration"):
        _context.full_profile_dir()


def test_context_maps_legacy_names(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cfg = AtmosphereConfig(lower_backend="merra2_gesdisc", blend_zone_width_km=4.0, asm_emix_km=6.0)
    ctx = atmosphere_context(cfg, s, celestrak=_celestrak(tmp_path))
    with _context.using(ctx):
        atm = _context.atm
        assert atm.LOWER_BACKEND_DEFAULT == "merra2_gesdisc"
        assert (atm.BLEND_ZONE_WIDTH_KM, atm.ASM_EMIX_KM) == (4.0, 6.0)
        assert atm.MAX_GRID_SEPARATION_DEG == 3.0
        assert atm.RFM_MIPAS_REFERENCE_PATH.endswith("rfm_mipas_midlatitude_night_v3_20010103.atm")
        assert atm.CELESTRAK_SPACE_WEATHER_SHA256 == "c" * 64
        assert (
            Path(_context.full_profile_dir())
            == tmp_path / "cache" / "atmos_profiles" / "full_profile"
        )
        assert Path(_context.atmos_profile_dir("era5")).is_dir()
        assert _context.eso_gdas_dir() == str(tmp_path / "res" / "atmosphere" / "eso_gdas")
        assert _context.OBSERVATORY_NAME == ""
        assert "kitt peak" in _context.load_observatory_table()
        with pytest.raises(AttributeError, match="not provided"):
            _ = atm.MODEL_DEFAULTS
        with pytest.raises(ConfigError, match="no credentials file"):
            _context.secrets_env()
        with pytest.raises(ConfigError, match="explicit site coordinates"):
            _context.resolve_observatory()
    with pytest.raises(ConfigError):
        _context.current()  # nothing survives the block


def test_context_is_per_thread(tmp_path: Path) -> None:
    ctx = atmosphere_context(
        AtmosphereConfig(), _settings(tmp_path), celestrak=_celestrak(tmp_path)
    )
    seen: list[BaseException] = []

    def other_thread() -> None:
        try:
            _context.current()
        except ConfigError as exc:
            seen.append(exc)

    with _context.using(ctx):
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
    assert len(seen) == 1


def test_celestrak_needed_only_for_msis(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    with pytest.raises(ResourceError, match="no CelesTrak snapshot recorded"):
        atmosphere_context(AtmosphereConfig(), s)
    ctx = atmosphere_context(AtmosphereConfig(upper_source="mipas"), s)
    with _context.using(ctx), pytest.raises(ConfigError, match="CelesTrak space-weather"):
        _ = _context.atm.CELESTRAK_SPACE_WEATHER_PATH
    snapshot = s.resource_root / "atmosphere" / "celestrak_space_weather_sw_all_20260923.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("DATE,BSRN\n")
    recorded = record_celestrak(snapshot)
    assert latest_celestrak(snapshot.parent) == recorded
    assert atmosphere_context(AtmosphereConfig(), s).celestrak_path == snapshot
    with pytest.raises(ResourceError, match="not found"):
        record_celestrak(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("<html>")
    with pytest.raises(ResourceError, match="not a CelesTrak"):
        record_celestrak(bad)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from gtellcorr.data.atmosphere import full_profile, meteo

    calls: dict[str, Any] = {}

    def fake_full(dt: datetime, **kwargs: Any) -> str:
        calls.update(kwargs, dt=dt, context=_context.current())
        return "FULL"

    def fake_meteo(**kwargs: Any) -> str:
        calls.update(kwargs, context=_context.current())
        return "LOWER"

    def fake_base(path: str, **kwargs: Any) -> dict[str, str]:
        calls.update(kwargs, path=path)
        return {"base": "yes"}

    monkeypatch.setattr(full_profile, "get_full_profile", fake_full)
    monkeypatch.setattr(full_profile, "extract_base_atmosphere", fake_base)
    monkeypatch.setattr(meteo, "get_meteo", fake_meteo)
    return calls


def test_get_full_profile_arguments(tmp_path: Path, captured: dict[str, Any]) -> None:
    site = Site(lat_deg=31.958092, lon_deg=-111.600562, alt_m=2091.0)
    onsite = OnSiteWeather(T_K=286.55, RH_pct=3.2, P_hpa=None, dewpoint_K=247.6, time=None)
    out = get_full_profile(
        T0, site, settings=_settings(tmp_path), site_label="KPNO", onsite=onsite,
        celestrak=_celestrak(tmp_path), save=False, reuse_cache=False, return_meta=True,
    )  # fmt: skip
    assert out == "FULL"
    assert captured["dt"] == T0
    assert captured["observatory"] == (31.958092, -111.600562, 2.091)  # km, as the reference driver
    assert captured["observatory_label"] == "KPNO"
    assert (captured["backend"], captured["upper_source"], captured["merge_onsite"]) == (
        "era5_cdsapi",
        "msis00",
        False,
    )
    assert (captured["surface_T_K"], captured["surface_RH_pct"], captured["surface_P_hpa"]) == (
        286.55,
        3.2,
        None,
    )
    assert (captured["save"], captured["reuse_cache"], captured["return_meta"]) == (
        False,
        False,
        True,
    )
    assert isinstance(captured["context"], _context.AtmosphereContext)


def test_site_by_name_and_no_onsite(tmp_path: Path, captured: dict[str, Any]) -> None:
    get_full_profile(T0, "Kitt Peak", settings=_settings(tmp_path), celestrak=_celestrak(tmp_path))
    assert captured["observatory"] == (31.96333, -111.6, 2.12)
    assert captured["surface_T_K"] is None and captured["observatory_label"] is None


def test_full_profile_for_exposure_uses_the_reference_call(
    tmp_path: Path, captured: dict[str, Any]
) -> None:
    spec = instruments.get("NEID")
    exposure = Exposure(
        path=tmp_path / "neidL2_x.fits",
        time_utc=T0,
        site=Site(lat_deg=31.9, lon_deg=-111.6, alt_m=2091.0),
        site_sources={},
        airmass=1.9,
        airmass_source="fits_header",
        weather=OnSiteWeather(T_K=280.0, RH_pct=10.0, P_hpa=None, dewpoint_K=None, time=None),
        resolution=110000.0,
        object="x",
    )
    cfg = AtmosphereConfig(merge_onsite=True, upper_source="mipas")
    full_profile_for_exposure(exposure, spec, config=cfg, settings=_settings(tmp_path))
    assert captured["observatory_label"] == "KPNO"  # instrument site token
    assert captured["merge_onsite"] is True and captured["upper_source"] == "mipas"
    assert captured["surface_T_K"] == 280.0


def test_get_meteo_and_reference_atmosphere(tmp_path: Path, captured: dict[str, Any]) -> None:
    cfg = AtmosphereConfig()  # msis00, and no CelesTrak snapshot: not needed for the lower part
    assert (
        get_meteo(T0, "LBT", config=cfg, settings=_settings(tmp_path), site_label="LBT") == "LOWER"
    )
    assert captured["context"].celestrak_path is None
    assert captured["dt_utc"] == T0 and captured["backend"] == "era5_cdsapi"
    base = api.reference_atmosphere(None, _settings(tmp_path))  # msis00: no CelesTrak needed
    assert base == {"base": "yes"}
    assert captured["expected_sha256"].startswith("539da944")
    assert captured["path"].endswith("rfm_mipas_midlatitude_night_v3_20010103.atm")


def test_settings_atmosphere_cache(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    assert s.atmosphere_cache == tmp_path / "cache" / "atmos_profiles"
    custom = Settings(resource_root=tmp_path, cache_dir=tmp_path, atmosphere_cache=tmp_path / "old")
    assert custom.atmosphere_cache == tmp_path / "old"


def test_manual_profile_file_config() -> None:
    with pytest.raises(ConfigError, match="manual_profile_file: is required"):
        AtmosphereConfig(lower_backend="manual")
    with pytest.raises(ConfigError, match="applies only to lower_backend"):
        AtmosphereConfig(manual_profile_file="profile.dat")
    cfg = AtmosphereConfig(lower_backend="manual", manual_profile_file="profile.dat")
    assert AtmosphereConfig.from_dict(cfg.to_dict()) == cfg
    assert "manual_profile_file" not in AtmosphereConfig().to_dict()  # None is not written


def test_manual_profile_file_must_exist(tmp_path: Path) -> None:
    cfg = AtmosphereConfig(
        lower_backend="manual", manual_profile_file=str(tmp_path / "missing.dat"),
        upper_source="mipas",
    )  # fmt: skip
    with pytest.raises(ConfigError, match="manual_profile_file not found"):
        atmosphere_context(cfg, _settings(tmp_path))


def test_manual_profile_is_read_every_time_and_not_cached(tmp_path: Path) -> None:
    from gtellcorr.data.atmosphere import meteo

    profile = tmp_path / "profile.dat"

    def write(t0: float) -> None:
        profile.write_text(
            "# height_km pressure_hPa temperature_K h2o_vmr_ppmv\n"
            f"5.0 550.0 {t0 - 30.0} 900.0\n2.0 790.0 {t0} 5000.0\n10.0 260.0 {t0 - 65.0} 50.0\n"
        )

    cfg = AtmosphereConfig(
        lower_backend="manual", manual_profile_file=str(profile), upper_source="mipas"
    )
    write(280.0)
    ctx = atmosphere_context(cfg, _settings(tmp_path))
    with _context.using(ctx):
        first = meteo.get_meteo(dt_utc=T0, observatory=(31.7, -109.9, 3.2), backend="manual")
        write(285.0)
        second = meteo.get_meteo(dt_utc=T0, observatory=(31.7, -109.9, 3.2), backend="manual")
    assert list(first.height_km) == [2.0, 5.0, 10.0]  # sorted bottom-up
    assert first.temperature_K[0] == 280.0 and second.temperature_K[0] == 285.0
    assert (first.backend, first.obs_lat_deg, first.grid_lat_deg) == ("manual", 31.7, 31.7)
    cache = tmp_path / "cache" / "atmos_profiles" / "manual"
    assert not cache.exists() or not any(cache.iterdir())


def test_missing_cdsapi_points_to_the_credentials_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from gtellcorr.data.atmosphere.backends.era5 import fetch_era5_cdsapi
    from gtellcorr.data.atmosphere.profile import ProfileFetchError

    monkeypatch.setitem(sys.modules, "cdsapi", None)  # import cdsapi -> ImportError
    with pytest.raises(ProfileFetchError, match="GTELLCORR_CREDENTIALS") as error:
        fetch_era5_cdsapi(31.7, -109.9, T0)
    assert "secrets.env" not in str(error.value)


def test_progress_is_logged_not_printed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    import logging

    from gtellcorr.data.atmosphere import full_profile, meteo

    profile = tmp_path / "profile.dat"
    profile.write_text("2.0 790.0 280.0 5000.0\n5.0 550.0 250.0 900.0\n")
    cfg = AtmosphereConfig(lower_backend="manual", manual_profile_file=str(profile))
    ctx = atmosphere_context(cfg, _settings(tmp_path), need_celestrak=False)
    with _context.using(ctx), caplog.at_level(logging.DEBUG, logger="gtellcorr.data.atmosphere"):
        meteo.get_meteo(dt_utc=T0, observatory=(31.7, -109.9, 3.2), backend="manual", verbose=True)
        meteo.get_meteo(dt_utc=T0, observatory=(31.7, -109.9, 3.2), backend="manual")
        full_profile._report(True, "full profile: constructed")
        full_profile._report(False, "full profile: reused")
    records = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, "MANUAL lower profile: read") in records
    assert (logging.DEBUG, "MANUAL lower profile: read") in records
    assert (logging.INFO, "full profile: constructed") in records
    assert (logging.DEBUG, "full profile: reused") in records
    assert not any("[1." in message for _, message in records)
    assert capsys.readouterr().out == ""


def test_unknown_site_name_names_the_registry(tmp_path: Path) -> None:
    from gtellcorr.data.atmosphere import meteo

    ctx = atmosphere_context(AtmosphereConfig(upper_source="mipas"), _settings(tmp_path))
    with _context.using(ctx), pytest.raises(ValueError, match="not in the site registry") as e:
        meteo._resolve_site(None, "Nowhere Peak", None)
    assert "known sites:" in str(e.value) and "observatories.csv" not in str(e.value)
