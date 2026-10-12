from __future__ import annotations

import pytest

from gtellcorr.data.atmosphere.config import AtmosphereConfig
from gtellcorr.data.observation.config import DataConfig, Frames, SiteCoords
from gtellcorr.data.observation.target import Star, Transit
from gtellcorr.data.products.config import Postprocessing
from gtellcorr.errors import ConfigError


def test_data_config_site_accepts_name_or_coordinates() -> None:
    by_name = DataConfig.from_dict({"directory": "/d", "fallback_site": "Kitt Peak"})
    assert by_name.fallback_site == "Kitt Peak"
    coords = DataConfig.from_dict(
        {"directory": "/d", "site": {"lat_deg": 31.96, "lon_deg": -111.6, "alt_m": 2096}}
    )
    assert coords.site == SiteCoords(lat_deg=31.96, lon_deg=-111.6, alt_m=2096.0)


def test_transit_is_all_or_nothing() -> None:
    assert not Transit().enabled
    assert Transit(t0_bjd=1.0, period_d=2.0, duration_d=0.1).enabled
    with pytest.raises(ConfigError, match=r"^t0_bjd: t0_bjd, period_d and duration_d"):
        Transit(t0_bjd=1.0)


@pytest.mark.parametrize(
    ("group", "data", "message"),
    [
        (DataConfig, {"directory": " "}, r"^directory: must not be empty"),
        (DataConfig, {"directory": "/d", "include": ["a"], "include_file": "f"}, r"^include:"),
        (DataConfig, {"directory": "/d", "site": 5}, r"^site: expected"),
        (SiteCoords, {"lat_deg": 91, "lon_deg": 0, "alt_m": 0}, r"^lat_deg: must be in"),
        (SiteCoords, {"lat_deg": 0, "lon_deg": 400, "alt_m": 0}, r"^lon_deg: must be in"),
        (Star, {"name": ""}, r"^name: must not be empty"),
        (Star, {"name": "s", "teff": -1}, r"^teff: must be > 0"),
        (Star, {"name": "s", "vsini_kms": -1}, r"^vsini_kms: must be >= 0"),
        (Star, {"name": "s", "limb_darkening_coeff": 2}, r"^limb_darkening_coeff"),
        (Transit, {"t0_bjd": 1, "period_d": -2, "duration_d": 1}, r"^period_d: must be > 0"),
        (Transit, {"t0_bjd": 1, "period_d": 2, "duration_d": 0}, r"^duration_d: must be > 0"),
        (AtmosphereConfig, {"lower_backend": "era6"}, r"^lower_backend: must be one of"),
        (AtmosphereConfig, {"blend_zone_width_km": 0}, r"^blend_zone_width_km: must be > 0"),
        (Frames, {"vacuum_air_formula": "edlen"}, r"^vacuum_air_formula: must be one of"),
        (Postprocessing, {"telluric_pca_components": 0}, r"^telluric_pca_components"),
        (
            DataConfig,
            {"directory": "/d", "instrument": "HARPS"},
            r"^instrument: unknown instrument",
        ),
        (
            DataConfig,
            {"directory": "/d", "instrument": "NEID", "channel": "red"},
            r"^channel: 'red'",
        ),
        (DataConfig, {"directory": "/d", "site": "Kit Peak"}, r"^site: unknown site.*'Kitt Peak'"),
        (
            DataConfig,
            {"directory": "/d", "fallback_site": "Nowhere"},
            r"^fallback_site: unknown site",
        ),
    ],
)
def test_validation(group: type, data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        group.from_dict(data)
