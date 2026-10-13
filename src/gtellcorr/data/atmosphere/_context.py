"""Per-call context read by the atmosphere modules (internal).

The profile modules read their parameters through module-level names (``atm.*``,
the cache paths, the observatory table, the credentials file). This module
answers them from an explicit :class:`AtmosphereContext` that the public
functions in
:mod:`gtellcorr.data.atmosphere.api` set for the duration of one call
(``contextvars``: thread- and task-safe, nothing survives the call).

Reading a value outside such a call raises :class:`~gtellcorr.errors.ConfigError`
instead of falling back to a hidden default.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtellcorr.data.atmosphere.config import AtmosphereConfig
from gtellcorr.errors import ConfigError

__all__ = [
    "BACKEND_SOURCE",
    "AtmosphereContext",
    "atm",
    "atmos_profile_basename",
    "atmos_profile_dir",
    "current",
    "eso_gdas_dir",
    "full_profile_dir",
    "load_observatory_table",
    "resolve_observatory",
    "secrets_env",
    "using",
]


@dataclass(frozen=True)
class AtmosphereContext:
    """Everything the atmosphere modules may read, for one call."""

    config: AtmosphereConfig
    profile_dir: Path
    """Cache of retrieved (``<source>/``) and stitched (``full_profile/``) profiles."""
    mipas_path: Path
    mipas_sha256: str
    mipas_url: str
    celestrak_path: Path | None
    celestrak_sha256: str | None
    celestrak_url: str
    eso_gdas_dir: Path
    secrets_env: Path | None


_CURRENT: contextvars.ContextVar[AtmosphereContext | None] = contextvars.ContextVar(
    "gtellcorr_atmosphere_context", default=None
)


@contextmanager
def using(context: AtmosphereContext) -> Iterator[AtmosphereContext]:
    """Make *context* current for the atmosphere modules inside the ``with`` block."""
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def current() -> AtmosphereContext:
    """The active context (:class:`ConfigError` outside a public call)."""
    context = _CURRENT.get()
    if context is None:
        raise ConfigError(
            "atmosphere functions need a configuration; call them through "
            "gtellcorr.data.atmosphere (get_meteo / get_full_profile), which pass "
            "an explicit AtmosphereConfig and Settings"
        )
    return context


# ---------------------------------------------------------------------------
# atmosphere parameters
# ---------------------------------------------------------------------------
class _AtmosphereView:
    """The upper-case parameter names the atmosphere modules read (``atm.*``),
    answered from the current context."""

    _MAP: Mapping[str, Any] = {
        "LOWER_BACKEND_DEFAULT": lambda c: c.config.lower_backend,
        "UPPER_SOURCE_DEFAULT": lambda c: c.config.upper_source,
        "BLEND_ZONE_WIDTH_KM": lambda c: c.config.blend_zone_width_km,
        "ASM_EMIX_KM": lambda c: c.config.asm_emix_km,
        "MAX_GRID_SEPARATION_DEG": lambda c: c.config.max_grid_separation_deg,
        "RFM_MIPAS_REFERENCE_PATH": lambda c: str(c.mipas_path),
        "RFM_MIPAS_REFERENCE_SHA256": lambda c: c.mipas_sha256,
        "RFM_MIPAS_REFERENCE_SOURCE_URL": lambda c: c.mipas_url,
        "CELESTRAK_SPACE_WEATHER_PATH": lambda c: _require(c.celestrak_path, "CelesTrak"),
        "CELESTRAK_SPACE_WEATHER_SHA256": lambda c: _require(c.celestrak_sha256, "CelesTrak"),
        "CELESTRAK_SPACE_WEATHER_SOURCE_URL": lambda c: c.celestrak_url,
        "MANUAL_PROFILE_FILE": lambda c: c.config.manual_profile_file,
    }

    def __getattr__(self, name: str) -> Any:
        """The value of *name* in the current context."""
        getter = self._MAP.get(name)
        if getter is None:
            raise AttributeError(f"config.atmosphere.{name} is not provided by gtellcorr")
        return getter(current())


def _require(value: Any, what: str) -> Any:
    """*value*, or :class:`ConfigError` if the table is missing."""
    if value is None:
        raise ConfigError(
            f"no {what} space-weather table is available; run "
            "'gtellcorr resources fetch celestrak' (needed for upper_source = 'msis00')"
        )
    return value


atm = _AtmosphereView()


# ---------------------------------------------------------------------------
# cache paths
# ---------------------------------------------------------------------------
#: Backend id -> source token used for cache sub-directories and file names.
BACKEND_SOURCE: Mapping[str, str] = {
    "era5_cdsapi": "era5",
    "merra2_gesdisc": "merra2",
    "ncep_r1_psl": "gdas",
    "gfs_thredds": "gdas",
    "eso_gdas": "eso",
    "manual": "manual",
}


def atmos_profile_dir(source: str) -> str:
    """Per-source cache directory of retrieved profiles (created on demand)."""
    directory = current().profile_dir / source
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def full_profile_dir() -> str:
    """Cache directory of stitched full profiles (created on demand)."""
    directory = current().profile_dir / "full_profile"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def atmos_profile_basename(source: str, observatory: str, dt: Any, kind: str) -> str:
    """``{source}_{observatory}_{YYYYMMDDTHHMMSS}_{kind}`` (spaces removed)."""
    obs = str(observatory).strip().replace(" ", "")
    return f"{source}_{obs}_{dt:%Y%m%dT%H%M%S}_{kind}"


def eso_gdas_dir() -> str:
    """Directory of the local ESO GDAS profile files."""
    return str(current().eso_gdas_dir)


def secrets_env() -> str:
    """The credentials file (``Settings.credentials_file``); error if unset."""
    path = current().secrets_env
    if path is None:
        raise ConfigError(
            "no credentials file is configured for the atmosphere archives; set "
            "credentials_file in the gtellcorr settings"
        )
    return str(path)


# ---------------------------------------------------------------------------
# observatory
# ---------------------------------------------------------------------------
def load_observatory_table() -> dict[str, tuple[str, float, float, float]]:
    """``{lower-case name: (name, lat_deg, lon_deg, alt_km)}`` from the site registry."""
    from gtellcorr.data.observation.site import sites

    table = {}
    for name in sites.names():
        site = sites.get(name)
        table[name.lower()] = (name, site.lat_deg, site.lon_deg, site.alt_m / 1000.0)
    return table


def resolve_observatory() -> tuple[float, float, float]:
    """Always an error: callers pass explicit site coordinates."""
    raise ConfigError(
        "no observatory given: gtellcorr passes explicit site coordinates to the "
        "atmosphere functions (there is no global site configuration)"
    )


def __getattr__(name: str) -> Any:
    """Module-level names read by the atmosphere modules (``OBSERVATORY_NAME``,
    ``SECRETS_ENV``, ``ESO_GDAS_DIR``)."""
    if name == "OBSERVATORY_NAME":
        return ""  # no global site: the fallback label becomes 'site'
    if name in ("SECRETS_ENV", "ESO_GDAS_DIR"):
        return secrets_env() if name == "SECRETS_ENV" else eso_gdas_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
