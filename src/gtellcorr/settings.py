"""Machine-local environment: where resources, caches and credentials live.

Scientific parameters never go here (they belong in ``FitConfig``); only facts
about *this machine*. Each field is resolved with the priority

    explicit argument  >  environment variable  >  user settings file  >  default

The user settings file is ``~/.config/gtellcorr/settings.toml`` (override its
location with ``GTELLCORR_SETTINGS``)::

    resource_root = "/data/gtellcorr/resources"
    cache_dir = "/scratch/gtellcorr-cache"
    lblrtm_timeout_s = 900
    lblrtm_workers = 4          # LBLRTM runs at a time; "auto" (default) or >= 1
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import platformdirs

from gtellcorr.errors import ConfigError, did_you_mean

__all__ = ["DEFAULT_SETTINGS_FILE", "ENV_VARS", "Settings"]

APP_NAME = "gtellcorr"
DEFAULT_SETTINGS_FILE = Path("~/.config/gtellcorr/settings.toml")
SETTINGS_FILE_ENV = "GTELLCORR_SETTINGS"

#: Environment variable consulted for each field.
ENV_VARS: Mapping[str, str] = {
    "resource_root": "GTELLCORR_RESOURCES",
    "lblrtm_bundle": "GTELLCORR_LBLRTM_BUNDLE",
    "atmosphere_cache": "GTELLCORR_ATMOSPHERE_CACHE",
    "tau_cache": "GTELLCORR_TAU_CACHE",
    "cache_dir": "GTELLCORR_CACHE",
    "credentials_file": "GTELLCORR_CREDENTIALS",
    "lblrtm_timeout_s": "GTELLCORR_LBLRTM_TIMEOUT",
    "lblrtm_workers": "GTELLCORR_LBLRTM_WORKERS",
    "stellar_linelist": "GTELLCORR_STELLAR_LINELIST",
    "ispec_dir": "GTELLCORR_ISPEC_DIR",
}


def _default_values() -> dict[str, Any]:
    """Defaults of every setting (platform-specific user data and cache folders)."""
    return {
        "resource_root": platformdirs.user_data_path(APP_NAME) / "resources",
        "cache_dir": platformdirs.user_cache_path(APP_NAME),
        "lblrtm_bundle": None,
        "atmosphere_cache": None,
        "tau_cache": None,
        "credentials_file": None,
        "lblrtm_timeout_s": 600.0,
        "lblrtm_workers": None,
        "stellar_linelist": None,
        "ispec_dir": None,
    }


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Resolved machine-local settings (immutable)."""

    resource_root: Path
    """Root of external resources: LBLRTM bundle, toolchain, reference files."""
    cache_dir: Path
    """Optical-depth and stellar-template caches."""
    lblrtm_bundle: Path | None = None
    """LBLRTM resource bundle; ``None`` = ``<resource_root>/lblrtm/gnu_double``."""
    atmosphere_cache: Path | None = None
    """Cache of retrieved and stitched atmospheric profiles; ``None`` =
    ``<cache_dir>/atmos_profiles``."""
    tau_cache: Path | None = None
    """Cache of per-order LBLRTM optical depths; ``None`` = ``<cache_dir>/lblrtm_tau_cache``.
    Entries are keyed by the complete physical input, so one cache can safely be
    shared between runs."""
    credentials_file: Path | None = None
    """File holding atmosphere-archive API credentials, if any."""
    lblrtm_timeout_s: float = 600.0
    """Wall-clock limit for one LBLRTM run, in seconds."""
    lblrtm_workers: int | None = None
    """LBLRTM runs of an optical-depth cache made at the same time; ``None`` (or
    ``"auto"``) = the CPUs this process may use, shared between the processes that
    build caches at once, at most 8
    (:func:`gtellcorr.telluric.lblrtm.workers.auto_lblrtm_workers`). Each run
    needs about 1 GB in TMPDIR while it runs. The caches do not depend on it."""
    stellar_linelist: Path | None = None
    """VALD line list for stellar-template synthesis."""
    ispec_dir: Path | None = None
    """iSpec installation (needed only for the iSpec synthesis backend)."""

    def __post_init__(self) -> None:
        """Resolve the derived defaults and turn every path into an absolute ``Path``."""
        object.__setattr__(self, "resource_root", _as_path(self.resource_root))
        object.__setattr__(self, "cache_dir", _as_path(self.cache_dir))
        if self.atmosphere_cache is None:
            object.__setattr__(self, "atmosphere_cache", self.cache_dir / "atmos_profiles")
        if self.tau_cache is None:
            object.__setattr__(self, "tau_cache", self.cache_dir / "lblrtm_tau_cache")
        if self.lblrtm_bundle is None:
            object.__setattr__(self, "lblrtm_bundle", self.resource_root / "lblrtm" / "gnu_double")
        for name in (
            "lblrtm_bundle",
            "atmosphere_cache",
            "tau_cache",
            "credentials_file",
            "stellar_linelist",
            "ispec_dir",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _as_path(value))
        timeout = float(self.lblrtm_timeout_s)
        if not timeout > 0:
            raise ConfigError(f"lblrtm_timeout_s must be positive, got {self.lblrtm_timeout_s!r}")
        object.__setattr__(self, "lblrtm_timeout_s", timeout)
        object.__setattr__(self, "lblrtm_workers", _parse_workers(self.lblrtm_workers))

    @classmethod
    def load(
        cls,
        *,
        settings_file: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> Settings:
        """Resolve settings from arguments, environment, settings file and defaults.

        Parameters
        ----------
        settings_file
            Explicit settings file. Defaults to ``$GTELLCORR_SETTINGS`` or
            ``~/.config/gtellcorr/settings.toml``; a missing default file is fine,
            a missing explicit file is an error.
        environ
            Environment mapping to read (defaults to ``os.environ``; tests pass
            their own).
        **overrides
            Field values that take precedence over everything else. ``None``
            means "not given".
        """
        env = os.environ if environ is None else environ
        valid = {f.name for f in fields(cls)}
        _reject_unknown(overrides, valid, source="Settings.load()")

        values = _default_values()
        values.update(_read_settings_file(settings_file, env, valid))
        for name, var in ENV_VARS.items():
            if env.get(var):
                values[name] = env[var]
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def _as_path(value: str | os.PathLike[str]) -> Path:
    """Absolute path with ``~`` expanded."""
    return Path(value).expanduser().absolute()


def _parse_workers(value: Any) -> int | None:
    """``lblrtm_workers`` as given in an argument, the environment or the settings
    file: ``None`` or ``"auto"``, or an integer >= 1."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f'lblrtm_workers must be "auto" or an integer >= 1, got {value!r}')
    return value


def _reject_unknown(given: Mapping[str, Any], valid: set[str], *, source: str) -> None:
    """Raise for a misspelt setting, with a "did you mean" hint."""
    for key in given:
        if key not in valid:
            raise ConfigError(
                f"unknown setting {key!r} in {source}.{did_you_mean(key, valid)} "
                f"Valid settings: {', '.join(sorted(valid))}."
            )


def _read_settings_file(
    explicit: str | os.PathLike[str] | None,
    env: Mapping[str, str],
    valid: set[str],
) -> dict[str, Any]:
    """Values from the explicit settings file, else from the default one if it exists."""
    if explicit is not None:
        path = _as_path(explicit)
        if not path.is_file():
            raise ConfigError(f"settings file not found: {path}")
    else:
        path = _as_path(env.get(SETTINGS_FILE_ENV) or DEFAULT_SETTINGS_FILE)
        if not path.is_file():
            return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"cannot parse settings file {path}: {exc}") from exc
    _reject_unknown(data, valid, source=str(path))
    return data
