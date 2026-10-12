"""Local secrets loader for atmosphere retriever backends."""

from __future__ import annotations

import ast
import os

from gtellcorr.data.atmosphere import _context


def _secrets_path(path=None):
    """Resolve the local secrets file path."""
    return path or _context.secrets_env()


def load_weather_keys(backend=None, path=None):
    """Read weather-service credentials from ``WEATHER_KEYS``.

    The credentials file (``Settings.credentials_file``, environment variable
    ``GTELLCORR_CREDENTIALS``) is the only credential source for the archive
    backends.  It must contain one assignment::

        WEATHER_KEYS={'era5_cdsapi': {...}, 'merra2_gesdisc': {...}}

    Values are returned as strings and are never printed by this helper.
    """
    fn = _secrets_path(path)
    if not os.path.exists(fn):
        raise FileNotFoundError(f"weather secrets file not found: {fn}")

    raw = None
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == "WEATHER_KEYS":
                raw = val.strip()
                break
    if raw is None:
        raise KeyError(f"WEATHER_KEYS is missing from {fn}")

    try:
        data = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"WEATHER_KEYS in {fn} is not a valid Python literal dict") from e
    if not isinstance(data, dict):
        raise TypeError(f"WEATHER_KEYS in {fn} must be a dict")

    if backend is None:
        return data
    keys = data.get(backend)
    if not isinstance(keys, dict):
        raise KeyError(f"WEATHER_KEYS has no credential dict for backend {backend!r}")
    return {str(k): str(v) for k, v in keys.items()}
