"""FITS-header parsing driven by an instrument's header schema.

Each instrument declares where its site coordinates, epoch, airmass and on-site
weather live (:class:`HeaderSchema`); the readers here apply those rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from gtellcorr.config.base import ConfigGroup

__all__ = [
    "HeaderField",
    "HeaderSchema",
    "OnSiteWeather",
    "SiteTime",
    "parse_sexagesimal",
    "read_on_site_weather",
    "read_scalar",
    "read_site_time",
    "read_time",
]

FieldFormat = Literal[
    "deg", "dms", "hms_hours", "m", "degC", "pct", "hpa", "s", "float", "iso", "date+time"
]


@dataclass(frozen=True, kw_only=True)
class HeaderField(ConfigGroup):
    """Where one quantity lives in the primary header."""

    keys: tuple[str, ...] = ()
    """Cards tried in order; the first present, finite value wins. Empty = not provided."""
    fmt: FieldFormat | None = None
    """``deg`` / ``m`` / ``degC`` / ``pct`` / ``hpa`` / ``s`` / ``float``: plain number;
    ``dms``: sexagesimal degrees; ``hms_hours``: sexagesimal hours (East-positive, x15);
    ``iso``: one ISO-8601 card; ``date+time``: ``keys = [date_card, time_card]``."""
    jd_keys: tuple[str, ...] = ()
    """Julian-date (or MJD) cards used when the time cards cannot be parsed. They must
    give the same epoch as the time cards (start or mid-exposure), in UTC."""

    def validate(self) -> None:
        """A format is needed for declared keys; ``date+time`` needs two keys."""
        if self.keys and self.fmt is None:
            self.fail("fmt", "required when keys are given")
        if self.fmt == "date+time" and len(self.keys) != 2:
            self.fail("keys", "fmt 'date+time' needs exactly [date_card, time_card]")


def _absent() -> HeaderField:
    """A field the header does not provide."""
    return HeaderField()


@dataclass(frozen=True, kw_only=True)
class HeaderSchema(ConfigGroup):
    """Header cards of one instrument: where each quantity is found."""

    lat: HeaderField = field(default_factory=_absent)
    """Site latitude [deg, North-positive]."""
    lon: HeaderField = field(default_factory=_absent)
    """Site longitude [deg, East-positive]."""
    alt_m: HeaderField = field(default_factory=_absent)
    """Site altitude [m]."""
    time: HeaderField = field(default_factory=_absent)
    """Exposure epoch (start)."""
    exptime: HeaderField = field(default_factory=_absent)
    """Exposure time [s], for the mid-exposure epoch."""
    bjd: HeaderField = field(default_factory=_absent)
    """Barycentric Julian Date (BJD_TDB) of mid-exposure, as a full JD (``fmt =
    "float"``); written to the RV table. Without it, the JD (UTC) of the exposure
    epoch is used."""
    airmass: HeaderField = field(default_factory=_absent)
    """Airmass of the exposure."""
    asm_T: HeaderField = field(default_factory=_absent)
    """On-site air temperature [degC]."""
    asm_RH: HeaderField = field(default_factory=_absent)
    """On-site relative humidity [%]."""
    asm_P: HeaderField = field(default_factory=_absent)
    """On-site pressure [hPa]."""
    asm_dew: HeaderField = field(default_factory=_absent)
    """On-site dew point [degC]."""
    asm_time: HeaderField = field(default_factory=_absent)
    """Time stamp of the on-site measurement."""


def parse_sexagesimal(value: Any) -> float:
    """``'+32:42:04.71'`` / ``'-07:19:33.38'`` -> signed decimal (``D:M:S``, ``D:M`` or ``D``)."""
    s = str(value).strip()
    sign = -1.0 if s.startswith("-") else 1.0
    parts = s.lstrip("+-").split(":")
    d = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
    m = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    sec = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    return sign * (d + m / 60.0 + sec / 3600.0)


def _get(header: Any, key: str) -> Any:
    """Card value, or ``None`` for a missing card or header."""
    return header.get(key) if hasattr(header, "get") else None


def read_scalar(header: Any, spec: HeaderField) -> float | None:
    """Read one numeric field; a finite float in the field's unit, or ``None``."""
    if header is None or not spec.keys:
        return None
    for key in spec.keys:
        value = _get(header, key)
        if value is None or value == "":
            continue
        try:
            if spec.fmt == "dms":
                number = parse_sexagesimal(value)
            elif spec.fmt == "hms_hours":
                number = parse_sexagesimal(value) * 15.0
            else:
                number = float(value)
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            continue
    return None


def _jd_to_datetime(jd: float) -> datetime:
    """Julian Date -> UTC datetime, accepting plain JD or MJD."""
    mjd = jd - 2400000.5 if jd > 2400000.5 else jd
    return datetime(1858, 11, 17, tzinfo=UTC) + timedelta(days=float(mjd))


def _first_finite(header: Any, keys: tuple[str, ...]) -> float | None:
    """First card among *keys* holding a finite number."""
    for key in keys:
        if header is None:
            continue
        try:
            number = float(_get(header, key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def read_time(header: Any, spec: HeaderField) -> datetime | None:
    """Exposure epoch as a timezone-aware UTC datetime (``None`` if absent)."""
    if header is None:
        return None
    dt_utc: datetime | None = None
    if spec.fmt == "iso":
        for key in spec.keys:
            value = _get(header, key)
            if isinstance(value, str) and value:
                try:
                    dt_utc = datetime.fromisoformat(value.strip().rstrip("Z")).replace(tzinfo=UTC)
                    break
                except ValueError:
                    continue
    elif spec.fmt == "date+time":
        date_value = _get(header, spec.keys[0])
        time_value = _get(header, spec.keys[1])
        if isinstance(date_value, str) and date_value:
            text = date_value.strip()
            if isinstance(time_value, str) and time_value:
                text = f"{text}T{time_value.strip()}"
            try:
                dt_utc = datetime.fromisoformat(text.rstrip("Z")).replace(tzinfo=UTC)
            except ValueError:
                dt_utc = None
    if dt_utc is None:
        jd = _first_finite(header, spec.jd_keys)
        dt_utc = _jd_to_datetime(jd) if jd is not None else None
    return dt_utc


@dataclass(frozen=True)
class SiteTime:
    """Geometry and epoch of one exposure, as read from its header."""

    lat_deg: float
    """Latitude [deg]."""
    lon_deg: float
    """East-positive longitude [deg]."""
    alt_m: float
    """Altitude [m]."""
    time_utc: datetime | None
    """Mid-exposure epoch if the instrument declares ``mid_exposure``, else the start."""
    airmass: float | None
    """Header airmass, ``None`` if absent."""
    sources: Mapping[str, str]
    """Where lat/lon/alt came from: ``"header"`` or ``"fallback"``."""


def read_site_time(
    header: Any,
    schema: HeaderSchema,
    *,
    mid_exposure: bool,
    fallback: tuple[float, float, float] | None,
) -> SiteTime:
    """Site coordinates, epoch and airmass of an exposure.

    Coordinates missing from the header are taken **field by field** from
    *fallback* ``(lat_deg, lon_deg, alt_m)``. If a field is
    missing and no fallback is given, a :class:`~gtellcorr.errors.DataError` is raised.
    """
    from gtellcorr.errors import DataError

    values = {
        "lat_deg": read_scalar(header, schema.lat),
        "lon_deg": read_scalar(header, schema.lon),
        "alt_m": read_scalar(header, schema.alt_m),
    }
    sources = {}
    for i, name in enumerate(("lat_deg", "lon_deg", "alt_m")):
        if values[name] is not None:
            sources[name] = "header"
            continue
        if fallback is None:
            raise DataError(f"header has no {name} and no fallback site is configured")
        values[name] = fallback[i]
        sources[name] = "fallback"
    dt_utc = read_time(header, schema.time)
    if dt_utc is not None and mid_exposure:
        exptime_s = read_scalar(header, schema.exptime)
        if exptime_s:
            dt_utc = dt_utc + timedelta(seconds=exptime_s / 2.0)
    lat, lon, alt = values["lat_deg"], values["lon_deg"], values["alt_m"]
    assert lat is not None and lon is not None and alt is not None
    return SiteTime(
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt,
        time_utc=dt_utc,
        airmass=read_scalar(header, schema.airmass),
        sources=sources,
    )


@dataclass(frozen=True)
class OnSiteWeather:
    """On-site meteorology from the header (``None`` where not provided)."""

    T_K: float | None
    """Air temperature [K]."""
    RH_pct: float | None
    """Relative humidity [%]."""
    P_hpa: float | None
    """Pressure [hPa]."""
    dewpoint_K: float | None
    """Dew point [K]."""
    time: Any
    """Time stamp of the measurement as written in the header."""


def read_on_site_weather(header: Any, schema: HeaderSchema) -> OnSiteWeather:
    """On-site T / RH / P / dew point (temperatures converted degC -> K)."""
    if header is None:
        return OnSiteWeather(None, None, None, None, None)
    t_c = read_scalar(header, schema.asm_T)
    dew_c = read_scalar(header, schema.asm_dew)
    measured = None
    for key in schema.asm_time.keys:
        value = _get(header, key)
        if value:
            measured = value
            break
    return OnSiteWeather(
        T_K=(t_c + 273.15) if t_c is not None else None,
        RH_pct=read_scalar(header, schema.asm_RH),
        P_hpa=read_scalar(header, schema.asm_P),
        dewpoint_K=(dew_c + 273.15) if dew_c is not None else None,
        time=measured,
    )
