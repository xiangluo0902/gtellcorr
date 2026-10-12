"""Observatory sites and the site registry.

Built-in sites come from ``sites/observatories.csv``, generated from astropy's
``EarthLocation`` site registry, so they work offline. Lookup is case-insensitive;
third-party packages can add sites through the ``gtellcorr.sites`` entry-point group.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from importlib.resources import files

from gtellcorr.data.observation.config import SiteCoords
from gtellcorr.errors import DataError, RegistryError
from gtellcorr.registry import Registry

__all__ = ["Site", "resolve_site", "sites"]


@dataclass(frozen=True)
class Site:
    """Observatory coordinates."""

    lat_deg: float
    """Latitude [deg, North-positive]."""
    lon_deg: float
    """Longitude [deg, East-positive]."""
    alt_m: float
    """Altitude above mean sea level [m]."""
    name: str | None = None
    """Registry name, if the site comes from the registry."""

    def __post_init__(self) -> None:
        """Check that the coordinates are finite and the latitude is in range."""
        for attr in ("lat_deg", "lon_deg", "alt_m"):
            value = float(getattr(self, attr))
            if not math.isfinite(value):
                raise DataError(f"site {attr} must be finite, got {value!r}")
            object.__setattr__(self, attr, value)
        if not -90 <= self.lat_deg <= 90:
            raise DataError(f"site latitude must be in [-90, 90], got {self.lat_deg}")

    @property
    def coords(self) -> tuple[float, float, float]:
        """``(lat_deg, lon_deg, alt_m)``."""
        return (self.lat_deg, self.lon_deg, self.alt_m)


sites: Registry[Site] = Registry("site", entry_point_group="gtellcorr.sites")


def _load_builtin_sites() -> None:
    """Register every site of ``sites/observatories.csv``."""
    text = (
        files("gtellcorr.data.observation").joinpath("sites/observatories.csv").read_text("utf-8")
    )
    for row in csv.DictReader(io.StringIO(text)):
        name = row["name"].strip()
        site = Site(
            lat_deg=float(row["latitude_deg"]),
            lon_deg=float(row["longitude_deg"]),
            alt_m=float(row["altitude_km"]) * 1000.0,
            name=name,
        )
        if name in sites:
            # astropy lists some sites under several capitalisations; they must agree.
            if sites.get(name).coords != site.coords:
                raise RegistryError(
                    f"built-in site {name!r} is listed twice with different coordinates"
                )
            continue
        sites.register(name, site)


_load_builtin_sites()


def resolve_site(ref: str | SiteCoords | Site) -> Site:
    """A :class:`Site` from a registry name, :class:`SiteCoords` or a :class:`Site`."""
    if isinstance(ref, Site):
        return ref
    if isinstance(ref, SiteCoords):
        return Site(lat_deg=ref.lat_deg, lon_deg=ref.lon_deg, alt_m=ref.alt_m)
    return sites.get(ref)
