"""Data selection (the ``[data]`` table) and wavelength-frame parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gtellcorr.config.base import ConfigGroup
from gtellcorr.data.observation.target import Star, Transit

__all__ = ["DataConfig", "Frames", "SiteCoords"]


@dataclass(frozen=True, kw_only=True)
class SiteCoords(ConfigGroup):
    """Explicit observatory coordinates."""

    lat_deg: float
    """Latitude [deg, North-positive]."""
    lon_deg: float
    """East-positive longitude [deg]."""
    alt_m: float
    """Altitude above mean sea level [m]."""

    def validate(self) -> None:
        """Latitude in [-90, 90], longitude in [-180, 360]."""
        if not -90 <= self.lat_deg <= 90:
            self.fail("lat_deg", f"must be in [-90, 90], got {self.lat_deg}")
        if not -180 <= self.lon_deg <= 360:
            self.fail("lon_deg", f"must be in [-180, 360], got {self.lon_deg}")


@dataclass(frozen=True, kw_only=True)
class DataConfig(ConfigGroup):
    """Which observations to process (the ``[data]`` table of a run file)."""

    directory: str
    """Directory holding the reduced spectra."""
    instrument: str | None = None
    """Registered instrument name, e.g. ``"WIYN/NEID"``; leave out to identify the
    instrument from the file headers."""
    channel: str | None = None
    """Channel (fibre / arm) to process; leave out for the instrument's default channel."""
    include: tuple[str, ...] | None = None
    """Keep only exposures whose file name contains one of these tokens; leave out to
    keep all."""
    include_file: str | None = None
    """File with one include token per line (``#`` comments allowed)."""
    exclude: tuple[str, ...] = ()
    """Drop exposures whose file name contains one of these tokens (applied everywhere)."""
    site: str | SiteCoords | None = None
    """Observatory override (registry name or coordinates); beats header values."""
    fallback_site: str | SiteCoords | None = None
    """Site used only for coordinates missing from the headers (registry name or
    coordinates); defaults to the instrument's site."""
    target: Star | None = None
    """The observed star: a name, and the parameters a synthesised template needs."""
    transit: Transit | None = None
    """Transit ephemeris; in-transit exposures are left out of the stellar master."""

    def validate(self) -> None:
        """Known instrument, channel and sites; include or include_file, not both."""
        # Registries are imported here: they depend on this module's SiteCoords.
        from gtellcorr.data.observation.instrument import instruments
        from gtellcorr.data.observation.site import sites
        from gtellcorr.errors import RegistryError

        if not self.directory.strip():
            self.fail("directory", "must not be empty")
        if self.include is not None and self.include_file is not None:
            self.fail("include", "give include or include_file, not both")
        if self.instrument is not None:
            try:
                spec = instruments.get(self.instrument)
            except RegistryError as exc:
                self.fail("instrument", str(exc))
            if self.channel is not None and self.channel not in spec.channels:
                self.fail(
                    "channel",
                    f"{self.channel!r} is not a channel of {spec.name}; "
                    f"channels are {', '.join(spec.channels)}",
                )
        for name in ("site", "fallback_site"):
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    sites.get(value)
                except RegistryError as exc:
                    self.fail(name, str(exc))


@dataclass(frozen=True, kw_only=True)
class Frames(ConfigGroup):
    """Wavelength-frame conventions."""

    vacuum_air_formula: Literal["espresso", "edlen1966", "ciddor1996"] = "espresso"
    """Vacuum <-> air conversion: ``espresso`` (Edlen 1953 + Birch & Downs 1994, as the
    ESPRESSO pipeline), ``edlen1966`` (as molecfit), ``ciddor1996`` (recommended for NIR)."""
