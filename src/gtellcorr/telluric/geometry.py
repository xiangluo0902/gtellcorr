"""Line-of-sight geometry of one exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gtellcorr.data.observation.observation import Exposure

__all__ = ["Geometry"]


@dataclass(frozen=True)
class Geometry:
    """Observatory and airmass of an exposure, fixed while fitting.

    The airmass is held fixed because it is degenerate with the molecular columns.
    """

    observatory: tuple[float, float, float]
    """``(lat_deg, lon_deg, alt_km)``."""
    airmass: float
    """Airmass of the exposure (the optical-depth cache uses at least 1.001)."""

    @classmethod
    def from_exposure(cls, exposure: Exposure) -> Geometry:
        """Site and airmass of *exposure* (its resolved site: configured, header or fallback)."""
        site = exposure.site
        return cls(
            observatory=(site.lat_deg, site.lon_deg, site.alt_m / 1000.0),
            airmass=float(exposure.airmass),
        )
