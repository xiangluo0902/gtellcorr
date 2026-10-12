"""Atmospheric profiles: reanalysis backends, on-site weather merge, the 0-120 km
full profile and its reference files.

Public API::

    from gtellcorr.data import atmosphere
    full = atmosphere.get_full_profile(time_utc, site, config=AtmosphereConfig(), settings=...)
    full = atmosphere.full_profile_for_exposure(exposure, instrument, config=..., settings=...)
"""

from __future__ import annotations

from gtellcorr.data.atmosphere.api import (
    atmosphere_context,
    full_profile_for_exposure,
    get_full_profile,
    get_meteo,
    reference_atmosphere,
)
from gtellcorr.data.atmosphere.config import AtmosphereConfig

__all__ = [
    "AtmosphereConfig",
    "atmosphere_context",
    "full_profile_for_exposure",
    "get_full_profile",
    "get_meteo",
    "reference_atmosphere",
]
