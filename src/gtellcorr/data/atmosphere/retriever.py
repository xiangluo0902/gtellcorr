"""Atmospheric profile retriever for ground-based telluric fitting.

Purpose
-------
Forward-modelling telluric correction needs an *actual* T(z), P(z), VMR(z)
profile above the observatory to predict telluric line shapes correctly.  A
single climatological profile is fine to first order but biases line cores
and wings on the few-percent level when the real atmosphere deviates
(Smette+2015).  This module pulls a time-and-location specific profile
from public meteorological archives and converts it into gtellcorr's
``AtmosphereProfile``, which :mod:`gtellcorr.data.atmosphere.full_profile`
stitches with the upper atmosphere.

Design
------
- **Modular**: backends are isolated functions returning a uniform
  ``AtmosphereProfile`` dataclass.  ERA5, MERRA-2, NCEP-NCAR R1, GFS and ESO
  GDAS backends ship; adding another is a one-function addition.
- **Lightweight**: the network backend uses only ``requests`` and the
  ``Pandas / NumPy`` parsing layer — no ``xarray`` / ``netCDF4`` / ``cfgrib``
  / ``eccodes`` dependency.  This is on purpose so it runs inside the
  project runtime without extra installs.
- **Reproducible**: every fetch records where the data came from (backend name,
  URL, grid point actually used, timestamp of retrieval, ECMWF/NCEP cycle
  time) inside the saved JSON.
- **Offline-friendly**: :func:`load_manual_profile` reads a user-supplied ASCII
  table (``lower_backend = "manual"`` with ``atmosphere.manual_profile_file``).

Backends shipped
----------------
- ``'ncep_r1_psl'``    : NCEP-NCAR Reanalysis 1 via NOAA PSL THREDDS NCSS
  (HTTPS, no auth, CSV).  Coverage: **1948 onward**, 6-hourly cycles
  (00, 06, 12, 18 UTC), 2.5° × 2.5° grid, 17 pressure levels for T/Z
  and 8 levels for RH.  Useful baseline because of its archive depth.
- ``'gfs_thredds'``    : NOAA GFS 0.25° analysis (= GDAS analysis) via
  UCAR THREDDS NCSS.  Higher resolution but only the last ~2 weeks
  online — use only for very recent exposures.
- ``'merra2_gesdisc'`` : NASA MERRA-2 ``M2I3NPASM`` (3-hourly,
  instantaneous, pressure-level, assimilation, meteorology) via the
  GES DISC OPeNDAP ``.ascii`` endpoint.  0.5° × 0.625° grid, 42
  pressure levels, 3-hourly cycles (00, 03, ..., 21 UTC).  Coverage
  1980 → present (~3-week latency).  Same data product PSG uses for
  its telluric backend (Villanueva+18).  Requires NASA Earthdata
  credentials in the credentials file (``Settings.credentials_file``)
  ``WEATHER_KEYS['merra2_gesdisc']['EARTHDATA_TOKEN']``.
- ``'era5_cdsapi'``    : ECMWF ERA5 ``reanalysis-era5-pressure-levels``
  via the Copernicus CDS API.  0.25° × 0.25° grid, 37 pressure levels,
  **hourly** cadence, 4D-Var assimilation — widely considered the
  state-of-the-art reanalysis.  Coverage 1940 → present (5-day latency
  for ERA5T).  Requires the ``cdsapi`` package, a CDS account, and
  the credentials file's ``WEATHER_KEYS['era5_cdsapi']`` with
  ``CDSAPI_URL`` and ``CDSAPI_KEY``.  Late-imports ``netCDF4`` for parsing
  the response.
- ``'eso_gdas'``      : ESO GDAS profiles from local files
  (``<resource_root>/atmosphere/eso_gdas``).
- ``'manual'``         : Local ASCII file with columns
  ``height_km, pressure_hpa, temperature_K, h2o_vmr_ppmv``.

Outputs
-------
An ``AtmosphereProfile`` dataclass (the lower-atmosphere profile) carrying
numpy arrays and metadata; serialized to a ``#``-header + column ``.dat`` via
``save_profile`` / ``load_profile``.

Scope
-----
This package ONLY retrieves the lower-atmosphere meteorology, config-driven.
Stitching the 0–120 km full profile lives in
:mod:`gtellcorr.data.atmosphere.full_profile`; the public entry points are in
:mod:`gtellcorr.data.atmosphere.api`.

Usage
-----
>>> from gtellcorr.data.atmosphere.retriever import fetch
>>> prof = fetch('era5_cdsapi', lat=lat, lon=lon, obs_alt_km=alt_km,
...              dt_utc=dt_utc)          # any backend, any site

Backend- and instrument-agnostic: the site, the instrument and the atmosphere
settings are passed in explicitly.
"""

from __future__ import annotations

from .backends import (
    fetch,
    fetch_era5_cdsapi,
    fetch_eso_gdas,
    fetch_gfs_thredds,
    fetch_merra2_gesdisc,
    fetch_ncep_r1_psl,
    load_manual_profile,
)
from .cycles import backend_cycle
from .merge import merge_site_measurements
from .profile import (
    AtmosphereProfile,
    ProfileFetchError,
    angular_sep_deg,
    check_site_coverage,
    compute_pwv_mm,
    has_grid_coords,
    load_profile,
    rh_to_vmr_ppmv,
    saturation_vapor_pressure_pa,
    save_profile,
)

__all__ = [
    "AtmosphereProfile",
    "ProfileFetchError",
    "angular_sep_deg",
    "backend_cycle",
    "check_site_coverage",
    "compute_pwv_mm",
    "fetch",
    "fetch_era5_cdsapi",
    "fetch_eso_gdas",
    "fetch_gfs_thredds",
    "fetch_merra2_gesdisc",
    "fetch_ncep_r1_psl",
    "has_grid_coords",
    "load_manual_profile",
    "load_profile",
    "merge_site_measurements",
    "rh_to_vmr_ppmv",
    "saturation_vapor_pressure_pa",
    "save_profile",
]
