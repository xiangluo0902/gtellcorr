"""Backend: ESO-archived NOAA-ARL GDAS (.gdas) local mirror.

Reads the molecfit-distributed ESO GDAS archive (3-hourly, 0.5 deg)
mirrored under ``<resource_root>/atmosphere/eso_gdas/``.  The nearest archived
site to the requested (lat, lon) is selected, the requested time is
rounded to the nearest 3-hourly cycle, and the matching ``.gdas`` file
is read into an :class:`AtmosphereProfile`.
"""

from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np

from gtellcorr.data.atmosphere import _context

from ..cycles import _nearest_3h_cycle
from ..profile import (
    AtmosphereProfile,
    ProfileFetchError,
    _now_iso,
    _populate_derived,
    angular_sep_deg,
    rh_to_vmr_ppmv,
)

logger = logging.getLogger(__name__)

# ESO GDAS archive: site dirs named C{lon}{lat}, files C{...}D{date}T{hh}.gdas.
_ESO_TAG_RE = re.compile(r"^C([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)$")


def nearest_eso_site(target_lat, target_lon, target_name=""):
    """Find the ESO-GDAS archive site dir nearest to (target_lat, target_lon)
    and log how well it matches the configured observatory.

    Returns (site_dir, tag, eso_lat, eso_lon, sep_deg).
    """
    cand = []
    for d in sorted(glob.glob(os.path.join(_context.eso_gdas_dir(), "C*"))):
        m = _ESO_TAG_RE.match(os.path.basename(d))
        if os.path.isdir(d) and m:
            eso_lon, eso_lat = float(m.group(1)), float(m.group(2))
            sep = angular_sep_deg(target_lat, target_lon, eso_lat, eso_lon)
            cand.append((sep, d, os.path.basename(d), eso_lat, eso_lon))
    if not cand:
        raise FileNotFoundError(
            f"No ESO-GDAS archive site dirs (C*) under {_context.eso_gdas_dir()}"
        )
    cand.sort()
    sep, site_dir, tag, eso_lat, eso_lon = cand[0]
    sep_km = np.radians(sep) * 6371.0
    who = target_name or f"(lat={target_lat:+.3f}, lon={target_lon:+.3f})"
    if sep <= 1.5:
        # ESO GDAS grid spacing is ~1 deg; within that == same grid cell.
        logger.info(
            "ESO-GDAS site match: %s (lat=%+.3f, lon=%+.3f) <-> "
            "archive %s (lat=%+.3f, lon=%+.3f); separation %.3f deg "
            "(%.0f km), within the ESO GDAS grid resolution",
            who,
            target_lat,
            target_lon,
            tag,
            eso_lat,
            eso_lon,
            sep,
            sep_km,
        )
    else:
        logger.warning(
            "ESO-GDAS site MISMATCH: %s (lat=%+.3f, lon=%+.3f); "
            "nearest archive is %s (lat=%+.3f, lon=%+.3f), "
            "separation %.2f deg (%.0f km) -- the comparison is "
            "only valid at that ESO grid point",
            who,
            target_lat,
            target_lon,
            tag,
            eso_lat,
            eso_lon,
            sep,
            sep_km,
        )
    return site_dir, tag, eso_lat, eso_lon, sep


def fetch_eso_gdas(lat, lon, dt_utc, *, obs_alt_km=float("nan"), **_ignored) -> AtmosphereProfile:
    """Read the local ESO-GDAS ``.gdas`` profile nearest to (lat, lon) at the
    3-hourly cycle nearest ``dt_utc``.

    The ESO archive holds molecfit's distributed NOAA-ARL GDAS 0.5 deg
    3-hourly mirror.  The nearest archived site is resolved via
    :func:`nearest_eso_site`; ``dt_utc`` is rounded to the nearest 3-hourly
    cycle; the ``.gdas`` file (columns ``P, z_m, T, RH``) for that site and
    cycle is read into an :class:`AtmosphereProfile`.

    Raises
    ------
    ProfileFetchError : if the matching ``.gdas`` file does not exist.
    """
    site_dir, tag, eso_lat, eso_lon, _sep = nearest_eso_site(lat, lon)
    cycle = _nearest_3h_cycle(dt_utc)
    fn = os.path.join(site_dir, f"{tag}D{cycle:%Y-%m-%d}T{cycle:%H}.gdas")
    if not os.path.exists(fn):
        raise ProfileFetchError(f"ESO-GDAS file not found for cycle {cycle.isoformat()}: {fn}")
    arr = np.loadtxt(fn, comments="#")
    P, z_km, T, RH = arr[:, 0], arr[:, 1] / 1000.0, arr[:, 2], arr[:, 3]
    o = np.argsort(z_km)
    P, z_km, T, RH = P[o], z_km[o], T[o], RH[o]
    h2o = rh_to_vmr_ppmv(RH, T, P)
    prof = AtmosphereProfile(
        backend="eso_gdas",
        fetched_utc=_now_iso(),
        source_url=fn,
        obs_lat_deg=lat,
        obs_lon_deg=lon,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=dt_utc.isoformat(),
        grid_lat_deg=eso_lat,
        grid_lon_deg=eso_lon,
        grid_datetime_utc=cycle.isoformat(),
        height_km=z_km,
        pressure_hpa=P,
        temperature_K=T,
        h2o_vmr_ppmv=h2o,
        relative_humidity_pct=RH,
        geopotential_height_km=z_km.copy(),
        notes="ESO molecfit-distributed NOAA-ARL GDAS 0.5deg 3-hourly mirror",
    )
    _populate_derived(prof)
    return prof
