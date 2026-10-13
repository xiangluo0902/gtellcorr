"""``get_meteo`` — config-aware retrieval of one atmospheric profile.

A thin, stateless orchestrator over ``retriever.fetch`` (+ optional on-site
ASM merge): every input defaults to the project configuration and can be
overridden per call; it returns the ``AtmosphereProfile`` object (the "currency"
shared by ``plot_meteo`` / ``plot_forward_model``) and optionally saves it to
the package data tree.

Designed so a future object-oriented shell (``TelluricModel.get_meteo()``)
is a trivial forwarder: no global state, explicit keyword inputs, returns the
profile object.
"""

from __future__ import annotations

import logging
import os

from gtellcorr.data.atmosphere import retriever as g
from gtellcorr.data.atmosphere._context import (
    BACKEND_SOURCE,
    atm,
    atmos_profile_basename,
    atmos_profile_dir,
    load_observatory_table,
    resolve_observatory,
)

log = logging.getLogger(__name__)


def _resolve_site(observatory, observatory_name, observatory_label):
    """Return (lat, lon, alt_km, label).  Priority: explicit coords >
    a name in the site registry (there is no global site)."""
    if observatory is not None:
        lat, lon, alt = observatory
        return float(lat), float(lon), float(alt), observatory_label or "site"
    if observatory_name:
        hit = load_observatory_table().get(observatory_name.strip().lower())
        if hit is None:
            known = ", ".join(sorted(entry[0] for entry in load_observatory_table().values()))
            raise ValueError(
                f"observatory_name {observatory_name!r} is not in the site registry; "
                f"known sites: {known}"
            )
        _, lat, lon, alt = hit
        return lat, lon, alt, observatory_label or observatory_name.replace(" ", "")
    lat, lon, alt = resolve_observatory()
    from gtellcorr.data.atmosphere._context import OBSERVATORY_NAME

    label = observatory_label or (OBSERVATORY_NAME.replace(" ", "") if OBSERVATORY_NAME else "site")
    return lat, lon, alt, label


def get_meteo(
    *,
    dt_utc,
    observatory=None,
    observatory_name=None,
    observatory_label=None,
    backend=None,
    interpolate=True,
    merge_onsite=False,
    surface_T_K=None,
    surface_RH_pct=None,
    surface_P_hpa=None,
    save=True,
    reuse_cache=True,
    out_dir=None,
    basename=None,
    verbose=False,
):
    """Retrieve one atmospheric profile for ``dt_utc`` at the configured site.

    Parameters (all keyword-only; ``None`` → resolved from config)
    ----------
    dt_utc : datetime (required, tz-aware UTC)
        Observation time.
    observatory : (lat_deg, lon_deg, alt_km) | None
        Explicit coordinates.  ``None`` → ``observatory_name``.
    observatory_name : str | None
        A name in the site registry (:data:`gtellcorr.data.observation.site.sites`;
        used if ``observatory`` is None).
    observatory_label : str | None
        Short tag for the output filename (e.g. ``'LBT'``).
    backend : str | None
        Retriever backend; ``None`` → ``config.atmosphere.LOWER_BACKEND_DEFAULT``.
        ``manual`` reads ``atmosphere.manual_profile_file`` on every call and never
        caches it.
    interpolate : bool
        Bilinearly interpolate to the site (ERA5 / MERRA-2 only).
    merge_onsite : bool
        Blend on-site surface T/RH/P into the lowest layers (needs the
        ``surface_*`` values).
    save : bool
        Persist the profile via ``save_profile`` (default True).
    out_dir, basename : str | None
        Override the output directory / filename stem (defaults:
        ``<atmosphere_cache>/<source>/`` and the standard ``..._raw`` stem).
    verbose : bool
        Log whether the lower profile was retrieved, reused or read at INFO
        (otherwise at DEBUG).

    Returns
    -------
    AtmosphereProfile
    """
    backend = backend or atm.LOWER_BACKEND_DEFAULT
    lat, lon, alt, label = _resolve_site(observatory, observatory_name, observatory_label)
    src = BACKEND_SOURCE.get(backend, backend)
    out_dir = out_dir or atmos_profile_dir(src)

    # Key the raw cache by the DATA CYCLE the backend snaps to (hourly ERA5,
    # 3-hourly MERRA-2, 6-hourly NCEP/GFS), not the exposure time: this is
    # the data's true resolution, so exposures sharing a cycle reuse one
    # download.  fetch() still receives the original dt_utc and rounds itself.
    cycle_dt = g.backend_cycle(backend, dt_utc)
    base = basename or atmos_profile_basename(src, label, cycle_dt, "raw")

    # The cached artifact is the raw (un-merged) retrieved profile, keyed by
    # source+site+cycle.  Reuse it if present; otherwise fetch (and save).
    if backend == "manual":
        # A user-supplied file is read every time and never cached: a cached copy
        # could outlive changes to the file.
        prof = g.load_manual_profile(
            os.path.expanduser(str(atm.MANUAL_PROFILE_FILE)),
            obs_lat_deg=lat,
            obs_lon_deg=lon,
            obs_alt_km=alt,
            obs_datetime_utc=dt_utc.isoformat(),
        )
        profile_status = "read"
    else:
        prof = g.load_profile(out_dir, base) if reuse_cache else None
        # A cache written before grid coordinates were recorded has non-finite
        # grid lat/lon, which would silently bypass the coverage guard below;
        # treat it as a miss and re-fetch so the guard runs on real coordinates.
        if prof is not None and not g.has_grid_coords(prof):
            prof = None
    if prof is None:
        kw = dict(lat=lat, lon=lon, dt_utc=dt_utc, obs_alt_km=alt)
        if backend == "era5_cdsapi":
            # keep the NetCDF too, so re-runs reuse it (no re-download).
            kw.update(interpolate=interpolate, keep_nc=True, cache_dir=out_dir)
        elif backend == "merra2_gesdisc":
            kw["interpolate"] = interpolate
        prof = g.fetch(backend, **kw)
        if save:
            g.save_profile(prof, out_dir, base)
        profile_status = "retrieved"
    elif backend != "manual":
        profile_status = "reused"

    log.log(
        logging.INFO if verbose else logging.DEBUG,
        "%s lower profile: %s",
        src.upper(),
        profile_status,
    )

    # Coverage guard on the returned profile (also catches a stale cached
    # wrong-location profile that bypassed fetch()'s own check).
    g.check_site_coverage(prof, backend)

    # On-site ASM merge is applied in memory on the raw profile (the cached
    # file stays the raw download).
    if merge_onsite and any(v is not None for v in (surface_T_K, surface_RH_pct, surface_P_hpa)):
        prof = g.merge_site_measurements(
            prof,
            surface_T_K=surface_T_K,
            surface_RH_pct=surface_RH_pct,
            surface_P_hpa=surface_P_hpa,
        )

    return prof
