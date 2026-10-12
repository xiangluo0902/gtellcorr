"""Backend: NOAA GFS 0.25° analysis via UCAR THREDDS NCSS."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

import numpy as np

from ..cycles import _nearest_gfs_cycle
from ..profile import (
    AtmosphereProfile,
    ProfileFetchError,
    _now_iso,
    _populate_derived,
    rh_to_vmr_ppmv,
)

logger = logging.getLogger(__name__)


# ==================================================================
# Backend: NOAA GFS 0.25° analysis via UCAR THREDDS NCSS
# ==================================================================
# Catalog: https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg_ana
# File pattern:  GFS_Global_0p25deg_ana_YYYYMMDD_HHMM.grib2
# NCSS POINT  :  /thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg_ana/<file>/dataset.html
THREDDS_NCSS_BASE = "https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg_ana"


def _gfs_analysis_filename(cycle_utc: datetime) -> str:
    """File naming for the GFS 0.25° analysis on UCAR THREDDS."""
    return f"GFS_Global_0p25deg_ana_{cycle_utc:%Y%m%d}_{cycle_utc:%H%M}.grib2"


def fetch_gfs_thredds(
    lat: float,
    lon: float,
    dt_utc: datetime,
    *,
    obs_alt_km: float = float("nan"),
    timeout_s: float = 60.0,
    session=None,
) -> AtmosphereProfile:
    """Fetch a GFS 0.25° analysis profile via UCAR THREDDS NCSS (CSV).

    Variables retrieved at isobaric levels:
        - ``Temperature_isobaric`` (K)
        - ``Relative_humidity_isobaric`` (%)
        - ``Geopotential_height_isobaric`` (m)

    Wind speed/direction are *not* retrieved (irrelevant for telluric
    absorption modelling).

    Parameters
    ----------
    lat, lon : float
        Decimal degrees.  ``lon`` may be in (-180, 180] or [0, 360);
        normalised internally.
    dt_utc : datetime
        Observation time in UTC (timezone-aware recommended).
    obs_alt_km : float
        Observatory altitude in km; stored in metadata, not used for the
        fetch itself.
    timeout_s : float
        Per-request timeout in seconds.
    session : requests.Session or None
        If given, use this session; otherwise a fresh one is created.

    Returns
    -------
    AtmosphereProfile

    Raises
    ------
    ProfileFetchError : if the network call fails or the response is empty.
    """
    import requests  # late import keeps the module importable offline

    if session is None:
        session = requests.Session()

    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    cycle = _nearest_gfs_cycle(dt_utc.astimezone(UTC))
    fname = _gfs_analysis_filename(cycle)
    url = f"{THREDDS_NCSS_BASE}/{fname}"

    # NCSS expects lon in (-180, 180] for global grids.
    lon_q = lon - 360.0 if lon > 180 else lon

    # Request all three vars in a single CSV.  ``vertCoord`` omitted →
    # all isobaric levels returned.  ``accept=csv`` puts level (Pa) as a
    # column and one row per (var × level).
    params = {
        "var": ("Temperature_isobaric,Relative_humidity_isobaric,Geopotential_height_isobaric"),
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon_q:.4f}",
        "time": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vertStride": "1",
        "accept": "csv",
    }

    logger.info("GFS analysis fetch: %s  params=%s", url, params)
    try:
        r = session.get(url, params=params, timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        raise ProfileFetchError(f"Network error retrieving GFS analysis: {e}\n  url={url}") from e
    if r.status_code != 200:
        raise ProfileFetchError(
            f"GFS NCSS returned HTTP {r.status_code} for {url}\nResponse head: {r.text[:400]}"
        )
    text = r.text.strip()
    if not text or "<html" in text[:200].lower():
        raise ProfileFetchError(
            f"GFS NCSS returned an HTML / empty body (likely the cycle "
            f"{cycle.isoformat()} is missing from the archive).\n"
            f"Response head: {text[:400]}"
        )

    return _parse_ncss_csv(
        text,
        backend="gfs_thredds",
        url=r.url,
        obs_lat=lat,
        obs_lon=lon,
        obs_alt_km=obs_alt_km,
        obs_dt_utc=dt_utc,
        grid_dt_utc=cycle,
    )


def _parse_ncss_csv(
    text: str,
    *,
    backend: str,
    url: str,
    obs_lat: float,
    obs_lon: float,
    obs_alt_km: float,
    obs_dt_utc: datetime,
    grid_dt_utc: datetime,
) -> AtmosphereProfile:
    """Parse a UCAR THREDDS NCSS CSV response into a AtmosphereProfile.

    The CSV layout used by NCSS for an isobaric-level request looks like:

        time,lat,lon,vertCoord (vertical coord name),Temperature_isobaric,
            Relative_humidity_isobaric,Geopotential_height_isobaric   (one line)
        2022-02-13T06:00:00Z,31.875,-111.625,10000.0,236.9,3.0,16210.5
        2022-02-13T06:00:00Z,31.875,-111.625,15000.0,...
        ...

    Some quirks worth handling defensively:
    - Column ordering can vary slightly with THREDDS version.
    - "vertCoord" header might be ``isobaric`` or ``isobaric1`` etc.
    - Pressure levels are in Pa, ascending (towards lower P / higher altitude).
    """
    import pandas as pd

    df = pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]

    def _find_col(name_substr):
        """The CSV column whose name contains *name_substr* (case-insensitive)."""
        for c in df.columns:
            if name_substr.lower() in c.lower():
                return c
        raise ValueError(
            f"No column containing {name_substr!r} in CSV. Columns = {list(df.columns)}"
        )

    col_T = _find_col("Temperature_isobaric")
    col_RH = _find_col("Relative_humidity_isobaric")
    col_Z = _find_col("Geopotential_height_isobaric")
    col_lat = _find_col("lat")
    col_lon = _find_col("lon")
    # The level column tends to be the first column that ISN'T one of the
    # known ones — usually labelled ``isobaric`` or ``vertCoord (...)``.
    known = {col_T, col_RH, col_Z, col_lat, col_lon}
    candidates = [c for c in df.columns if c not in known and c.lower() not in ("time", "date")]
    if not candidates:
        raise ValueError(f"No pressure-level column in CSV: {list(df.columns)}")
    col_level = candidates[0]

    grid_lat = float(df[col_lat].iloc[0])
    grid_lon = float(df[col_lon].iloc[0])

    P_pa = df[col_level].to_numpy(dtype=float)
    T_K = df[col_T].to_numpy(dtype=float)
    RH = df[col_RH].to_numpy(dtype=float)
    Z_m = df[col_Z].to_numpy(dtype=float)

    # Sort bottom-up (low altitude → high altitude = high P → low P)
    order = np.argsort(-P_pa)
    P_pa = P_pa[order]
    T_K = T_K[order]
    RH = RH[order]
    Z_m = Z_m[order]

    P_hpa = P_pa / 100.0
    z_km = Z_m / 1000.0
    h2o_ppmv = rh_to_vmr_ppmv(RH, T_K, P_hpa)

    prof = AtmosphereProfile(
        backend=backend,
        fetched_utc=_now_iso(),
        source_url=url,
        obs_lat_deg=obs_lat,
        obs_lon_deg=obs_lon,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=obs_dt_utc.astimezone(UTC).isoformat(),
        grid_lat_deg=grid_lat,
        grid_lon_deg=grid_lon,
        grid_datetime_utc=grid_dt_utc.astimezone(UTC).isoformat(),
        height_km=z_km,
        pressure_hpa=P_hpa,
        temperature_K=T_K,
        h2o_vmr_ppmv=h2o_ppmv,
        relative_humidity_pct=RH,
        geopotential_height_km=z_km.copy(),
    )
    _populate_derived(prof)
    return prof
