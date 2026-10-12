"""Backend: NCEP-NCAR Reanalysis 1 (PSL THREDDS, archival)."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

import numpy as np

from ..cycles import _nearest_6h_cycle
from ..profile import (
    AtmosphereProfile,
    ProfileFetchError,
    _now_iso,
    _populate_derived,
    rh_to_vmr_ppmv,
)

logger = logging.getLogger(__name__)


# ==================================================================
# Backend: NCEP-NCAR Reanalysis 1 (PSL THREDDS, archival)
# ==================================================================
# PSL keeps NCEP R1 1948 → present.  Three separate annual files per
# variable; we request each on the chosen 6-hourly cycle then join on
# pressure level.
#   T(z) : air.YYYY.nc   → variable 'air'   (K),     17 pressure levels
#   Z(z) : hgt.YYYY.nc   → variable 'hgt'   (m),     17 levels
#   RH(z): rhum.YYYY.nc  → variable 'rhum'  (%),     8  levels (≥ 300 hPa)
PSL_NCSS_BASE = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/pressure"


def fetch_ncep_r1_psl(
    lat: float,
    lon: float,
    dt_utc: datetime,
    *,
    obs_alt_km: float = float("nan"),
    timeout_s: float = 60.0,
    session=None,
) -> AtmosphereProfile:
    """Fetch a profile from NCEP-NCAR Reanalysis 1 via NOAA PSL THREDDS NCSS.

    Three NCSS CSV requests are made — for temperature, geopotential
    height, and relative humidity — then joined on pressure level.
    Above the RH ceiling (~300 hPa) we set RH=0 (negligible water
    vapor at high altitude, supports a downstream PWV integration that
    behaves sensibly).
    """
    import requests

    if session is None:
        session = requests.Session()

    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    cycle = _nearest_6h_cycle(dt_utc.astimezone(UTC))
    year = cycle.year

    # PSL R1 longitude is given 0..360E or -180..180 (NCSS accepts both,
    # confirmed empirically); we send a -180..180 normalised value.
    lon_q = lon - 360.0 if lon > 180 else lon

    def _get(var):
        """``(csv_text, url)`` of variable *var* at the site from the PSL NCSS server."""
        url = f"{PSL_NCSS_BASE}/{var}.{year}.nc"
        params = {
            "var": var,
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon_q:.4f}",
            "time": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "accept": "csv",
        }
        logger.info("NCEP R1 fetch %s: %s", var, url)
        try:
            r = session.get(url, params=params, timeout=timeout_s)
        except requests.exceptions.RequestException as e:
            raise ProfileFetchError(f"Network error pulling NCEP R1 {var}: {e}") from e
        if r.status_code != 200:
            raise ProfileFetchError(
                f"PSL NCSS returned HTTP {r.status_code} for {url}\nBody head: {r.text[:400]}"
            )
        if not r.text.strip() or "<html" in r.text[:200].lower():
            raise ProfileFetchError(f"PSL NCSS returned empty / HTML body for {url}.")
        return r.text, r.url

    air_txt, air_url = _get("air")
    hgt_txt, _ = _get("hgt")
    rh_txt, _ = _get("rhum")

    df_T = _psl_csv_to_df(air_txt, "air")  # cols: P_hpa, value
    df_Z = _psl_csv_to_df(hgt_txt, "hgt")
    df_RH = _psl_csv_to_df(rh_txt, "rhum")

    # Join on pressure level (descending P = ascending altitude).
    df = df_T.merge(df_Z, on="P_hpa", suffixes=("_T", "_Z"), how="outer")
    df = df.merge(df_RH, on="P_hpa", how="outer")
    df = df.rename(columns={"value_T": "T_K", "value_Z": "Z_m", "value": "RH_pct"})
    df = df.sort_values("P_hpa", ascending=False).reset_index(drop=True)

    # Fill missing levels (rhum stops at 300 hPa) with NaN-aware
    # interpolation in pressure; above 300 hPa set RH = 0 explicitly.
    df["RH_pct"] = df["RH_pct"].astype(float)
    df.loc[df["P_hpa"] < 300.0, "RH_pct"] = df.loc[df["P_hpa"] < 300.0, "RH_pct"].fillna(0.0)
    # T or Z genuinely missing → drop (shouldn't happen)
    df = df.dropna(subset=["T_K", "Z_m"]).reset_index(drop=True)
    # Capture the lat/lon the grid actually resolved to (in any non-empty
    # input table they're identical; pull them from one of the raw texts).
    grid_lat, grid_lon = _parse_psl_gridpoint(air_txt)

    P_hpa = df["P_hpa"].to_numpy(dtype=float)
    T_K = df["T_K"].to_numpy(dtype=float)
    z_km = df["Z_m"].to_numpy(dtype=float) / 1000.0
    RH = df["RH_pct"].to_numpy(dtype=float)
    h2o_ppmv = rh_to_vmr_ppmv(RH, T_K, P_hpa)

    # NCEP R1 levels are ordered low altitude (1000 hPa) → high (10 hPa).
    # AtmosphereProfile wants bottom-up, increasing height — sort accordingly.
    order = np.argsort(z_km)
    prof = AtmosphereProfile(
        backend="ncep_r1_psl",
        fetched_utc=_now_iso(),
        source_url=air_url,
        obs_lat_deg=lat,
        obs_lon_deg=lon,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=dt_utc.astimezone(UTC).isoformat(),
        grid_lat_deg=grid_lat,
        grid_lon_deg=grid_lon,
        grid_datetime_utc=cycle.astimezone(UTC).isoformat(),
        height_km=z_km[order],
        pressure_hpa=P_hpa[order],
        temperature_K=T_K[order],
        h2o_vmr_ppmv=h2o_ppmv[order],
        relative_humidity_pct=RH[order],
        geopotential_height_km=z_km[order],
        notes="NCEP/NCAR Reanalysis 1, 2.5° grid, 17 P-levels (8 with RH); rhum=0 above 300 hPa.",
    )
    _populate_derived(prof)
    return prof


def _psl_csv_to_df(text, varname):
    """Parse a PSL NCSS CSV (text/plain) into a tiny DataFrame with
    columns ['P_hpa', 'value']."""
    import pandas as pd

    df = pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    # Pressure level column is usually labelled 'alt[unit="millibar"]'.
    pcol = None
    for c in df.columns:
        cl = c.lower()
        if "alt" in cl or "isobar" in cl or "pressure" in cl or "level" in cl:
            pcol = c
            break
    if pcol is None:
        raise ValueError(f"No pressure level column in CSV: {list(df.columns)}")
    # Value column = anything starting with varname.
    vcol = None
    for c in df.columns:
        if c.lower().startswith(varname.lower()):
            vcol = c
            break
    if vcol is None:
        raise ValueError(f"No '{varname}' value column in CSV: {list(df.columns)}")
    out = pd.DataFrame({"P_hpa": df[pcol].astype(float), "value": df[vcol].astype(float)})
    return out.dropna(subset=["P_hpa"])


def _parse_psl_gridpoint(text):
    """Extract grid lat/lon from a PSL NCSS CSV. The station column
    looks like ``GridPointRequestedAt[31.960N_111.600W]`` and an
    explicit numeric column is also present."""
    import pandas as pd

    df = pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    glat = float("nan")
    glon = float("nan")
    for c in df.columns:
        if c.lower().startswith("latitude"):
            glat = float(df[c].iloc[0])
        if c.lower().startswith("longitude"):
            glon = float(df[c].iloc[0])
    return glat, glon
