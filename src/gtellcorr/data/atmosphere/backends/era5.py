"""Backend: ECMWF ERA5 via Copernicus CDS API."""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime, timedelta

import numpy as np

from gtellcorr.constants import G_EARTH

from ..profile import (
    AtmosphereProfile,
    ProfileFetchError,
    _now_iso,
    _populate_derived,
    _specific_humidity_to_ppmv_rh,
)
from ..secrets import load_weather_keys

logger = logging.getLogger(__name__)


# ==================================================================
# Backend: ECMWF ERA5 via Copernicus CDS API
# ==================================================================
# Product: reanalysis-era5-pressure-levels (37 standard levels,
#          1 → 1000 hPa, hourly, 0.25° × 0.25°, 4D-Var IFS Cy41r2).
# Auth   : WEATHER_KEYS['era5_cdsapi'] with CDSAPI_URL and CDSAPI_KEY in
#          the credentials file (Settings.credentials_file).
# Subset : we request a 0.5° × 0.5° bounding box around the site (the
#          minimum reliably accepted CDS area), then pick the nearest
#          grid point on the returned file.  This keeps the response
#          file at ~5--20 KB regardless of duration.
# Format : netCDF; parsed with ``netCDF4`` (late import).  Variables:
#              t (K), q (specific humidity kg/kg), z (geopotential m²/s²)
#          Geopotential height = z / G_EARTH; q → ppmv via the same
#          molar-mass formula as MERRA-2.
ERA5_DATASET = "reanalysis-era5-pressure-levels"
ERA5_LEVELS_HPA = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


def fetch_era5_cdsapi(
    lat: float,
    lon: float,
    dt_utc: datetime,
    *,
    obs_alt_km: float = float("nan"),
    cache_dir: str | None = None,
    timeout_s: float = 600.0,
    keep_nc: bool = False,
    interpolate: bool = True,
) -> AtmosphereProfile:
    """Fetch a single-point ERA5 profile via the Copernicus CDS API.

    With ``interpolate=True`` (default) the grid cell bracketing the
    station is bilinearly interpolated to the exact (lat, lon); with
    ``interpolate=False`` the nearest grid point is used.  Combine with
    ``keep_nc=True`` to obtain both from a single CDS download.

    The CDS service queues requests on the server side; a typical
    single-hour single-point subset returns in 1–5 min when the queue
    is short, but can take 10–30 min during peak hours.  This call
    blocks until the file is ready.

    Parameters
    ----------
    lat, lon : float
        Site coordinates (deg).  CDS expects lon in (-180, 180].
    dt_utc : datetime
        Observation time in UTC.  ERA5 is hourly; we request the
        closest UTC hour (no interpolation).
    obs_alt_km : float
        Observatory altitude (km); stored in metadata.
    cache_dir : str or None
        Directory to drop the temporary NetCDF into.  Defaults to
        ``$TMPDIR``.  If ``keep_nc=True`` the file is retained at
        ``{cache_dir}/era5_{lat}_{lon}_{YYYYMMDDTHH}.nc`` for reuse.
    timeout_s : float
        Maximum wait time before giving up on the CDS queue.
    keep_nc : bool
        Whether to keep the downloaded NetCDF file after parsing.

    Returns
    -------
    AtmosphereProfile

    Raises
    ------
    ProfileFetchError : on missing credentials, CDS failure, or parse error.

    Notes
    -----
    * Requires ``cdsapi`` (``pip install cdsapi``) and ``netCDF4``
      (``conda install -c conda-forge netcdf4``).  Both are imported
      lazily so the module remains importable without them.
    * Requires the credentials file (``Settings.credentials_file``) to define
      ``WEATHER_KEYS['era5_cdsapi']`` with ``CDSAPI_URL`` and
      ``CDSAPI_KEY``.  Register at https://cds.climate.copernicus.eu/
      and accept the dataset licence once before the first call.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise ProfileFetchError(
            "fetch_era5_cdsapi requires the ``cdsapi`` package.  "
            "Install with ``pip install cdsapi`` and set "
            'WEATHER_KEYS["era5_cdsapi"] in the credentials file (the credentials_file '
            "setting or the GTELLCORR_CREDENTIALS environment variable)."
        ) from e
    try:
        import netCDF4  # noqa: F401  (probe early; real import below)
    except ImportError as e:
        raise ProfileFetchError(
            "fetch_era5_cdsapi requires the ``netCDF4`` package to read "
            "CDS responses.  Install with "
            "``conda install -c conda-forge netcdf4``."
        ) from e

    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    dt_utc = dt_utc.astimezone(UTC)
    # Round to nearest whole UTC hour.
    cycle_dt = dt_utc.replace(minute=0, second=0, microsecond=0)
    if dt_utc.minute >= 30:
        cycle_dt = cycle_dt + timedelta(hours=1)

    # Normalise lon to (-180, 180].
    lon_q = lon - 360.0 if lon > 180.0 else lon
    # Tiny bounding box around the site; CDS rejects single-point area.
    # Keep extents on a 0.25° multiple to land squarely on the ERA5 grid.
    half = 0.25
    north = min(90.0, np.round((lat + half) / 0.25) * 0.25)
    south = max(-90.0, np.round((lat - half) / 0.25) * 0.25)
    east = np.round((lon_q + half) / 0.25) * 0.25
    west = np.round((lon_q - half) / 0.25) * 0.25
    if east > 180.0:
        east -= 360.0
    if west < -180.0:
        west += 360.0

    # Where to drop the temporary NetCDF.
    if cache_dir is None:
        import tempfile

        cache_dir = tempfile.gettempdir()
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"era5_{lat:+.3f}_{lon:+.3f}_{cycle_dt:%Y%m%dT%H}.nc"
    target = os.path.join(cache_dir, fname)

    if not (keep_nc and os.path.exists(target)):
        request = {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["temperature", "specific_humidity", "geopotential"],
            "pressure_level": [str(p) for p in ERA5_LEVELS_HPA],
            "year": f"{cycle_dt.year:04d}",
            "month": f"{cycle_dt.month:02d}",
            "day": f"{cycle_dt.day:02d}",
            "time": f"{cycle_dt.hour:02d}:00",
            "area": [float(north), float(west), float(south), float(east)],  # N, W, S, E
        }
        logger.info("ERA5 CDS request: %s", request)
        keys = load_weather_keys("era5_cdsapi")
        url = keys.get("CDSAPI_URL")
        key = keys.get("CDSAPI_KEY")
        if not url or not key:
            raise ProfileFetchError(
                'WEATHER_KEYS["era5_cdsapi"] must define CDSAPI_URL and CDSAPI_KEY'
            )
        try:
            c = cdsapi.Client(url=url, key=key, timeout=timeout_s, quiet=True)
            c.retrieve(ERA5_DATASET, request, target)
        except ProfileFetchError:
            raise
        except Exception as e:
            raise ProfileFetchError(
                f"CDS request failed for ERA5 {cycle_dt.isoformat()}: {e}"
            ) from e

    # Parse the NetCDF response.
    import netCDF4 as _nc

    try:
        ds = _nc.Dataset(target, mode="r")
    except OSError as e:
        raise ProfileFetchError(f"Failed to open downloaded ERA5 NetCDF {target}: {e}") from e

    try:
        # CDS ERA5 NetCDF conventions:
        #   t : (time, level, lat, lon)   K
        #   q : (time, level, lat, lon)   kg/kg
        #   z : (time, level, lat, lon)   m²/s²  (geopotential)
        # The new (post-2024) CADS schema may rename ``level`` →
        # ``pressure_level`` and ``latitude``/``longitude`` keep their
        # full names; we look them up flexibly.
        def _v(*names):
            """The first of *names* present in the dataset."""
            for n in names:
                if n in ds.variables:
                    return ds.variables[n]
            raise KeyError(f"No variable in {names}; got {list(ds.variables)}")

        T = np.asarray(_v("t", "temperature")[:], dtype=float)
        q = np.asarray(_v("q", "specific_humidity")[:], dtype=float)
        z = np.asarray(_v("z", "geopotential")[:], dtype=float)
        levels = np.asarray(_v("pressure_level", "level", "plev")[:], dtype=float)
        lats = np.asarray(_v("latitude", "lat")[:], dtype=float)
        lons = np.asarray(_v("longitude", "lon")[:], dtype=float)
    finally:
        ds.close()
    if not keep_nc:
        with contextlib.suppress(OSError):
            os.remove(target)

    # Squeeze the singleton time axis if present.
    if T.ndim == 4:
        T = T[0]
        q = q[0]
        z = z[0]  # → (level, lat, lon)

    # Select the station column.  interpolate=True bilinearly interpolates
    # the bracketing grid cell to the exact (lat, lon) -- a refinement
    # over nearest-grid selection (ERA5's 0.25 deg grid leaves a site up
    # to ~14 km from a node; the 0.5 deg bbox always brackets the site).
    if interpolate:

        def _brack(coords, x0):
            """Return (i0, i1, w) so interp = (1-w)*f[i0] + w*f[i1] for a
            monotonic (either order) 1-D coordinate array."""
            c = np.asarray(coords, dtype=float)
            asc = c[0] <= c[-1]
            cc = c if asc else c[::-1]
            x0c = float(np.clip(x0, cc[0], cc[-1]))
            k = int(np.clip(np.searchsorted(cc, x0c), 1, len(cc) - 1))
            denom = cc[k] - cc[k - 1]
            w = 0.0 if denom == 0 else (x0c - cc[k - 1]) / denom
            i0, i1 = k - 1, k
            if not asc:
                n = len(c)
                i0, i1 = n - 1 - i0, n - 1 - i1
            return i0, i1, w

        jlo, jhi, wlat = _brack(lats, lat)
        ilo, ihi, wlon = _brack(lons, lon_q)

        def _bilin(f):  # f: (level, lat, lon) -> (level,)
            """Bilinear interpolation of a (level, lat, lon) field to the site."""
            return (1.0 - wlat) * ((1.0 - wlon) * f[:, jlo, ilo] + wlon * f[:, jlo, ihi]) + wlat * (
                (1.0 - wlon) * f[:, jhi, ilo] + wlon * f[:, jhi, ihi]
            )

        T_K = np.asarray(_bilin(T), dtype=float).ravel()
        q_kgkg = np.asarray(_bilin(q), dtype=float).ravel()
        z_m2s2 = np.asarray(_bilin(z), dtype=float).ravel()
        grid_lat_used, grid_lon_used = float(lat), float(lon)
    else:
        j = int(np.argmin(np.abs(lats - lat)))
        ii = int(np.argmin(np.abs(((lons - lon_q + 180.0) % 360.0) - 180.0)))
        T_K = np.asarray(T[:, j, ii], dtype=float).ravel()
        q_kgkg = np.asarray(q[:, j, ii], dtype=float).ravel()
        z_m2s2 = np.asarray(z[:, j, ii], dtype=float).ravel()
        grid_lat_used, grid_lon_used = float(lats[j]), float(lons[ii])

    # geopotential (m²/s²) → geopotential height (m) → km
    H_km = z_m2s2 / G_EARTH / 1000.0

    # specific humidity → ppmv VMR (+ RH for diagnostic storage).
    h2o_ppmv, RH = _specific_humidity_to_ppmv_rh(q_kgkg, levels, T_K)

    # Sort bottom-up (CDS may return either order).
    order = np.argsort(H_km)
    H_km = H_km[order]
    levels = levels[order]
    T_K = T_K[order]
    h2o_ppmv = h2o_ppmv[order]
    RH = RH[order]

    prof = AtmosphereProfile(
        backend="era5_cdsapi",
        fetched_utc=_now_iso(),
        source_url=target if keep_nc else f"CDS:{ERA5_DATASET}:{cycle_dt.isoformat()}",
        obs_lat_deg=lat,
        obs_lon_deg=lon,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=dt_utc.isoformat(),
        grid_lat_deg=grid_lat_used,
        grid_lon_deg=grid_lon_used,
        grid_datetime_utc=cycle_dt.isoformat(),
        height_km=H_km,
        pressure_hpa=levels,
        temperature_K=T_K,
        h2o_vmr_ppmv=h2o_ppmv,
        relative_humidity_pct=RH,
        geopotential_height_km=H_km.copy(),
        notes=(
            "ERA5 pressure-levels (0.25°×0.25°, hourly, 37 P-levels); "
            + (
                "bilinearly interpolated to the station coordinates"
                if interpolate
                else "nearest grid point"
            )
            + "; specific humidity → ppmv via M_dry/M_H2O."
        ),
    )
    _populate_derived(prof)
    return prof
