"""Backend: manual ASCII file."""

from __future__ import annotations

import os

import numpy as np

from ..profile import AtmosphereProfile, _now_iso, _populate_derived


# ==================================================================
# Backend: manual ASCII file
# ==================================================================
def load_manual_profile(
    path: str,
    *,
    obs_lat_deg: float = float("nan"),
    obs_lon_deg: float = float("nan"),
    obs_alt_km: float = float("nan"),
    obs_datetime_utc: str = "",
) -> AtmosphereProfile:
    """Read a 4-column whitespace-separated ASCII profile.

    Expected format (lines starting with ``#`` are comments)::

        # height_km   pressure_hPa   temperature_K   h2o_vmr_ppmv
        0.0          1013.25         288.15          8000.0
        1.0           898.7          281.6           5500.0
        2.0           794.9          275.1           3400.0
        ...

    Supplementary metadata (lat, lon, alt, obs time) is taken from the
    ``obs_*`` kwargs; if absent it stays NaN/empty.
    """
    arr = np.loadtxt(path, comments=("#", ";"))
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(
            f"Expected ≥4 columns (z_km, P_hpa, T_K, h2o_ppmv); got "
            f"{arr.shape[1]} columns in {path}"
        )
    z, P, T, h2o = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    order = np.argsort(z)  # bottom-up
    prof = AtmosphereProfile(
        backend="manual",
        fetched_utc=_now_iso(),
        source_url=os.path.abspath(path),
        obs_lat_deg=obs_lat_deg,
        obs_lon_deg=obs_lon_deg,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=obs_datetime_utc,
        grid_lat_deg=obs_lat_deg,
        grid_lon_deg=obs_lon_deg,
        grid_datetime_utc=obs_datetime_utc,
        height_km=z[order],
        pressure_hpa=P[order],
        temperature_K=T[order],
        h2o_vmr_ppmv=h2o[order],
        notes="ASCII manual profile",
    )
    _populate_derived(prof)
    return prof
