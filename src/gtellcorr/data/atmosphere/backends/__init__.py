"""Backend dispatch and re-exports for the atmosphere retriever."""

from __future__ import annotations

from ..profile import AtmosphereProfile, check_site_coverage
from .era5 import fetch_era5_cdsapi
from .eso import fetch_eso_gdas
from .gfs import fetch_gfs_thredds
from .manual import load_manual_profile
from .merra2 import fetch_merra2_gesdisc
from .ncep import fetch_ncep_r1_psl


# ==================================================================
# Dispatch
# ==================================================================
def fetch(backend: str, **kwargs) -> AtmosphereProfile:
    """Backend dispatch.

    Examples
    --------
    >>> fetch('ncep_r1_psl', lat=31.96, lon=-111.6,
    ...       dt_utc=datetime(2022, 2, 13, 7, 30, tzinfo=timezone.utc),
    ...       obs_alt_km=2.096)
    >>> fetch('gfs_thredds', lat=..., lon=..., dt_utc=..., obs_alt_km=...)
    >>> fetch('merra2_gesdisc', lat=31.96, lon=-111.6,
    ...       dt_utc=datetime(2022, 2, 13, 7, 30, tzinfo=timezone.utc),
    ...       obs_alt_km=2.096)
    >>> fetch('era5_cdsapi', lat=..., lon=..., dt_utc=..., obs_alt_km=...,
    ...       keep_nc=True, cache_dir='/path/to/cache')
    >>> fetch('manual', path='/path/to/profile.dat',
    ...       obs_lat_deg=31.96, obs_lon_deg=-111.6, obs_alt_km=2.096)
    """
    _backends = {
        "ncep_r1_psl": fetch_ncep_r1_psl,
        "gfs_thredds": fetch_gfs_thredds,
        "merra2_gesdisc": fetch_merra2_gesdisc,
        "era5_cdsapi": fetch_era5_cdsapi,
        "eso_gdas": fetch_eso_gdas,
        "manual": load_manual_profile,
    }
    if backend not in _backends:
        raise ValueError(f"Unknown backend: {backend!r}. Available: {', '.join(_backends)}")
    return check_site_coverage(_backends[backend](**kwargs), backend)
