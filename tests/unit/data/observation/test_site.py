from __future__ import annotations

import pytest

from gtellcorr.data.observation.config import SiteCoords
from gtellcorr.data.observation.site import Site, resolve_site, sites
from gtellcorr.errors import DataError, RegistryError


def test_builtin_sites_from_reference_table() -> None:
    assert len(sites) > 250
    kitt = sites.get("kitt peak")
    assert kitt.coords == (31.96333, -111.6, 2120.0)
    assert sites.get("LBT").coords == sites.get("Mt Graham").coords == (32.7016, -109.8719, 2902.0)
    assert sites.get("paranal").alt_m == 2669.0
    with pytest.raises(RegistryError, match=r"unknown site 'Kit Peak'.*'Kitt Peak'"):
        sites.get("Kit Peak")


def test_resolve_site_accepts_names_coordinates_and_sites() -> None:
    assert resolve_site("WIYN").coords == (31.95805, -111.60062, 2091.0)
    assert resolve_site(SiteCoords(lat_deg=1, lon_deg=2, alt_m=3)).coords == (1.0, 2.0, 3.0)
    s = Site(lat_deg=1, lon_deg=2, alt_m=3, name="x")
    assert resolve_site(s) is s


def test_site_validation() -> None:
    with pytest.raises(DataError, match="finite"):
        Site(lat_deg=float("nan"), lon_deg=0, alt_m=0)
    with pytest.raises(DataError, match="latitude"):
        Site(lat_deg=100, lon_deg=0, alt_m=0)
