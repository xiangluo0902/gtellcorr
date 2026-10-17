"""MERRA-2 backend without the network: the Earthdata Cloud OPeNDAP CSV is parsed and
turned into a profile, interpolated or at the nearest node."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from gtellcorr.data.atmosphere.backends import merra2

LEVELS = merra2.MERRA2_LEVELS_HPA
LATS, LONS = (32.5, 33.0), (-110.0, -109.375)


def _csv() -> str:
    """A CSV response in the cloud format for one time, 42 levels and a 2x2 cell.
    Temperature depends on the node so the selected node can be recognised."""
    lines = [
        "Dataset: MERRA2_400.inst3_3d_asm_Np.20250619.nc4",
        "/time, 540",
        "/lat, " + ", ".join(map(str, LATS)),
        "/lev, " + ", ".join(f"{p:g}" for p in LEVELS),
        "/lon, " + ", ".join(map(str, LONS)),
    ]
    height = 44330.0 * (1.0 - (LEVELS / 1013.25) ** 0.19)
    for name in ("T", "QV", "H"):
        for k in range(LEVELS.size):
            for i in range(2):
                if name == "T":
                    values = [250.0 + 10 * i + j for j in range(2)]
                elif name == "QV":
                    values = [1e-3 * (LEVELS[k] / 1000.0)] * 2
                else:
                    values = [height[k]] * 2
                lines.append(f"/{name}[0][{k}][{i}], " + ", ".join(map(str, values)))
    return "\n".join(lines)


class _Response:
    status_code = 200
    text = _csv()


class _Session:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, timeout: float) -> Any:
        self.urls.append(url)
        return _Response()


def test_parse_cloud_csv() -> None:
    parsed = merra2._parse_opendap_csv(_csv())
    assert parsed["lat"].tolist() == list(LATS) and parsed["lev"].size == 42
    assert parsed["T"].size == parsed["QV"].size == parsed["H"].size == 42 * 4


@pytest.mark.parametrize(("interpolate", "temperature"), [(False, 261.0), (True, 256.64)])
def test_profile_from_the_cloud_response(interpolate: bool, temperature: float) -> None:
    session = _Session()
    profile = merra2.fetch_merra2_gesdisc(
        32.8, -109.6, datetime(2025, 6, 19, 8, 34, tzinfo=UTC),
        session=session, interpolate=interpolate,
    )  # fmt: skip
    (url,) = session.urls
    assert url.startswith(merra2.OPENDAP_GRANULES) and ".dap.csv?dap4.ce=" in url
    assert "MERRA2_400.inst3_3d_asm_Np.20250619.nc4" in url
    # nearest node (33.0, -109.375); bilinear weights 0.6 in latitude, 0.64 in longitude
    assert np.nanmax(profile.temperature_K) == pytest.approx(temperature, abs=1e-6)
    if not interpolate:
        assert (profile.grid_lat_deg, profile.grid_lon_deg) == (33.0, -109.375)
