"""Stitching a lower-atmosphere profile that starts well above 0 km (MERRA-2 at a
mountain site: lowest valid level 850 hPa, about 1.5 km) onto the 0--120 km grid."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gtellcorr.data.atmosphere import AtmosphereConfig, _context, atmosphere_context, full_profile
from gtellcorr.data.atmosphere.profile import AtmosphereProfile
from gtellcorr.settings import Settings

GRID = full_profile.CANONICAL_ALTITUDE_GRID_KM
SCALE_HEIGHT_KM = 8.0


@pytest.fixture(autouse=True)
def context(tmp_path: Path) -> Iterator[None]:
    settings = Settings(resource_root=tmp_path / "res", cache_dir=tmp_path / "cache")
    with _context.using(atmosphere_context(AtmosphereConfig(upper_source="mipas"), settings)):
        yield


def _base() -> dict:
    """A reference atmosphere on the project grid (isothermal, exponential pressure)."""
    return {
        "z_km": GRID.copy(),
        "T_K": np.full(GRID.shape, 250.0),
        "P_hPa": 1013.25 * np.exp(-GRID / SCALE_HEIGHT_KM),
        "H2O_ppmv": np.full(GRID.shape, 5.0),
        "reference_provenance": {"source_sha256": "0" * 64},
    }


def _lower(z_bottom_km: float) -> AtmosphereProfile:
    z = np.arange(z_bottom_km, 20.0, 0.5)
    return AtmosphereProfile(
        backend="test",
        fetched_utc="2025-06-19T00:00:00Z",
        obs_lat_deg=32.7,
        obs_lon_deg=-109.9,
        obs_alt_km=3.2,
        obs_datetime_utc="2025-06-19T08:34:12Z",
        height_km=z,
        pressure_hpa=1000.0 * np.exp(-z / 7.5),
        temperature_K=290.0 - 6.5 * z,
        h2o_vmr_ppmv=8000.0 * np.exp(-z / 2.0),
    )


@pytest.mark.parametrize("z_bottom_km", [0.19, 1.5])
def test_pressure_below_the_profile_is_extrapolated(z_bottom_km: float) -> None:
    lower = _lower(z_bottom_km)
    full, _ = full_profile._build_full_profile_resolved(lower, upper_source="mipas", base=_base())
    assert np.all(np.diff(full["P_hPa"]) < 0.0)
    z = full["z_km"]
    below = z < z_bottom_km
    assert below.any()
    # the lower profile is exactly exponential, so the extrapolation reproduces it
    np.testing.assert_allclose(full["P_hPa"][below], 1000.0 * np.exp(-z[below] / 7.5))
    # T and H2O keep the value of the lowest level
    assert np.all(full["T_K"][below] == lower.temperature_K[0])
    assert np.all(full["H2O_ppmv"][below] == lower.h2o_vmr_ppmv[0])
    # at and above the lowest level nothing changes
    above = ~below & (z < 10.0)
    np.testing.assert_allclose(full["P_hPa"][above], 1000.0 * np.exp(-z[above] / 7.5), rtol=1e-3)


def test_lowest_levels_must_fall_in_pressure() -> None:
    with pytest.raises(ValueError, match="lowest two levels"):
        full_profile._extrapolate_pressure_below(
            GRID, np.ones(GRID.shape), np.array([1.5, 2.0]), np.array([850.0, 850.0])
        )
