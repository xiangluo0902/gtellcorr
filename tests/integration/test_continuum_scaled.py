"""``continuum_mode="scaled"`` TauCache against direct LBLRTM runs.

A scaled cache is built under the O2 A band (O2 collision-induced absorption, H2O
continuum) with H2O and O2 fitted. Its continuum optical depth at other columns must
match the continuum of direct LBLRTM runs at those columns (with minus without the
continua), it must equal a ``full`` cache at the reference columns, and it must
survive a save/load round trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gtellcorr.data.observation.frames import air_to_vac
from gtellcorr.settings import Settings
from gtellcorr.telluric.lblrtm import load_lblrtm_resource_manifest, validate_lblrtm_resource_bundle
from gtellcorr.telluric.lblrtm.tau_cache import (
    MOLECULES,
    TauCache,
    _convert_result_grid,
    _scaled_relative_column,
)

pytestmark = pytest.mark.lblrtm

FIXTURES = Path(__file__).parents[1] / "fixtures" / "lblrtm"
FORMULA = "espresso"
WINDOW_AIR_NM = (758.0, 772.0)
FIT = ("h2o", "o2")


def _profile() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "reference_profile.npz", allow_pickle=False) as archive:
        profile = {k: np.asarray(archive[k], dtype=np.float64) for k in ("z_km", "P_hPa", "T_K")}
        for m in MOLECULES:
            profile[f"{m.upper()}_ppmv"] = np.asarray(archive[f"{m.upper()}_ppmv"], np.float64)
    return profile


def _cache(mode: str, resources: Any) -> TauCache:
    profile = _profile()
    state = json.loads((FIXTURES / "reference_state_definition.json").read_text())
    los = state["expected_tau_cache_identity"]["line_of_sight"]
    alt = float(los["observer_altitude_km"])
    vac = np.asarray(air_to_vac(np.asarray(WINDOW_AIR_NM), formula=FORMULA))
    return TauCache(
        lowfreq=1.0e7 / float(vac[-1]),
        highfreq=1.0e7 / float(vac[0]),
        pressure=float(np.interp(alt, profile["z_km"], profile["P_hPa"])),
        temperature=float(np.interp(alt, profile["z_km"], profile["T_K"])),
        airmass=float(los["effective_airmass"]),
        lat=float(los["reference_latitude_deg"]),
        alt=alt,
        vac2air=True,
        atmosphere_profile=profile,
        resources=resources,
        wavelength_conversion_formula=FORMULA,
        wave_margin_nm=0.0,
        continuum_mode=mode,
        optical_depth_grid_spacing_cm1=0.002,
        timeout_s=600.0,
        verbose=False,
    ).build(fit_molecules=FIT)


@pytest.fixture(scope="module")
def caches() -> dict[str, TauCache]:
    settings = Settings.load()
    assert settings.lblrtm_bundle is not None
    resources = validate_lblrtm_resource_bundle(
        load_lblrtm_resource_manifest(), settings.lblrtm_bundle
    )
    return {mode: _cache(mode, resources) for mode in ("full", "scaled")}


def _direct_continuum(cache: TauCache, relative: dict[str, float]) -> np.ndarray:
    """Continuum optical depth of direct LBLRTM runs with the fitted molecules at
    *relative* times their reference column, the others at their fixed column."""
    columns = {m: 0.0 for m in MOLECULES}
    for m, column in cache.fixed.items():
        columns[m] = column / cache.column_scale_denominators[m]
    for m in FIT:
        columns[m] = _scaled_relative_column(m, relative[m], cache.column_scale_denominators)
    depth = {}
    for mode in ("full", "none"):
        grid, depth[mode] = _convert_result_grid(
            cache._run_lblrtm(columns, continuum_mode=mode),
            vac2air=cache.vac2air,
            wavelength_conversion_formula=cache.wavelength_conversion_formula,
        )
        assert np.array_equal(grid, cache.wavegrid)
    return np.asarray(depth["full"] - depth["none"])


def test_scaled_equals_full_at_the_reference_columns(caches: dict[str, TauCache]) -> None:
    full, scaled = caches["full"], caches["scaled"]
    assert np.array_equal(full.wavegrid, scaled.wavegrid)
    assert full.tau_fixed.tobytes() == scaled.tau_fixed.tobytes()
    assert full.tau_continuum.tobytes() == scaled.tau_continuum.tobytes()
    assert full.optical_depth().tobytes() == scaled.optical_depth().tobytes()
    assert full.cache_key != scaled.cache_key


@pytest.mark.parametrize(
    ("relative", "tolerance"),
    [
        # O2 alone: the quadratic is exact (O2-O2 CIA ~ r**2, O2-N2 ~ r)
        ({"h2o": 1.0, "o2": 0.5}, 1e-9),
        # H2O: a small departure from a quadratic in LBLRTM (~5e-7 in optical depth
        # under the A band, where the whole H2O-continuum change is ~5e-6)
        ({"h2o": 1.7, "o2": 1.0}, 2e-6),
        # both, with the cross terms left out
        ({"h2o": 0.4, "o2": 1.3}, 2e-6),
    ],
)
def test_scaled_continuum_matches_direct_lblrtm(
    caches: dict[str, TauCache], relative: dict[str, float], tolerance: float
) -> None:
    scaled = caches["scaled"]
    columns = {m: relative[m] * scaled.ref[m] for m in FIT}
    predicted = -np.log(scaled.continuum_transmittance(**columns))
    direct = _direct_continuum(scaled, relative)
    # the frozen (full) continuum misses the whole change
    change = np.max(np.abs(direct - caches["full"].tau_continuum))
    error = np.max(np.abs(predicted - direct))
    assert change > 0.0
    assert error <= tolerance, (error, change)
    assert error <= 0.2 * change, (error, change)


def test_scaled_cache_round_trip(caches: dict[str, TauCache], tmp_path: Path) -> None:
    scaled = caches["scaled"]
    path = scaled.save(tmp_path / "scaled.npz")
    loaded = TauCache.load(path, expected_cache_key=scaled.cache_key)
    columns = {"h2o": 1.4 * scaled.ref["h2o"], "o2": 0.8 * scaled.ref["o2"]}
    assert loaded.continuum_mode == "scaled"
    assert loaded.optical_depth(**columns).tobytes() == scaled.optical_depth(**columns).tobytes()
    assert (
        loaded.continuum_transmittance(**columns).tobytes()
        == scaled.continuum_transmittance(**columns).tobytes()
    )
