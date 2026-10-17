"""Continuum optical depth of a ``scaled`` TauCache: the quadratic through the runs
at 0, 1 and 2 times the reference column, and its evaluation. The cache state is set
directly, without LBLRTM (the build is covered by the ``lblrtm`` integration test)."""

from __future__ import annotations

import numpy as np
import pytest

from gtellcorr.telluric.lblrtm.tau_cache import (
    REF_COLUMN,
    TauCache,
    _continuum_run_plan,
    _quadratic_continuum_coefficients,
)

GRID = np.linspace(686.0, 695.0, 401)
MOLECULES = ("h2o", "o2")


def _continuum(molecule: str, r: float) -> np.ndarray:
    """Synthetic continuum optical depth, quadratic in the relative column ``r`` with
    a column-independent part (Rayleigh-like)."""
    a = 0.01 + 0.002 * np.sin(GRID / 3.0)
    b = {"h2o": 0.003, "o2": 0.001}[molecule] * (1.0 + 0.1 * np.cos(GRID))
    c = {"h2o": 0.004, "o2": 0.006}[molecule] * np.exp(-0.5 * ((GRID - 690.0) / 2.0) ** 2)
    return a + b * r + c * r**2


def _cache(mode: str) -> TauCache:
    rng = np.random.default_rng(7)
    cache = TauCache.__new__(TauCache)
    cache.wavegrid = GRID
    cache.fit_molecules = MOLECULES
    cache.ref = {m: float(REF_COLUMN[m]) for m in MOLECULES}
    cache.tau_unit = {m: rng.uniform(0.0, 1e-3, GRID.size) / REF_COLUMN[m] for m in MOLECULES}
    cache.continuum_mode = mode
    cache.tau_cont_beta = {}
    cache.tau_cont_gamma = {}
    # reference continuum: both molecules at r = 1, column-independent part once
    reference = _continuum("h2o", 1.0) + _continuum("o2", 1.0) - (0.01 + 0.002 * np.sin(GRID / 3))
    lines_fixed = rng.uniform(0.0, 0.05, GRID.size)
    if mode == "none":
        cache.tau_continuum = None
        cache.tau_fixed = lines_fixed
    else:
        cache.tau_continuum = reference
        cache.tau_fixed = lines_fixed + reference
    if mode == "scaled":
        for m in MOLECULES:
            others = reference - _continuum(m, 1.0)
            beta, gamma = _quadratic_continuum_coefficients(
                others + _continuum(m, 0.0), reference, others + _continuum(m, 2.0)
            )
            cache.tau_cont_beta[m] = beta
            cache.tau_cont_gamma[m] = gamma
    return cache


def test_quadratic_recovers_the_three_runs_and_any_column() -> None:
    at = {r: _continuum("o2", r) for r in (0.0, 1.0, 2.0, 0.37, 3.5)}
    beta, gamma = _quadratic_continuum_coefficients(at[0.0], at[1.0], at[2.0])
    for r, expected in at.items():
        d = r - 1.0
        np.testing.assert_allclose(at[1.0] + beta * d + gamma * d * d, expected, rtol=1e-12)


def test_scaled_equals_full_at_the_reference_columns() -> None:
    full = _cache("full")
    scaled = _cache("scaled")
    assert full.optical_depth().tobytes() == scaled.optical_depth().tobytes()
    ref = {m: scaled.ref[m] for m in MOLECULES}
    assert full.optical_depth(**ref).tobytes() == scaled.optical_depth(**ref).tobytes()
    assert full.continuum_transmittance().tobytes() == scaled.continuum_transmittance().tobytes()


def test_scaled_continuum_follows_the_columns() -> None:
    cache = _cache("scaled")
    r = {"h2o": 1.7, "o2": 0.6}
    columns = {m: r[m] * cache.ref[m] for m in MOLECULES}
    rayleigh = 0.01 + 0.002 * np.sin(GRID / 3.0)
    expected_continuum = _continuum("h2o", r["h2o"]) + _continuum("o2", r["o2"]) - rayleigh
    np.testing.assert_allclose(
        cache.continuum_transmittance(**columns), np.exp(-expected_continuum), rtol=1e-12
    )
    lines = cache.tau_fixed - cache.tau_continuum
    expected = lines + expected_continuum + sum(columns[m] * cache.tau_unit[m] for m in MOLECULES)
    np.testing.assert_allclose(cache.optical_depth(**columns), expected, rtol=1e-12)


def test_full_continuum_ignores_the_columns() -> None:
    cache = _cache("full")
    columns = {m: 1.7 * cache.ref[m] for m in MOLECULES}
    assert (
        cache.continuum_transmittance(**columns).tobytes()
        == cache.continuum_transmittance().tobytes()
    )
    assert _cache("none").continuum_transmittance(**columns) is None


def test_unknown_molecule_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in fit_molecules"):
        _cache("scaled").continuum_transmittance(co2=400.0)


def test_continuum_run_plan() -> None:
    assert _continuum_run_plan(MOLECULES, "none") == ()
    assert [p[0] for p in _continuum_run_plan(MOLECULES, "full")] == [
        "continuum_full",
        "continuum_none",
    ]
    plan = _continuum_run_plan(MOLECULES, "scaled")
    assert len(plan) == 2 + 4 * len(MOLECULES)
    assert plan[2:] == (
        ("continuum_scaled_full", "h2o", 0.0, "full"),
        ("continuum_scaled_none", "h2o", 0.0, "none"),
        ("continuum_scaled_full", "h2o", 2.0, "full"),
        ("continuum_scaled_none", "h2o", 2.0, "none"),
        ("continuum_scaled_full", "o2", 0.0, "full"),
        ("continuum_scaled_none", "o2", 0.0, "none"),
        ("continuum_scaled_full", "o2", 2.0, "full"),
        ("continuum_scaled_none", "o2", 2.0, "none"),
    )
