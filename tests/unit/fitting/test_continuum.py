"""Continuum projected out of the telluric fit (``fitting/continuum.py``)."""

from __future__ import annotations

import numpy as np

from gtellcorr.fitting.config import Continuum
from gtellcorr.fitting.continuum import fit_continuum


def _spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = np.linspace(690.0, 700.0, 2000)
    model = 1.0 - 0.5 * np.exp(-0.5 * ((x - 695.0) / 0.01) ** 2)
    continuum = 1.0 + 0.02 * (x - 695.0)
    flux = continuum * model + rng.normal(0.0, 0.002, x.size)
    flux[100] += 0.5  # a cosmic-ray-like outlier
    weight = np.full(x.size, 1.0 / 0.002**2)
    return x, flux, model, weight


def test_one_iteration_is_the_unclipped_weighted_fit() -> None:
    x, flux, model, weight = _spectrum()
    got = fit_continuum(x, flux, model, order=2, max_iter=1, weight=weight)
    xn = (x - x.mean()) / np.ptp(x)
    coef = np.polyfit(xn, flux / model, 2, w=np.sqrt(weight * model**2))
    np.testing.assert_allclose(got, np.polyval(coef, xn), rtol=1e-12)


def test_clipping_rejects_the_outlier() -> None:
    x, flux, model, weight = _spectrum()
    clean = flux.copy()
    clean[100] -= 0.5
    reference = fit_continuum(x, clean, model, order=2, max_iter=1, weight=weight)
    clipped = fit_continuum(x, flux, model, order=2, max_iter=5, weight=weight)
    unclipped = fit_continuum(x, flux, model, order=2, max_iter=1, weight=weight)
    assert np.max(np.abs(clipped - reference)) < np.max(np.abs(unclipped - reference))


def test_unclipped_continuum_is_continuous_in_the_model() -> None:
    """Without clipping the continuum (hence the fit objective) changes smoothly with
    the model: a tiny change of the model gives a proportionally tiny change."""
    x, flux, model, weight = _spectrum()
    kwargs = {"order": Continuum().order, "max_iter": Continuum().niter, "weight": weight}
    base = fit_continuum(x, flux, model, **kwargs)
    steps = (1 + 1e-6, 1 + 1e-4)
    deltas = [np.max(np.abs(fit_continuum(x, flux, model**s, **kwargs) - base)) for s in steps]
    assert 50 < deltas[1] / deltas[0] < 200
