"""Continuum of the telluric fit, projected out at every residual evaluation.

Given the observed flux and the current model on the same grid, the continuum C
is the low-order polynomial that best maps model→obs.  It is *projected out* every
residual evaluation rather than carried in the optimizer vector (keeps the outer
fit low-dim).  The implementation is the exact Golub–Pereyra inner solution:
minimizing the same weighted objective as the outer fit,
Σ w·(obs − C·model)², over the polynomial C is a weighted least-squares fit of
``obs/model`` with per-pixel weight ``w·model²`` (deep cores + noisy pixels
naturally down-weighted, as in molecfit's model-weighted ``match``).

An explicit inverse-variance ``weight`` is mandatory: ``weight=None`` is rejected
rather than silently changing the objective to an unweighted continuum fit. The
model is floored at 0.05 before dividing, so saturated cores do not blow up the
ratio. The fit honours ``mask`` (True = keep).
"""

from __future__ import annotations

import numpy as np

_MODEL_FLOOR = 0.05  # don't divide obs by near-zero (deep line cores)


def _weighted_continuum(wave_nm, ratio, model, w, order, lo, hi, max_iter, mask):
    """Weighted σ-clipped polynomial of ``ratio = obs/model`` with VP weight
    ``w·model²`` — the exact minimizer of Σ w·(obs − C·model)² over C.

    At most 20 clipping iterations (fewer if *max_iter* is smaller); clipping stops
    early when nothing changes. Too few pixels for a first fit give the median
    ratio (1 if none is finite)."""
    wt = np.asarray(w, float) * model**2  # VP weight per pixel
    keep = np.isfinite(ratio) & np.isfinite(wt) & (wt > 0)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    xn = (wave_nm - wave_nm.mean()) / (np.ptp(wave_nm) or 1.0)  # condition the poly
    coef = None
    for _ in range(max(min(int(max_iter), 20), 1)):  # σ-clip iterations (fast-converging)
        if keep.sum() < int(order) + 2:
            break
        coef = np.polyfit(xn[keep], ratio[keep], int(order), w=np.sqrt(wt[keep]))
        resid = ratio - np.polyval(coef, xn)
        s = np.std(resid[keep])
        new = keep & (resid >= -lo * s) & (resid <= hi * s)  # drop absorption / cosmics
        if s == 0 or new.sum() == keep.sum() or new.sum() < int(order) + 2:
            break
        keep = new
    if coef is None:  # too few pixels → flat
        fin = np.isfinite(ratio)
        return np.full_like(ratio, np.nanmedian(ratio[fin]) if fin.any() else 1.0)
    return np.polyval(coef, xn)


def fit_continuum(
    wave_nm: np.ndarray,
    obs_flux: np.ndarray,
    model_trans: np.ndarray,
    *,
    order: int = 7,
    sigma_clip: tuple[float, float] = (2.0, 3.0),
    max_iter: int = 10000,
    mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return the continuum array ``C`` such that ``obs_flux ≈ C · model_trans``.

    ``weight`` is the explicit per-pixel inverse variance ``w = 1/σ²`` from the
    fit.  The continuum is the exact weighted variable-projection solution — a
    σ-clipped weighted polynomial of ``obs/model`` with weight ``w·model²`` (see
    module docstring).  ``None`` is rejected.

    ``mask`` (True = KEEP): fit only on the kept pixels; excluded pixels have
    already been removed by the caller's active mask union.
    ``sigma_clip = (lowreject, highreject)`` are the σ thresholds (low rejects
    absorption, high rejects emission/cosmics).  Returns ``C`` on the input grid.
    """
    if weight is None:
        raise ValueError(
            "fit_continuum requires explicit inverse-variance weight; "
            "unweighted continuum fitting is not supported"
        )
    wave_nm = np.asarray(wave_nm, float)
    obs_flux = np.asarray(obs_flux, float)
    model = np.clip(np.asarray(model_trans, float), _MODEL_FLOOR, None)
    ratio = obs_flux / model
    lo, hi = sigma_clip
    return _weighted_continuum(wave_nm, ratio, model, weight, order, lo, hi, max_iter, mask)
