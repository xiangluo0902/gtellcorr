"""Continuum of the stellar master spectrum, per order.

``ispec``: iSpec's upper-envelope method (median + max base points, strong-line
rejection, spline), with broad-winged lines excluded from the base points.
``poly``: a polynomial fitted with asymmetric sigma clipping, which rejects the
absorption lines below it. Any iSpec failure (including a missing ``ispec_dir``)
falls back to ``poly`` with a warning.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.preprocessing._ispec import import_ispec
from gtellcorr.preprocessing.windows import Window

if TYPE_CHECKING:
    from gtellcorr.preprocessing.config import MasterContinuum

__all__ = ["normalise_continuum"]

log = logging.getLogger(__name__)
Array = NDArray[np.float64]


def _ispec_continuum(
    wave_nm: Any,
    flux: Any,
    err: Any,
    config: MasterContinuum,
    broad_windows: Iterable[Window],
    ispec_dir: str | os.PathLike[str] | None,
) -> Array:
    """iSpec ``fit_continuum`` with the ``master_spectrum.continuum`` settings; the
    knots are ``nknots`` or one per ``knot_wave_nm``, the errors weight the fit if any
    is positive, and *broad_windows* overlapping the order are ignored when
    ``ignore_broad_lines``."""
    ispec = import_ispec(ispec_dir)
    w = np.asarray(wave_nm, float)
    spec = ispec.create_spectrum_structure(w)
    spec["flux"] = np.asarray(flux, float)
    use_err = err is not None and np.any(np.asarray(err) > 0)
    if use_err:
        spec["err"] = np.asarray(err, float)
    nknots = config.nknots
    if nknots is None:
        span = float(w.max() - w.min())
        nknots = max(1, round(span / config.knot_wave_nm))
    ignore = None
    windows = list(broad_windows) if config.ignore_broad_lines else []
    windows = [(lo, hi) for lo, hi in windows if hi >= w.min() and lo <= w.max()]
    if windows:
        ignore = np.array(
            [(lo, hi) for lo, hi in windows], dtype=[("wave_base", float), ("wave_top", float)]
        )
    model = ispec.fit_continuum(
        spec,
        from_resolution=config.from_resolution,
        ignore=ignore,
        nknots=nknots,
        degree=config.degree,
        median_wave_range=config.median_wave_range,
        max_wave_range=config.max_wave_range,
        model=config.model,
        order=config.order,
        automatic_strong_line_detection=config.strong_line_detection,
        strong_line_probability=config.strong_line_probability,
        use_errors_for_fitting=use_err,
    )
    return np.asarray(model(w), float)


def _poly_continuum(wave_nm: Any, flux: Any, config: MasterContinuum) -> Array:
    """Polynomial of order ``poly_order`` in wavelength, refitted up to six times to the
    points within -1.5 to +3 sigma of the previous fit; 1 everywhere if there are
    too few points for a first fit."""
    w = np.asarray(wave_nm, float)
    f = np.asarray(flux, float)
    w0 = w - w.mean()
    order = int(config.poly_order)
    good = np.isfinite(f)
    cont = np.ones_like(f)
    for _ in range(6):
        if good.sum() < order + 2:
            break
        coef = np.polyfit(w0[good], f[good], order)
        cont = np.polyval(coef, w0)
        resid = f - cont
        s = np.std(resid[good])
        if not (s > 0):
            break
        good = np.isfinite(f) & (resid < 3.0 * s) & (resid > -1.5 * s)
    return cont


def normalise_continuum(
    wave_nm: Any,
    flux: Any,
    *,
    config: MasterContinuum,
    broad_windows: Iterable[Window] = (),
    err: Any = None,
    ispec_dir: str | os.PathLike[str] | None = None,
) -> tuple[Array, Array]:
    """``(continuum, flux / continuum)`` of one spectrum (``config.method``); where the
    continuum is non-finite or not positive it is set to 1."""
    if config.method == "ispec":
        try:
            cont = _ispec_continuum(wave_nm, flux, err, config, broad_windows, ispec_dir)
        except Exception as exc:
            log.warning(
                "iSpec continuum failed (%s: %s); using the polynomial", type(exc).__name__, exc
            )
            cont = _poly_continuum(wave_nm, flux, config)
    else:
        cont = _poly_continuum(wave_nm, flux, config)
    f = np.asarray(flux, float)
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, 1.0)
    return cont, f / cont
