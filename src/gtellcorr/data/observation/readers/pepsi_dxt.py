"""Reader for PEPSI ``.dxt.nor`` files: a binary table (HDU 1) holding ONE merged,
continuum-normalised spectrum.

Without ``order_windows_nm`` the whole spectrum is order 0 (PEPSI is fibre-fed with
one R and one atmospheric column); ``order_windows_nm`` can split it into
wavelength windows treated as orders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from gtellcorr.data.observation.frames import VacuumAirFormula, to_air_topocentric
from gtellcorr.data.observation.instrument import SpectralSpec
from gtellcorr.data.observation.readers._common import (
    normalise_positive_flux,
    raw_order_status,
    raw_wave_fit_nm,
    wave_scale,
)
from gtellcorr.data.observation.spectrum import InputGrid, OrderSpectrum, RawOrder
from gtellcorr.errors import DataError


def _read(path: Path, order: int, spec: SpectralSpec) -> tuple[Any, ...]:
    """Header, wavelengths [nm], flux, variance, error, bad-pixel mask and order
    windows of the merged spectrum."""
    windows = spec.order_windows_nm
    if windows is None and order != 0:
        raise DataError(f"pepsi_dxt: no order_windows_nm, so only order 0 exists; got {order}")
    col = spec.columns
    missing = sorted({"wave", "flux", "variance"} - set(col))
    if missing:
        raise DataError(f"pepsi_dxt reader needs spectral.columns {missing}")
    with fits.open(path, memmap=True) as h:
        hdr0 = h[0].header
        table = h[1].data
        wave = np.asarray(table[col["wave"]], float)
        flux = np.asarray(table[col["flux"]], float)
        var = np.asarray(table[col["variance"]], float)
        mask_col = col.get("mask")
        bad = (
            np.asarray(table[mask_col]).astype(bool)
            if mask_col in table.columns.names
            else np.zeros(wave.shape, bool)
        )
    wave_nm = wave / wave_scale(spec.wave_unit)
    err = np.sqrt(np.clip(var, 0, None)) if spec.uncertainty_kind == "var" else var
    return hdr0, wave_nm, flux, var, err, bad, windows


def load_order(
    path: Path, order: int, spec: SpectralSpec, formula: VacuumAirFormula
) -> OrderSpectrum:
    """Fit-capable pixels of one order (see :class:`OrderSpectrum`)."""
    hdr0, wave_nm, flux, var, err, bad, windows = _read(path, order, spec)
    wave_topo = to_air_topocentric(wave_nm, spec.frame, hdr0, formula=formula)
    # Keep finite non-positive fluxes: deep telluric cores can scatter below zero.
    sel = (~bad) & np.isfinite(flux) & np.isfinite(err) & (var > 0)
    region = np.ones(wave_nm.shape, bool)
    if windows is not None:
        lo, hi = windows[order]
        region = (wave_topo >= lo) & (wave_topo <= hi)
        sel &= region
    x = wave_topo[sel]
    x_in = wave_nm[sel]
    y = flux[sel]
    e = err[sel]
    positive = y[np.isfinite(y) & (y > 0)]
    med = np.nanmedian(positive) if positive.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    y = y / med
    e = e / med
    grid_wave_fit = raw_wave_fit_nm(wave_nm, spec.frame, hdr0, formula)
    grid_wave_fit[sel] = x
    grid = InputGrid(
        wave_input=wave_nm[region],
        wave_fit=grid_wave_fit[region],
        flux=flux[region] / med,
        err=err[region] / med,
        kept=sel[region],
    )
    return OrderSpectrum(
        wave_fit=x, wave_input=x_in, flux=y, cont=np.ones_like(y), err=e, grid=grid
    )


def load_order_raw(
    path: Path, order: int, spec: SpectralSpec, formula: VacuumAirFormula
) -> RawOrder:
    """All pixels of one order (see :class:`RawOrder`)."""
    hdr0, wave_nm, flux, _var, err, bad, windows = _read(path, order, spec)
    wave_fit_nm = raw_wave_fit_nm(wave_nm, spec.frame, hdr0, formula)
    flux_n, err_n = normalise_positive_flux(flux, err)
    if windows is not None:
        lo, hi = windows[order]
        region = (wave_fit_nm >= lo) & (wave_fit_nm <= hi)
    else:
        region = np.ones(wave_nm.shape, bool)
    sel = region & ~bad
    w_in, w_fit, f, e = wave_nm[sel], wave_fit_nm[sel], flux_n[sel], err_n[sel]
    valid = np.isfinite(w_fit) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    grid = InputGrid(
        wave_input=wave_nm[region],
        wave_fit=wave_fit_nm[region],
        flux=flux_n[region],
        err=err_n[region],
        kept=sel[region],
    )
    return RawOrder(
        wave_input=w_in,
        wave_fit=w_fit,
        flux=f,
        err=e,
        cont=np.ones(np.count_nonzero(sel), dtype=float),
        valid_fit_pixels=valid,
        status=raw_order_status(w_in, f, e),
        grid=grid,
    )
