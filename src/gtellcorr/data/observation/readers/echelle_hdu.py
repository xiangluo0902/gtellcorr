"""Reader for per-order 2-D image HDUs (NEID L2, ESPRESSO S2D).

The HDU names, units and frame come from the instrument's :class:`SpectralSpec`.
Flux and uncertainty are divided by the blaze (if declared) and then by the median
positive flux of the order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

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


def _quality_flags(path: Path, hdu: str, order: int) -> NDArray[np.bool_]:
    """Pixels of one order with a non-zero quality flag.

    Integer flag images are usually stored with ``BZERO`` (unsigned 16 bit), which
    astropy cannot scale in a memory map; the stored values are read unscaled and
    scaled here.
    """
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as h:
        image = h[hdu]
        stored = np.asarray(image.data[order], np.float64)
        bscale = float(image.header.get("BSCALE", 1.0))
        bzero = float(image.header.get("BZERO", 0.0))
    return np.asarray(stored * bscale + bzero != 0)


def _read(path: Path, order: int, spec: SpectralSpec) -> tuple[Any, ...]:
    """Header, wavelengths [nm], de-blazed flux and error, and flagged pixels of one
    order."""
    if spec.wave is None or spec.flux is None or spec.uncertainty is None:
        raise DataError("echelle_hdu reader needs spectral.wave, .flux and .uncertainty")
    with fits.open(path, memmap=True) as h:
        hdr0 = h[0].header
        wave = np.asarray(h[spec.wave].data[order], float)
        flux = h[spec.flux].data[order].astype(np.float64)
        unc = h[spec.uncertainty].data[order].astype(np.float64)
        blaze = h[spec.blaze].data[order].astype(np.float64) if spec.blaze else np.ones_like(flux)
    flagged = (
        _quality_flags(path, spec.quality, order) if spec.quality else np.zeros(flux.shape, bool)
    )
    err = np.sqrt(np.clip(unc, 0, None)) if spec.uncertainty_kind == "var" else unc
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_n = np.where(blaze > 0, flux / blaze, np.nan)
        err_n = np.where(blaze > 0, err / blaze, np.nan)
    wave_nm = wave / wave_scale(spec.wave_unit)
    return hdr0, wave_nm, flux_n, err_n, flagged


def load_order(
    path: Path, order: int, spec: SpectralSpec, formula: VacuumAirFormula
) -> OrderSpectrum:
    """Fit-capable pixels of one order (see :class:`OrderSpectrum`)."""
    hdr0, wave_nm, flux_n, err_n, flagged = _read(path, order, spec)
    # Keep finite non-positive fluxes: saturated telluric cores are excluded later by
    # the strong-telluric mask, so ingestion must not censor negative noise only.
    mask = (
        np.isfinite(wave_nm)
        & (wave_nm > 0)
        & np.isfinite(flux_n)
        & np.isfinite(err_n)
        & (err_n > 0)
        & ~flagged
    )
    x_orig_nm = wave_nm[mask]
    x_fit_nm = to_air_topocentric(x_orig_nm, spec.frame, hdr0, formula=formula)
    y = flux_n[mask]
    e = err_n[mask]
    positive = y[np.isfinite(y) & (y > 0)]
    med = np.nanmedian(positive) if positive.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    y = y / med
    e = e / med
    grid_wave_fit = raw_wave_fit_nm(wave_nm, spec.frame, hdr0, formula)
    grid_wave_fit[mask] = x_fit_nm
    grid = InputGrid(
        wave_input=wave_nm, wave_fit=grid_wave_fit, flux=flux_n / med, err=err_n / med, kept=mask
    )
    return OrderSpectrum(
        wave_fit=x_fit_nm,
        wave_input=x_orig_nm,
        flux=y,
        cont=np.ones_like(y),
        err=e,
        grid=grid,
    )


def load_order_raw(
    path: Path, order: int, spec: SpectralSpec, formula: VacuumAirFormula
) -> RawOrder:
    """All pixels of one order (see :class:`RawOrder`)."""
    hdr0, wave_nm, flux_n, err_n, flagged = _read(path, order, spec)
    wave_fit_nm = raw_wave_fit_nm(wave_nm, spec.frame, hdr0, formula)
    flux_n, err_n = normalise_positive_flux(flux_n, err_n)
    valid = (
        np.isfinite(wave_nm)
        & (wave_nm > 0)
        & np.isfinite(flux_n)
        & np.isfinite(err_n)
        & (err_n > 0)
        & ~flagged
    )
    return RawOrder(
        wave_input=wave_nm,
        wave_fit=wave_fit_nm,
        flux=flux_n,
        err=err_n,
        cont=np.ones_like(flux_n),
        valid_fit_pixels=valid,
        status=raw_order_status(wave_nm, flux_n, err_n),
    )
