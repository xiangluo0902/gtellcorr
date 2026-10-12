"""Helpers shared by the built-in readers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.data.observation.frames import FrameSpec, VacuumAirFormula, to_air_topocentric

Array = NDArray[np.float64]

#: Airmass used when the header has no usable airmass.
FALLBACK_AIRMASS = 1.5


def normalise_positive_flux(flux: Array, err: Array) -> tuple[Array, Array]:
    """Divide by the median positive flux (1 if there is none)."""
    positive = flux[np.isfinite(flux) & (flux > 0)]
    med = np.nanmedian(positive) if positive.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return flux / med, err / med


def raw_order_status(wave_nm: Array, flux: Array, err: Array) -> str:
    """Why an order cannot be fitted, or ``raw_only`` if its arrays are usable."""
    if not np.any(np.isfinite(wave_nm) & (wave_nm > 0)):
        return "no_valid_wavelength"
    if not np.any(np.isfinite(err) & (err > 0)):
        return "no_valid_error"
    if not np.any(np.isfinite(flux)):
        return "no_valid_flux"
    return "raw_only"


def raw_wave_fit_nm(
    wave_nm: Array, frame: FrameSpec, header: Any, formula: VacuumAirFormula
) -> Array:
    """Fit-frame wavelengths, NaN where the delivered wavelength is unusable."""
    wave_nm = np.asarray(wave_nm, float)
    out = np.full(wave_nm.shape, np.nan, dtype=float)
    valid = np.isfinite(wave_nm) & (wave_nm > 0)
    if np.any(valid):
        out[valid] = to_air_topocentric(wave_nm[valid], frame, header, formula=formula)
    return out


def wave_scale(wave_unit: str) -> float:
    """Divisor that converts delivered wavelengths to nm."""
    return 10.0 if wave_unit == "A" else 1.0
