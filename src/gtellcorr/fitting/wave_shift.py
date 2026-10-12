"""Wavelength offset of the telluric model.

The offset is a velocity ``v(x) = sum_k c_k T_k(u)`` [km/s]. The Chebyshev argument
is a velocity coordinate too: ``u`` maps ``ln x`` (``d ln x = dv / c``) over the
order's finite positive wavelength range onto [-1, 1], so equal steps in ``u`` are
equal steps in velocity across the order. The model is resampled as
``T_obs(x) = T_model(x / (1 + v(x) / c))``: a positive ``v`` moves the telluric
lines to the red. Only the telluric transmission is shifted; the data grid and the
stellar model (placed by the measured RV) are not.

The same factor read the other way round gives the wavelengths of the data pixels
anchored on the telluric lines, ``lambda / (1 + v(x) / c)``
(:func:`corrected_wavelengths`). They are an output only: the fit, the correction and
the stellar RV all use the delivered wavelength solution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.polynomial import chebyshev
from numpy.typing import NDArray

__all__ = ["C_KMS", "corrected_wavelengths", "offset_velocity", "shifted_grid"]

C_KMS = 299792.458
"""Speed of light [km/s]."""


def offset_velocity(x: Any, coefficients: Sequence[float]) -> NDArray[np.float64]:
    """``v(x)`` [km/s] for Chebyshev *coefficients* in ``ln x`` over the finite range of *x*."""
    x = np.asarray(x, float)
    coeffs = np.asarray(coefficients, float)
    usable = x[np.isfinite(x) & (x > 0.0)]
    if coeffs.size == 1 or usable.size < 2 or usable.max() == usable.min():
        return np.full(x.shape, float(coeffs[0]) if coeffs.size else 0.0)
    lo, hi = np.log(float(usable.min())), np.log(float(usable.max()))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 2.0 * (np.log(x) - lo) / (hi - lo) - 1.0
    # numpy < 2 types chebval as untyped, numpy >= 2 does not
    return np.asarray(chebyshev.chebval(u, coeffs), dtype=np.float64)  # type: ignore[no-untyped-call, unused-ignore]


def shifted_grid(x: Any, coefficients: Sequence[float]) -> NDArray[np.float64]:
    """Model wavelengths sampled at the data wavelengths *x*: ``x / (1 + v(x) / c)``."""
    x = np.asarray(x, float)
    return np.asarray(x / (1.0 + offset_velocity(x, coefficients) / C_KMS), float)


def corrected_wavelengths(
    wave_fit: Any, wave_input: Any, coefficients: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Telluric-anchored wavelengths of the same pixels: ``(wave_fit, wave_input) / (1 + v / c)``.

    ``v`` is evaluated on *wave_fit*, the grid the offset was fitted on, so
    ``wave_fit / (1 + v / c)`` equals :func:`shifted_grid`. The Doppler factor is
    frame-independent and is applied unchanged to the delivered grid *wave_input*.
    """
    wave_fit = np.asarray(wave_fit, float)
    wave_input = np.asarray(wave_input, float)
    factor = 1.0 + offset_velocity(wave_fit, coefficients) / C_KMS
    return np.asarray(wave_fit / factor, float), np.asarray(wave_input / factor, float)
