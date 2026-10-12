"""Spectrum data types produced by the readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["OrderSpectrum", "RawOrder"]

Array = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class OrderSpectrum:
    """One fit-capable order in the AIR + TOPOCENTRIC frame.

    Only pixels with finite, positive wavelength and finite flux / positive error are
    kept; flux and error are divided by the median positive flux.
    """

    wave_fit: Array
    """Air + topocentric wavelengths [nm] (the telluric-model frame)."""
    wave_input: Array
    """Delivered wavelengths [nm] of the same pixels, for labelling outputs."""
    flux: Array
    """Flux divided by the median positive flux of the order."""
    cont: Array
    """Initial continuum guess (ones)."""
    err: Array
    """1-sigma flux uncertainty."""
    airmass: float
    """From the reader's ``airmass_card`` (1.5 if absent)."""


@dataclass(frozen=True, eq=False)
class RawOrder:
    """One order as delivered, without dropping unusable pixels; used for output
    placeholders and diagnostics."""

    wave_input: Array
    """Delivered wavelengths [nm]."""
    wave_fit: Array
    """NaN where the delivered wavelength is not finite and positive."""
    flux: Array
    """Flux divided by the median positive flux."""
    err: Array
    """1-sigma flux uncertainty, scaled like the flux."""
    cont: Array
    """Initial continuum guess (ones)."""
    airmass: float
    valid_fit_pixels: NDArray[np.bool_]
    """Pixels that :class:`OrderSpectrum` would keep."""
    status: str
    """``raw_only`` / ``no_valid_wavelength`` / ``no_valid_error`` / ``no_valid_flux``."""
