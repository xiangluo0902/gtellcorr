"""Spectrum data types produced by the readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["InputGrid", "OrderSpectrum", "RawOrder"]

Array = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class InputGrid:
    """Every pixel of an order as delivered, and which of them a spectrum holds.

    Results are written on this grid, so that pixel ``i`` of a result row is pixel
    ``i`` of the delivered order. Flux and error are scaled like the spectrum's.
    """

    wave_input: Array
    """Delivered wavelengths [nm] of every pixel, as delivered."""
    wave_fit: Array
    """Air + topocentric wavelengths [nm]; NaN where the delivered wavelength is not
    finite and positive."""
    flux: Array
    """Delivered flux, scaled like the spectrum's flux."""
    err: Array
    """Delivered 1-sigma uncertainty, scaled like the flux."""
    kept: NDArray[np.bool_]
    """Pixels the spectrum holds, in the same order."""

    def __post_init__(self) -> None:
        n = self.kept.size
        for name in ("wave_input", "wave_fit", "flux", "err"):
            shape = np.shape(getattr(self, name))
            if shape != (n,):
                raise ValueError(f"InputGrid.{name}: shape {shape} != ({n},)")


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
    grid: InputGrid | None = None
    """The delivered order these pixels come from (``None``: the pixels are the whole
    order, e.g. a spectrum built in memory)."""

    def __post_init__(self) -> None:
        if self.grid is not None and int(np.count_nonzero(self.grid.kept)) != self.flux.size:
            raise ValueError(
                f"OrderSpectrum: grid keeps {int(np.count_nonzero(self.grid.kept))} pixels, "
                f"the spectrum has {self.flux.size}"
            )


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
    valid_fit_pixels: NDArray[np.bool_]
    """Pixels that :class:`OrderSpectrum` would keep."""
    status: str
    """``raw_only`` / ``no_valid_wavelength`` / ``no_valid_error`` / ``no_valid_flux``."""
    grid: InputGrid | None = None
    """The delivered order these pixels come from (``None``: the pixels are the whole
    order)."""
