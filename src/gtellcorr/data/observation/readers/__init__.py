"""Spectrum readers (plugin registry ``gtellcorr.readers``).

A reader turns one file + order into an :class:`OrderSpectrum` (fit-capable
pixels only) or a :class:`RawOrder` (all pixels, for output placeholders),
driven by the instrument's :class:`SpectralSpec`. Built-in readers:

* ``echelle_hdu`` — per-order 2-D image HDUs (NEID, ESPRESSO);
* ``pepsi_dxt``   — PEPSI ``.dxt.nor`` binary table holding one merged spectrum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gtellcorr.registry import Registry

if TYPE_CHECKING:
    from pathlib import Path

    from gtellcorr.data.observation.frames import VacuumAirFormula
    from gtellcorr.data.observation.instrument import SpectralSpec
    from gtellcorr.data.observation.spectrum import OrderSpectrum, RawOrder

__all__ = ["Reader", "readers"]


@dataclass(frozen=True)
class Reader:
    """A spectrum reader: two functions sharing the same arguments."""

    load_order: Callable[[Path, int, SpectralSpec, VacuumAirFormula], OrderSpectrum]
    load_order_raw: Callable[[Path, int, SpectralSpec, VacuumAirFormula], RawOrder]


readers: Registry[Reader] = Registry("spectrum reader", entry_point_group="gtellcorr.readers")


def _register_builtin() -> None:
    """Register the ``echelle_hdu`` and ``pepsi_dxt`` readers."""
    from gtellcorr.data.observation.readers import echelle_hdu, pepsi_dxt

    readers.register("echelle_hdu", Reader(echelle_hdu.load_order, echelle_hdu.load_order_raw))
    readers.register("pepsi_dxt", Reader(pepsi_dxt.load_order, pepsi_dxt.load_order_raw))


_register_builtin()
