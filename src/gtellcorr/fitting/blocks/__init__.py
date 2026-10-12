"""Parameter blocks: the free parameters of the telluric fit.

A :class:`ParameterBlock` is one kind of free parameter of the telluric fit. The
solver builds theta by concatenating the active blocks; in a joint fit,
``exposure``-scope blocks are shared by all orders and ``order``-scope blocks are
repeated per order. Adding a fit parameter means writing a block and registering
it (entry-point group ``gtellcorr.parameter_blocks``); the solver is unchanged.

With the built-in blocks theta = ``[relcol..., resolution?, lsf_shape?]``, followed
by the wavelength-offset coefficients when ``wavelength.fit`` is set.
"""

from __future__ import annotations

from gtellcorr.fitting.blocks.base import ModelState, ParameterBlock, Scope
from gtellcorr.fitting.blocks.builtin import (
    LsfShapeBlock,
    RelcolBlock,
    ResolutionBlock,
    SharedWaveShiftBlock,
    WaveShiftBlock,
)
from gtellcorr.registry import Registry

__all__ = [
    "LsfShapeBlock",
    "ModelState",
    "ParameterBlock",
    "RelcolBlock",
    "ResolutionBlock",
    "Scope",
    "SharedWaveShiftBlock",
    "WaveShiftBlock",
    "default_blocks",
    "parameter_blocks",
]

#: Registered block classes, in theta order.
parameter_blocks: Registry[type] = Registry(
    "parameter block", entry_point_group="gtellcorr.parameter_blocks"
)
parameter_blocks.register("relcol", RelcolBlock)
parameter_blocks.register("resolution", ResolutionBlock)
parameter_blocks.register("lsf_shape", LsfShapeBlock)
parameter_blocks.register("wave_shift", WaveShiftBlock)
parameter_blocks.register("shared_wave_shift", SharedWaveShiftBlock)


def default_blocks() -> tuple[ParameterBlock, ...]:
    """The built-in blocks in theta order: relcol, resolution, LSF shape, then the
    wavelength offsets (exposure-wide, then per order)."""
    return (
        RelcolBlock(),
        ResolutionBlock(),
        LsfShapeBlock(),
        SharedWaveShiftBlock(),
        WaveShiftBlock(),
    )
