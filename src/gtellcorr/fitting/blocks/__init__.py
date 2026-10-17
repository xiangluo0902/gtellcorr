"""Parameter blocks: the free parameters of the telluric fit.

A :class:`ParameterBlock` is one kind of free parameter of the telluric fit. The
solver builds theta by concatenating the active blocks; in a joint fit,
``exposure``-scope blocks are shared by all orders and ``order``-scope blocks are
repeated per order. Adding a fit parameter means writing a block and registering
it (entry-point group ``gtellcorr.parameter_blocks``); the solver is unchanged.

With the built-in blocks theta = ``[relcol..., resolution?, lsf_shape?]``, followed
by the template velocity when ``stellar_model.fit_shift`` is set (``template``
treatment) and the wavelength-solution correction coefficients when
``wavelength.fit`` is set.
"""

from __future__ import annotations

from gtellcorr.fitting.blocks.base import ModelState, ParameterBlock, Scope
from gtellcorr.fitting.blocks.builtin import (
    LsfShapeBlock,
    RelcolBlock,
    ResolutionBlock,
    SharedWaveShiftBlock,
    StellarShiftBlock,
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
    "StellarShiftBlock",
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
parameter_blocks.register("stellar_shift", StellarShiftBlock)


def default_blocks() -> tuple[ParameterBlock, ...]:
    """The built-in blocks in theta order: relcol, resolution, LSF shape, the
    exposure-wide wavelength-solution correction, the template velocity, then the
    per-order wavelength-solution correction."""
    return (
        RelcolBlock(),
        ResolutionBlock(),
        LsfShapeBlock(),
        SharedWaveShiftBlock(),
        StellarShiftBlock(),
        WaveShiftBlock(),
    )
