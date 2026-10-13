"""Built-in parameter blocks: relative columns, resolving power, LSF shape and the
wavelength-solution correction."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from gtellcorr.fitting.blocks.base import ModelState, Scope

if TYPE_CHECKING:
    from gtellcorr.fitting.fit_spec import FitSpec

__all__ = [
    "LsfShapeBlock",
    "RelcolBlock",
    "ResolutionBlock",
    "SharedWaveShiftBlock",
    "WaveShiftBlock",
]

Array = NDArray[np.float64]


class RelcolBlock:
    """Relative column of each fitted molecule; shared across orders in a joint fit."""

    name: ClassVar[str] = "relcol"
    scope: ClassVar[Scope] = "exposure"

    def active(self, spec: FitSpec) -> bool:
        """Whenever a molecule is fitted."""
        return bool(spec.molecules)

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        """``relcol_<molecule>``, in LBLRTM ID order."""
        return tuple(f"relcol_{m}" for m in spec.molecules)

    def initial(self, spec: FitSpec) -> Array:
        """The molecules' configured ``relcol``."""
        return np.asarray([float(spec.relcol_init[m]) for m in spec.molecules], float)

    def bounds(self, spec: FitSpec) -> tuple[Array, Array]:
        """The molecules' configured ``bounds``."""
        lo = [float(spec.relcol_bounds[m][0]) for m in spec.molecules]
        hi = [float(spec.relcol_bounds[m][1]) for m in spec.molecules]
        return np.asarray(lo, float), np.asarray(hi, float)

    def apply(self, values: Array, spec: FitSpec, state: ModelState) -> ModelState:
        """Set the relative columns."""
        scales = {m: float(values[i]) for i, m in enumerate(spec.molecules)}
        return dataclasses.replace(state, scales=scales)


class ResolutionBlock:
    """Resolving power R of the LSF; one per order."""

    name: ClassVar[str] = "resolution"
    scope: ClassVar[Scope] = "order"

    def active(self, spec: FitSpec) -> bool:
        """When ``lsf.fit_resolution``."""
        return bool(spec.fit_resolution)

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        """``resolution``."""
        return ("resolution",)

    def initial(self, spec: FitSpec) -> Array:
        """The initial R of the spec (configured, header or instrument R)."""
        return np.asarray([float(spec.resolution_init)], float)

    def bounds(self, spec: FitSpec) -> tuple[Array, Array]:
        """The spec's R bounds."""
        lo, hi = spec.resolution_bounds
        return np.asarray([float(lo)], float), np.asarray([float(hi)], float)

    def apply(self, values: Array, spec: FitSpec, state: ModelState) -> ModelState:
        """Set R."""
        return dataclasses.replace(state, resolution=float(values[0]))


class LsfShapeBlock:
    """Shape of a shaped LSF (super-Gaussian exponent, Voigt weight); one per order."""

    name: ClassVar[str] = "lsf_shape"
    scope: ClassVar[Scope] = "order"

    def active(self, spec: FitSpec) -> bool:
        """When ``lsf.fit_shape`` and the LSF kind has a shape."""
        return bool(spec.fit_lsf_shape)

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        """``lsf_shape``."""
        return ("lsf_shape",)

    def initial(self, spec: FitSpec) -> Array:
        """The configured shape of the active kind."""
        return np.asarray([float(spec.lsf_shape)], float)  # type: ignore[arg-type]

    def bounds(self, spec: FitSpec) -> tuple[Array, Array]:
        """The configured shape bounds of the active kind."""
        lo, hi = spec.lsf_shape_bounds  # type: ignore[misc]
        return np.asarray([float(lo)], float), np.asarray([float(hi)], float)

    def apply(self, values: Array, spec: FitSpec, state: ModelState) -> ModelState:
        """Set the LSF shape."""
        return dataclasses.replace(state, lsf_shape=float(values[0]))


class WaveShiftBlock:
    """Wavelength-solution correction as a velocity [km/s]; Chebyshev coefficients per order.

    Active for ``wavelength.scope = "order"``.
    """

    name: ClassVar[str] = "wave_shift"
    scope: ClassVar[Scope] = "order"

    def active(self, spec: FitSpec) -> bool:
        """When ``wavelength.fit`` and ``wavelength.scope`` is this block's scope."""
        return bool(spec.fit_wave_shift) and spec.wave_shift_scope == self.scope

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        """``wave_shift_c0`` .. ``wave_shift_c<degree>``."""
        return tuple(f"wave_shift_c{k}" for k in range(int(spec.wave_shift_degree) + 1))

    def initial(self, spec: FitSpec) -> Array:
        """No correction."""
        return np.zeros(int(spec.wave_shift_degree) + 1, float)

    def bounds(self, spec: FitSpec) -> tuple[Array, Array]:
        """``wavelength.bounds_kms`` for every coefficient."""
        n = int(spec.wave_shift_degree) + 1
        lo, hi = spec.wave_shift_bounds
        return np.full(n, float(lo)), np.full(n, float(hi))

    def apply(self, values: Array, spec: FitSpec, state: ModelState) -> ModelState:
        """Set the correction coefficients."""
        return dataclasses.replace(state, wave_shift=tuple(float(v) for v in values))


class SharedWaveShiftBlock(WaveShiftBlock):
    """One constant velocity correction shared by all orders of a joint fit
    (``wavelength.scope = "exposure"``, degree 0)."""

    scope: ClassVar[Scope] = "exposure"
