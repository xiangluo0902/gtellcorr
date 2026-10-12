"""ParameterBlock protocol and the model state it acts on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gtellcorr.fitting.fit_spec import FitSpec

__all__ = ["ModelState", "ParameterBlock", "Scope"]

#: ``exposure``: one value shared by all orders of a joint fit; ``order``: one per order.
Scope = Literal["exposure", "order"]


@dataclass(frozen=True)
class ModelState:
    """Physical and nuisance values of one order's forward model."""

    scales: Mapping[str, float]
    """Relative column per fitted molecule (in the spec's molecule order)."""
    resolution: float
    """Resolving power R of the LSF."""
    lsf_shape: float | None
    """Shape of a shaped LSF; ``None`` evaluates as 2.0 (the Gaussian limit)."""
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Values of plug-in blocks."""
    wave_shift: tuple[float, ...] | None = None
    """Chebyshev coefficients [km/s] of the telluric-model velocity offset across the
    order; ``None`` = no offset (the model is not resampled)."""

    @classmethod
    def initial(cls, spec: FitSpec) -> ModelState:
        """All values at the spec's initial / fixed values."""
        return cls(
            scales={m: float(spec.relcol_init[m]) for m in spec.molecules},
            resolution=float(spec.resolution_init),
            lsf_shape=spec.lsf_shape,
            wave_shift=spec.wave_shift,
        )


class ParameterBlock(Protocol):
    """One kind of free parameter."""

    name: ClassVar[str]
    """Registry name."""
    scope: ClassVar[Scope]
    """Shared by the orders of a joint fit, or one per order."""

    def active(self, spec: FitSpec) -> bool:
        """Whether the block has free parameters for this spec."""
        ...

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        """Names of the parameters (in a joint fit, order-scope names get the order
        as a prefix, e.g. ``order084_resolution``)."""
        ...

    def initial(self, spec: FitSpec) -> NDArray[np.float64]:
        """Initial values, in ``names`` order."""
        ...

    def bounds(self, spec: FitSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lower and upper bounds, in ``names`` order."""
        ...

    def apply(self, values: NDArray[np.float64], spec: FitSpec, state: ModelState) -> ModelState:
        """Return *state* with this block's values set (``values`` follow ``names``)."""
        ...
