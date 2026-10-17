"""A stellar model the telluric fit can move by a velocity.

The ``template`` treatment places the stellar template on an order by the exposure's
measured RV. That RV carries the template's own systematic error (line-by-line
differences between the template and the star shift a CCF by tens of m/s), and a
misplaced template drags the fitted wavelength-solution correction with it. With
``stellar_model.fit_shift`` the fit adds a velocity ``stellar_shift`` [km/s] to the
placement RV (one per exposure in a joint fit): the template is placed at
``RV + stellar_shift``. The telluric transmission is not moved by it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ShiftableStellarModel", "stellar_on_data"]

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ShiftableStellarModel:
    """A stellar model on an order's wavelengths, re-placeable by a velocity.

    ``place(shift_kms)`` returns the model on the order's wavelengths with the
    template moved by *shift_kms* beyond its placement; ``place(0.0)`` is the model
    as placed by the measured RV. As an array it is ``place(0.0)``.
    """

    place: Callable[[float], Array]

    def at(self, shift_kms: float) -> Array:
        """The model with the template moved by *shift_kms* [km/s]."""
        return np.asarray(self.place(float(shift_kms)), float)

    def __array__(self, dtype: Any = None, copy: Any = None) -> Array:
        """The model as placed (no shift)."""
        values = self.at(0.0)
        return values if dtype is None else values.astype(dtype)


def stellar_on_data(stellar_model: Any | None, shift_kms: float | None) -> Array | None:
    """The stellar model array of an evaluation: ``None`` without a model; the model
    as given without a shift; moved by *shift_kms* otherwise (it must then be a
    :class:`ShiftableStellarModel`)."""
    if stellar_model is None:
        return None
    if shift_kms is None:
        return np.asarray(stellar_model, float)
    if not isinstance(stellar_model, ShiftableStellarModel):
        raise ValueError("a stellar shift needs a ShiftableStellarModel, not a fixed array")
    return stellar_model.at(shift_kms)
