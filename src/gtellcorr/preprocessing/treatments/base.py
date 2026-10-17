"""StellarTreatment protocol: how the star enters the telluric fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from gtellcorr.fitting.stellar_shift import ShiftableStellarModel

if TYPE_CHECKING:
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.observation import Observation
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.preprocessing.prepare import StellarProducts
    from gtellcorr.settings import Settings

__all__ = ["OrderStellarModel", "StellarTreatment"]

OrderStellarModel = tuple[NDArray[np.float64] | ShiftableStellarModel | None, NDArray[np.bool_]]
"""``(stellar model M on the order grid or None, exclusion mask with True = exclude)``;
a :class:`~gtellcorr.fitting.stellar_shift.ShiftableStellarModel` lets the fit move
the template by a velocity."""


class StellarTreatment(Protocol):
    """A stellar-line treatment (registry ``gtellcorr.stellar_treatments``)."""

    name: ClassVar[str]

    def prepare(
        self,
        observation: Observation,
        config: FitConfig,
        settings: Settings,
        layout: RunLayout,
        **options: Any,
    ) -> StellarProducts:
        """Products needed for the run (generated or reused)."""
        ...

    def order_model(
        self, products: StellarProducts, exposure: str, wave_nm: Any
    ) -> OrderStellarModel:
        """Stellar model and exclusion mask of one order of one exposure."""
        ...
