"""``mask`` treatment: the telluric model is ``C * T``; stellar lines are excluded."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from gtellcorr.preprocessing.prepare import StellarProducts, prepare_stellar
from gtellcorr.preprocessing.stellar_mask import fit_exclusion
from gtellcorr.preprocessing.treatments.base import OrderStellarModel

if TYPE_CHECKING:
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.observation import Observation
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.settings import Settings

__all__ = ["MaskTreatment"]


class MaskTreatment:
    """Exclude the RV-shifted stellar lines, activity windows, bad regions and strong
    telluric cores; no stellar model."""

    name: ClassVar[str] = "mask"

    def prepare(
        self,
        observation: Observation,
        config: FitConfig,
        settings: Settings,
        layout: RunLayout,
        **options: Any,
    ) -> StellarProducts:
        """Template, RV table and per-exposure stellar-line masks
        (:func:`~gtellcorr.preprocessing.prepare.prepare_stellar`)."""
        return prepare_stellar(
            observation, config.replace(stellar="mask"), settings, layout, **options
        )

    def order_model(
        self, products: StellarProducts, exposure: str, wave_nm: Any
    ) -> OrderStellarModel:
        """No stellar model; the exposure's line mask joins the exclusions."""
        excl = fit_exclusion(
            wave_nm,
            stellar_mask_file=products.layout.exposure_linemask(exposure),
            activity_windows=products.activity_windows,
            bad_regions=products.bad_regions,
            strong_windows=products.strong_windows,
        )
        return None, excl
