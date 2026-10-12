"""``template`` treatment: the model is ``C * M * T`` with the RV-shifted stellar template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from gtellcorr.preprocessing.prepare import StellarProducts, prepare_stellar
from gtellcorr.preprocessing.stellar_mask import fit_exclusion
from gtellcorr.preprocessing.synthesis import stellar_model_for_order
from gtellcorr.preprocessing.treatments.base import OrderStellarModel

if TYPE_CHECKING:
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.observation import Observation
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.settings import Settings

__all__ = ["TemplateTreatment"]


class TemplateTreatment:
    """Model the star with the template shifted by the exposure's RV (not shifted for a
    ``topocentric`` user template); exclude activity windows, bad regions and strong
    telluric cores."""

    name: ClassVar[str] = "template"

    def prepare(
        self,
        observation: Observation,
        config: FitConfig,
        settings: Settings,
        layout: RunLayout,
        **options: Any,
    ) -> StellarProducts:
        """Template and RV table
        (:func:`~gtellcorr.preprocessing.prepare.prepare_stellar`)."""
        return prepare_stellar(
            observation, config.replace(stellar="template"), settings, layout, **options
        )

    def order_model(
        self, products: StellarProducts, exposure: str, wave_nm: Any
    ) -> OrderStellarModel:
        """The template placed on *wave_nm* (1 outside it) and the exclusions."""
        rv = products.placement_rv(exposure)
        excl = fit_exclusion(
            wave_nm,
            activity_windows=products.activity_windows,
            bad_regions=products.bad_regions,
            strong_windows=products.strong_windows,
        )
        return stellar_model_for_order(wave_nm, rv, products.template), excl
