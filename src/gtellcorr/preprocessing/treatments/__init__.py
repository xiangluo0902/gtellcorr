"""Stellar-line treatments: how the star enters the telluric fit.

``mask`` excludes the stellar lines; ``template`` models them with a stellar
template. Further treatments can be registered under the entry-point group
``gtellcorr.stellar_treatments``.
"""

from __future__ import annotations

from gtellcorr.preprocessing.treatments.base import OrderStellarModel, StellarTreatment
from gtellcorr.preprocessing.treatments.mask import MaskTreatment
from gtellcorr.preprocessing.treatments.template import TemplateTreatment
from gtellcorr.registry import Registry

__all__ = [
    "MaskTreatment",
    "OrderStellarModel",
    "StellarTreatment",
    "TemplateTreatment",
    "stellar_treatments",
]

#: Registered treatment classes (``FitConfig.stellar`` names one).
stellar_treatments: Registry[type] = Registry(
    "stellar treatment", entry_point_group="gtellcorr.stellar_treatments"
)
stellar_treatments.register("mask", MaskTreatment)
stellar_treatments.register("template", TemplateTreatment)
