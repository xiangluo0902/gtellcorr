"""Preparation before the fit: stellar template, radial velocities, stellar-line
masks, strong-telluric mask and bad regions."""

from __future__ import annotations

from gtellcorr.preprocessing.config import (
    FitMasks,
    IspecSettings,
    MasterContinuum,
    MasterSpectrum,
    PysmeSettings,
    RVConfig,
    StellarLineMask,
    StellarModelConfig,
    StrongTelluricMask,
)
from gtellcorr.preprocessing.prepare import StellarProducts, prepare_stellar
from gtellcorr.preprocessing.treatments import (
    MaskTreatment,
    StellarTreatment,
    TemplateTreatment,
    stellar_treatments,
)

__all__ = [
    "FitMasks",
    "IspecSettings",
    "MaskTreatment",
    "MasterContinuum",
    "MasterSpectrum",
    "PysmeSettings",
    "RVConfig",
    "StellarLineMask",
    "StellarModelConfig",
    "StellarProducts",
    "StellarTreatment",
    "StrongTelluricMask",
    "TemplateTreatment",
    "prepare_stellar",
    "stellar_treatments",
]
