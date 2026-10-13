"""Read-only results of the LBLRTM transmission quality check."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lblrtm_errors import LblrtmInputError
from .transmission_quality import (
    TransmissionQualitySummary,
    build_transmission_fit_mask,
    classify_transmission_values,
    summarize_transmission_quality,
)

__all__ = [
    "LblrtmAuditedTransmissionSpectrum",
    "TelluricTransmissionQualityResult",
]


def _immutable_array(value: object, *, name: str, dtype: object) -> np.ndarray:
    """A read-only C-ordered copy of *value* as *dtype*."""
    try:
        array = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise LblrtmInputError(f"{name} cannot be represented as an array") from exc
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TelluricTransmissionQualityResult:
    """Raw transmission plus non-destructive quality flags and fit mask."""

    wavelength_nm: np.ndarray
    transmission: np.ndarray
    wavelength_medium: str
    quality_flags: np.ndarray | None = None
    fit_usable: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = _immutable_array(self.wavelength_nm, name="wavelength_nm", dtype=np.float64)
        transmission = _immutable_array(self.transmission, name="transmission", dtype=np.float64)
        if wavelength.ndim != 1 or transmission.ndim != 1:
            raise LblrtmInputError("quality-result arrays must be one-dimensional")
        if wavelength.size < 2 or wavelength.shape != transmission.shape:
            raise LblrtmInputError("quality-result arrays have invalid shapes")
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise LblrtmInputError("quality-result wavelengths must be finite and positive")
        if np.any(np.diff(wavelength) <= 0.0):
            raise LblrtmInputError("quality-result wavelengths must be strictly increasing")
        if not isinstance(self.wavelength_medium, str):
            raise LblrtmInputError("wavelength_medium must be a string")
        flags = (
            classify_transmission_values(transmission)
            if self.quality_flags is None
            else _immutable_array(
                self.quality_flags,
                name="quality_flags",
                dtype=np.uint8,
            )
        )
        usable = (
            build_transmission_fit_mask(transmission, flags)
            if self.fit_usable is None
            else _immutable_array(
                self.fit_usable,
                name="fit_usable",
                dtype=bool,
            )
        )
        if flags.shape != transmission.shape or usable.shape != transmission.shape:
            raise LblrtmInputError("quality-result arrays must match transmission")
        if np.any(usable & ~build_transmission_fit_mask(transmission, flags)):
            raise LblrtmInputError("fit_usable cannot enable a nonphysical pixel")
        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "quality_flags", flags)
        object.__setattr__(self, "fit_usable", usable)

    @property
    def summary(self) -> TransmissionQualitySummary:
        """Return the quality summary of this spectrum."""

        return summarize_transmission_quality(
            self.transmission,
            self.quality_flags,
        )

    def has_nonphysical_values(self) -> bool:
        """Return whether any sample is excluded from fitting."""

        return bool(np.any(~self.fit_usable))


# Compatibility name retained for existing diagnostic users.  New code should
# use the backend-independent TelluricTransmissionQualityResult name.
LblrtmAuditedTransmissionSpectrum = TelluricTransmissionQualityResult
