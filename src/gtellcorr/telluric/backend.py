"""Radiative-transfer backends (plugin registry ``gtellcorr.rt_backends``).

A backend turns a complete atmospheric profile and a line of sight into an
unconvolved transmission or optical-depth spectrum. LBLRTM is the only built-in
backend; the protocol exists so another RT code can be plugged in later without
touching the fitting code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import ArrayLike

from gtellcorr.registry import Registry

__all__ = ["LblrtmBackend", "RTBackend", "rt_backends"]


class RTBackend(Protocol):
    """What the telluric model needs from a radiative-transfer code."""

    def transmission(
        self,
        profile: Mapping[str, Any],
        wavelength_range_nm: ArrayLike,
        *,
        wavelength_medium: str,
        formula: str,
        line_of_sight: Any,
        relative_columns: Mapping[str, float] | None = None,
        wave_margin_nm: float = 0.0,
    ) -> Any:
        """Unconvolved transmission of *profile* over *wavelength_range_nm*."""
        ...

    def optical_depth(
        self,
        profile: Mapping[str, Any],
        wavelength_range_nm: ArrayLike,
        *,
        wavelength_medium: str,
        formula: str,
        line_of_sight: Any,
        relative_columns: Mapping[str, float] | None = None,
        wave_margin_nm: float = 0.0,
    ) -> Any:
        """Optical depth of *profile* over *wavelength_range_nm*."""
        ...


class LblrtmBackend:
    """LBLRTM v12.17 through the calculators of :mod:`gtellcorr.telluric.lblrtm`.

    ``resources`` are the verified bundle files
    (:func:`gtellcorr.telluric.resources.verify`).
    """

    def __init__(self, resources: Any, *, timeout_s: float | None = None) -> None:
        self.resources = resources
        self.timeout_s = timeout_s

    def transmission(
        self,
        profile: Mapping[str, Any],
        wavelength_range_nm: ArrayLike,
        *,
        wavelength_medium: str,
        formula: str,
        line_of_sight: Any,
        relative_columns: Mapping[str, float] | None = None,
        wave_margin_nm: float = 0.0,
    ) -> Any:
        """LBLRTM transmission (``quality_mode="evaluate"``)."""
        from gtellcorr.telluric.lblrtm.transmission_calculator import (
            calculate_lblrtm_transmission,
        )

        return calculate_lblrtm_transmission(
            profile,
            wavelength_range_nm,
            wavelength_medium=wavelength_medium,
            wavelength_conversion_formula=formula,
            line_of_sight=line_of_sight,
            molecule_relative_columns=relative_columns,
            resources=self.resources,
            wave_margin_nm=wave_margin_nm,
            timeout_s=self.timeout_s,
            quality_mode="evaluate",
        )

    def optical_depth(
        self,
        profile: Mapping[str, Any],
        wavelength_range_nm: ArrayLike,
        *,
        wavelength_medium: str,
        formula: str,
        line_of_sight: Any,
        relative_columns: Mapping[str, float] | None = None,
        wave_margin_nm: float = 0.0,
    ) -> Any:
        """LBLRTM optical depth."""
        from gtellcorr.telluric.lblrtm.optical_depth_calculator import (
            calculate_lblrtm_optical_depth,
        )

        return calculate_lblrtm_optical_depth(
            profile,
            wavelength_range_nm,
            wavelength_medium=wavelength_medium,
            wavelength_conversion_formula=formula,
            line_of_sight=line_of_sight,
            molecule_relative_columns=relative_columns,
            resources=self.resources,
            wave_margin_nm=wave_margin_nm,
            timeout_s=self.timeout_s,
        )


#: Registered backend *classes*, instantiated with verified resources.
rt_backends: Registry[type] = Registry(
    "radiative-transfer backend", entry_point_group="gtellcorr.rt_backends"
)
rt_backends.register("lblrtm", LblrtmBackend)
