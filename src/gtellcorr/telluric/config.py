"""Parameter group of the telluric forward model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gtellcorr.config.base import ConfigGroup

__all__ = ["TelluricConfig"]

ContinuumMode = Literal["full", "scaled", "none"]
_CONTINUUM_MODES = ("full", "scaled", "none")


@dataclass(frozen=True, kw_only=True)
class TelluricConfig(ConfigGroup):
    """Telluric forward-model settings."""

    wave_margin_nm: float = 20.0
    """Padding [nm] added on each side of an order before the RT calculation, so
    line wings just outside the order are modelled."""
    continuum_mode: ContinuumMode = "full"
    """LBLRTM continua in the optical-depth cache. ``full``: the H2O / O2 / N2
    continua and collision-induced absorption, Rayleigh extinction and the O3 cross
    sections, computed at the reference and fixed columns and held fixed while the
    fitted columns are scaled (e.g. the O2 collision-induced absorption under the
    O2 A band). ``scaled``: the same continua, following the fitted columns (a
    quadratic in each fitted column, exact for each molecule on its own: e.g. the
    H2O self continuum and the O2 collision-induced absorption go as the square of
    the column); four more LBLRTM runs per fitted molecule when the cache is built.
    ``none``: line absorption only, as gtellcorr before 0.9.2."""

    def validate(self) -> None:
        """Non-negative margin, known continuum mode."""
        if self.wave_margin_nm < 0:
            self.fail("wave_margin_nm", f"must be >= 0, got {self.wave_margin_nm}")
        if self.continuum_mode not in _CONTINUUM_MODES:
            self.fail(
                "continuum_mode", f"must be one of {_CONTINUUM_MODES}, got {self.continuum_mode!r}"
            )
