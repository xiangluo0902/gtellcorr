"""Parameter group of the telluric forward model."""

from __future__ import annotations

from dataclasses import dataclass

from gtellcorr.config.base import ConfigGroup

__all__ = ["TelluricConfig"]


@dataclass(frozen=True, kw_only=True)
class TelluricConfig(ConfigGroup):
    """Telluric forward-model settings."""

    wave_margin_nm: float = 20.0
    """Padding [nm] added on each side of an order before the RT calculation, so
    line wings just outside the order are modelled."""

    def validate(self) -> None:
        """Non-negative margin."""
        if self.wave_margin_nm < 0:
            self.fail("wave_margin_nm", f"must be >= 0, got {self.wave_margin_nm}")
