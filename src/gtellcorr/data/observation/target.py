"""The observed target: star and (optional) transiting planet."""

from __future__ import annotations

from dataclasses import dataclass

from gtellcorr.config.base import ConfigGroup

__all__ = ["Star", "Transit"]


@dataclass(frozen=True, kw_only=True)
class Star(ConfigGroup):
    """Stellar parameters. Values needed by a chosen stellar treatment (e.g. the
    template synthesis needs ``teff``, ``logg``, ``feh``, ``vsini_kms``) are
    checked when that treatment runs."""

    name: str
    """Target name, used in output directory and file names."""
    teff: float | None = None
    """Effective temperature [K]."""
    logg: float | None = None
    """Surface gravity [cgs]."""
    feh: float | None = None
    """Metallicity [M/H]."""
    vsini_kms: float | None = None
    """Projected rotational velocity [km/s]."""
    vsys_kms: float | None = None
    """Systemic radial velocity [km/s]."""
    vmic_kms: float | None = None
    """Microturbulence [km/s]; ``None`` = estimated by the synthesis backend."""
    vmac_kms: float | None = None
    """Macroturbulence [km/s]; ``None`` = estimated by the synthesis backend."""
    alpha: float | None = None
    """[alpha/Fe]; ``None`` = derived from [M/H]."""
    limb_darkening_coeff: float = 0.6
    """Linear limb-darkening coefficient of the template synthesis."""

    def validate(self) -> None:
        """Name not empty; temperatures and velocities in range."""
        if not self.name.strip():
            self.fail("name", "must not be empty")
        if self.teff is not None and self.teff <= 0:
            self.fail("teff", f"must be > 0, got {self.teff}")
        for name in ("vsini_kms", "vmic_kms", "vmac_kms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                self.fail(name, f"must be >= 0, got {value}")
        if not 0 <= self.limb_darkening_coeff <= 1:
            self.fail("limb_darkening_coeff", "must be in [0, 1]")


@dataclass(frozen=True, kw_only=True)
class Transit(ConfigGroup):
    """Transit ephemeris, used to drop in-transit exposures from the stellar master."""

    name: str | None = None
    """Planet name (for reference only)."""
    t0_bjd: float | None = None
    """Mid-transit time [BJD_TDB]."""
    period_d: float | None = None
    """Orbital period [d]."""
    duration_d: float | None = None
    """Total transit duration T14 [d]."""

    @property
    def enabled(self) -> bool:
        """Whether an ephemeris is given."""
        return self.t0_bjd is not None

    def validate(self) -> None:
        """The ephemeris is all or nothing, with positive period and duration."""
        given = [v is not None for v in (self.t0_bjd, self.period_d, self.duration_d)]
        if any(given) and not all(given):
            self.fail("t0_bjd", "t0_bjd, period_d and duration_d must be given together")
        if self.period_d is not None and self.period_d <= 0:
            self.fail("period_d", f"must be > 0, got {self.period_d}")
        if self.duration_d is not None and self.duration_d <= 0:
            self.fail("duration_d", f"must be > 0, got {self.duration_d}")
