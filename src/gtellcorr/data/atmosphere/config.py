"""Parameter group of the atmospheric-profile construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gtellcorr.config.base import ConfigGroup

__all__ = ["AtmosphereConfig", "LowerBackend"]

#: Lower-atmosphere backends: reanalyses, the local ESO GDAS mirror, or a
#: user-supplied profile file (``manual``, see ``manual_profile_file``).
LowerBackend = Literal[
    "era5_cdsapi", "merra2_gesdisc", "ncep_r1_psl", "gfs_thredds", "eso_gdas", "manual"
]


@dataclass(frozen=True, kw_only=True)
class AtmosphereConfig(ConfigGroup):
    """How the per-exposure 0-120 km atmospheric profile is built."""

    lower_backend: LowerBackend = "era5_cdsapi"
    """Source of the lower atmosphere: a reanalysis, or ``manual`` for a user-supplied
    profile (``manual_profile_file``)."""
    manual_profile_file: str | None = None
    """Lower-atmosphere profile for ``lower_backend = "manual"``: a whitespace-separated
    text file with columns ``height_km pressure_hPa temperature_K h2o_vmr_ppmv``
    (altitude above sea level; ``#`` / ``;`` comments). It is used for every exposure
    of the run and stitched to the upper atmosphere like a reanalysis profile, so it
    must start at or below the site altitude. Required with ``manual``, not allowed
    otherwise."""
    upper_source: Literal["msis00", "mipas"] = "msis00"
    """Model above the reanalysis top: NRLMSISE-00 (T, P) or the MIPAS profile."""
    blend_zone_width_km: float = 6.0
    """Width of the linear lower/upper blend [km]."""
    asm_emix_km: float = 5.0
    """Altitude [km MSL] where the on-site correction tapers off."""
    max_grid_separation_deg: float = 3.0
    """Reject reanalysis grid points farther than this from the site [deg]."""
    merge_onsite: bool = False
    """Blend the on-site header weather (T, RH, P) into the lowest layers. Off by
    default: the header values are passed on but not merged."""

    def validate(self) -> None:
        """Positive widths and separation; a profile file exactly with ``manual``."""
        for name in ("blend_zone_width_km", "asm_emix_km", "max_grid_separation_deg"):
            if getattr(self, name) <= 0:
                self.fail(name, f"must be > 0, got {getattr(self, name)}")
        if self.lower_backend == "manual" and not self.manual_profile_file:
            self.fail("manual_profile_file", 'is required with lower_backend = "manual"')
        if self.lower_backend != "manual" and self.manual_profile_file is not None:
            self.fail(
                "manual_profile_file",
                f'applies only to lower_backend = "manual" (got {self.lower_backend!r})',
            )
