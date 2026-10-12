"""Molecule names known to the telluric model, in LBLRTM ID order."""

from __future__ import annotations

from typing import Final

__all__ = ["ENGINE_MOLECULES", "MOLECULE_NAMES"]

#: Molecules represented by the complete production atmosphere (LBLRTM IDs 1-12);
#: only these can be scaled (fitted) by the current optical-depth cache.
ENGINE_MOLECULES: Final[tuple[str, ...]] = (
    "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "no", "so2", "no2", "nh3", "hno3",
)  # fmt: skip

#: All 39 molecule names of the LBLRTM line-list registry, in LBLRTM ID order.
#: IDs 13-39 are kept for configuration completeness but cannot be fitted yet.
MOLECULE_NAMES: Final[tuple[str, ...]] = (
    *ENGINE_MOLECULES,
    "oh", "hf", "hcl", "hbr", "hi", "clo", "ocs", "h2co", "hocl", "n2", "hcn", "ch3cl",
    "h2o2", "c2h2", "c2h6", "ph3", "cof2", "sf6", "h2s", "hcooh", "ho2", "o", "clono2",
    "no+", "hobr", "c2h4", "ch3oh",
)  # fmt: skip
