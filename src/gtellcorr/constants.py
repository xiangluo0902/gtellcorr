"""Physical constants: the single source of truth for the whole package.

Import from here instead of re-declaring a constant locally.
"""

from typing import Final

__all__ = ["C_KMS", "G_EARTH", "M_DRY", "M_H2O", "R_DRY_AIR", "R_UNIV"]

C_KMS: Final = 299792.458  # speed of light [km/s]
G_EARTH: Final = 9.80665  # standard gravity [m/s^2]
R_UNIV: Final = 8.31446  # universal gas constant [J/(mol K)]
R_DRY_AIR: Final = 287.058  # specific gas constant, dry air [J/(kg K)]
M_H2O: Final = 0.0180153  # molar mass, water [kg/mol]
M_DRY: Final = 0.0289644  # molar mass, dry air [kg/mol]
