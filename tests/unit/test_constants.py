from __future__ import annotations

from gtellcorr import constants


def test_physical_constants() -> None:
    assert constants.C_KMS == 299792.458
    assert constants.G_EARTH == 9.80665
    assert constants.R_UNIV == 8.31446
    assert constants.R_DRY_AIR == 287.058
    assert constants.M_H2O == 0.0180153
    assert constants.M_DRY == 0.0289644
