from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from gtellcorr.constants import C_KMS
from gtellcorr.data.observation.frames import (
    FrameSpec,
    air_to_vac,
    from_air_topocentric,
    n_air,
    n_air_ciddor1996,
    n_air_edlen1966,
    n_air_espresso,
    to_air_topocentric,
    vac_to_air,
)
from gtellcorr.errors import ConfigError, DataError

WAVE = np.array([400.0, 656.28, 760.0, 1000.0])


def test_refractive_index_values() -> None:
    # Standard air at 600 nm: n - 1 ~ 2.77e-4 for all three formulas.
    for func in (n_air_espresso, n_air_edlen1966, n_air_ciddor1996):
        n = func(np.array([600.0]))[0]
        assert 2.75e-4 < n - 1 < 2.79e-4
    # Formulas agree to better than 1e-7 in the optical (< 30 m/s).
    assert np.max(np.abs(n_air_espresso(WAVE) - n_air_ciddor1996(WAVE))) < 1e-7
    assert np.array_equal(n_air(WAVE, "edlen1966"), n_air_edlen1966(WAVE))
    with pytest.raises(DataError, match="unknown vacuum/air formula"):
        n_air(WAVE, "bad")  # type: ignore[arg-type]


def test_vacuum_air_round_trip_error() -> None:
    # The one-step air->vacuum inverse (reference behaviour) leaves O((n-1)^2):
    # ~0.5 m/s at 720 nm, ~2 m/s at 400 nm.
    back = air_to_vac(vac_to_air(WAVE))
    err_ms = np.abs(back / WAVE - 1) * C_KMS * 1000
    assert err_ms[WAVE >= 650].max() < 0.7
    assert err_ms.max() < 2.5


def test_frame_spec_validation() -> None:
    with pytest.raises(ConfigError, match="berv_kms: required"):
        FrameSpec(wavelength_medium="air", berv_corrected=True)
    with pytest.raises(ConfigError, match="vsys_kms: required"):
        FrameSpec(wavelength_medium="air", vsys_corrected=True)


def test_vacuum_topocentric_is_plain_vac_to_air() -> None:
    frame = FrameSpec(wavelength_medium="vacuum")
    assert np.array_equal(to_air_topocentric(WAVE, frame), vac_to_air(WAVE))


def test_berv_and_vsys_are_undone_and_redone() -> None:
    frame = FrameSpec(
        wavelength_medium="air",
        berv_corrected=True,
        berv_kms="BERV",
        vsys_corrected=True,
        vsys_kms=-12.5,
    )
    header = fits.Header({"BERV": 20.0})
    topo = to_air_topocentric(WAVE, frame, header)
    expected = vac_to_air(air_to_vac(WAVE) * (1 - 12.5 / C_KMS) / (1 + 20.0 / C_KMS))
    assert np.array_equal(topo, expected)
    back = from_air_topocentric(topo, frame, header)
    assert np.max(np.abs(back / WAVE - 1)) * C_KMS * 1000 < 5.0


def test_combined_removed_velocity_ignores_berv_flags() -> None:
    frame = FrameSpec(
        wavelength_medium="air",
        removed_velocity_mps=("SSTVEL", 100.0),
        berv_corrected=True,
        berv_kms=99.0,
    )
    header = fits.Header({"SSTVEL": -25000.0})
    topo = to_air_topocentric(WAVE, frame, header)
    expected = vac_to_air(air_to_vac(WAVE) * (1 + (-25000.0 + 100.0) / 1000.0 / C_KMS))
    assert np.array_equal(topo, expected)
    back = from_air_topocentric(topo, frame, header)
    assert np.max(np.abs(back / WAVE - 1)) * C_KMS * 1000 < 5.0
    vac = FrameSpec(wavelength_medium="vacuum", removed_velocity_mps=(0.0,))
    assert np.allclose(from_air_topocentric(to_air_topocentric(WAVE, vac), vac), WAVE)


def test_formula_is_passed_through() -> None:
    frame = FrameSpec(wavelength_medium="vacuum")
    a = to_air_topocentric(WAVE, frame, formula="espresso")
    b = to_air_topocentric(WAVE, frame, formula="ciddor1996")
    assert not np.array_equal(a, b)


@pytest.mark.parametrize(
    ("header", "message"),
    [(None, "no FITS header supplied"), (fits.Header(), "not in FITS header")],
)
def test_missing_velocity_keyword(header: fits.Header | None, message: str) -> None:
    frame = FrameSpec(wavelength_medium="air", berv_corrected=True, berv_kms="BERV")
    with pytest.raises(DataError, match=message):
        to_air_topocentric(WAVE, frame, header)
