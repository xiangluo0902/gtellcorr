"""``broaden_template`` against the exact result: Gaussian lines convolved with a Gaussian."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import curve_fit

from gtellcorr.constants import C_KMS
from gtellcorr.preprocessing.synthesis import broaden_template

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
CENTRES = np.linspace(627.5, 631.5, 5)
S0 = 4.0 / FWHM  # intrinsic line width [km/s]


def _lines(wave: Any, sigma_kms: float) -> Any:
    """Lines Gaussian in velocity with the equivalent width of depth 0.5 at ``S0``."""
    flux = np.ones_like(wave)
    for c in CENTRES:
        v = (wave / c - 1.0) * C_KMS
        flux -= 0.5 * S0 / sigma_kms * np.exp(-0.5 * (v / sigma_kms) ** 2)
    return flux


def _sigma_and_shift(wave: Any, flux: Any, centre: float) -> tuple[float, float]:
    """Fitted width [km/s] and centre shift [m/s] of one line."""
    v = (wave / centre - 1.0) * C_KMS
    sel = np.abs(v) < 40.0
    (_, mu, s), _ = curve_fit(
        lambda x, a, mu, s: 1.0 - a * np.exp(-0.5 * ((x - mu) / s) ** 2),
        v[sel], flux[sel], p0=(0.3, 0.0, 3.0),
    )  # fmt: skip
    return abs(float(s)), float(mu) * 1000.0


@pytest.mark.parametrize("resolution", [130_000.0, 110_000.0, 50_000.0])
@pytest.mark.parametrize("r_from", [np.inf, 500_000.0])
@pytest.mark.parametrize("grid", ["uniform", "log", "irregular"])
def test_matches_the_exact_convolution(resolution: float, r_from: float, grid: str) -> None:
    lo, hi = 626.0, 633.0
    if grid == "uniform":
        wave = np.arange(lo, hi, 0.0005)
    elif grid == "log":
        wave = np.exp(np.linspace(np.log(lo), np.log(hi), 14_000))
    else:
        rng = np.random.default_rng(1)
        wave = np.sort(rng.uniform(lo, hi, 14_000))
    s_from = 0.0 if np.isinf(r_from) else C_KMS / r_from / FWHM
    flux = _lines(wave, np.hypot(S0, s_from))
    s_true = np.hypot(S0, C_KMS / resolution / FWHM)

    broad = broaden_template(wave, flux, from_resolution=r_from, to_resolution=resolution)

    core = (wave > 627.0) & (wave < 632.0)
    # a random grid has gaps of several pixels next to some lines, which limit any
    # interpolation and make single lines appear shifted by ~1 m/s
    tolerance, max_shift = (5e-4, 2.0) if grid == "irregular" else (1e-4, 0.1)
    assert np.max(np.abs(broad - _lines(wave, s_true))[core]) < tolerance
    for centre in CENTRES:
        sigma, shift = _sigma_and_shift(wave, broad, centre)
        r_eff = C_KMS / (FWHM * np.sqrt(sigma**2 - S0**2))
        assert r_eff == pytest.approx(resolution, rel=2e-3)  # constant across the window
        assert abs(shift) < max_shift  # m/s


def test_rejects_sharpening() -> None:
    wave = np.linspace(600.0, 601.0, 1000)
    with pytest.raises(Exception, match="must exceed"):
        broaden_template(wave, np.ones_like(wave), from_resolution=50_000, to_resolution=60_000)


def test_near_duplicate_points_do_not_ring() -> None:
    # stitched chunks can hold two points 1e-6 nm apart with slightly different flux
    wave = np.arange(626.0, 633.0, 0.0005)
    flux = _lines(wave, S0)
    i = 7000
    joined_w = np.insert(wave, i + 1, wave[i] + 1e-6)
    joined_f = np.insert(flux, i + 1, flux[i] + 1e-3)
    plain = broaden_template(wave, flux, from_resolution=np.inf, to_resolution=130_000.0)
    joined = broaden_template(joined_w, joined_f, from_resolution=np.inf, to_resolution=130_000.0)
    # averaging the pair changes one sample by 5e-4; nothing beyond that propagates
    assert np.max(np.abs(np.delete(joined, i + 1) - plain)) < 1e-4


def test_synthesised_templates_start_from_the_synthesis_resolution(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gtellcorr.preprocessing import synthesis
    from gtellcorr.preprocessing.synthesis import StellarParameters, Synthesis

    star = StellarParameters(name="s", teff=5000.0, logg=4.5, feh=0.0, vsini_kms=2.0)
    pysme = Synthesis(star=star, backend="pysme", linelist="ll", ispec_dir=None)
    ispec = Synthesis(star=star, backend="ispec", linelist=None, ispec_dir="i",
                      ispec={"resolution": 500_000})  # fmt: skip
    raw = Synthesis(star=star, backend="ispec", linelist=None, ispec_dir="i",
                    ispec={"resolution": 0})  # fmt: skip
    assert synthesis.synthesis_resolution(pysme) == np.inf
    assert synthesis.synthesis_resolution(ispec) == 500_000.0
    assert synthesis.synthesis_resolution(raw) == np.inf

    # the chunk worker broadens from that resolution, without iSpec
    wave = np.arange(626.0, 633.0, 0.0005)
    s_from = C_KMS / 500_000.0 / FWHM
    monkeypatch.setattr(
        synthesis, "stellar_template", lambda *a, **k: (wave, _lines(wave, np.hypot(S0, s_from)))
    )
    out = tmp_path / "chunk.dat"
    synthesis._synth_chunk((ispec, 626, 633, None, 130_000.0, str(out)))
    got = np.loadtxt(out)
    exact = _lines(wave, np.hypot(S0, C_KMS / 130_000.0 / FWHM))
    core = (wave > 627.0) & (wave < 632.0)
    # the method's accuracy at R = 130000 (~3e-5); broadening from infinite R instead
    # of 500000 would be ~3 % too broad and miss by ~1e-3
    assert np.max(np.abs(got[:, 1] - exact)[core]) < 5e-5
