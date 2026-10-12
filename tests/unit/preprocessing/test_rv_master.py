from __future__ import annotations

import csv
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.target import Transit
from gtellcorr.errors import DataError
from gtellcorr.fitting.config import Molecules
from gtellcorr.preprocessing import rv
from gtellcorr.preprocessing.config import MasterContinuum, RVConfig
from gtellcorr.preprocessing.continuum import normalise_continuum
from gtellcorr.preprocessing.master_spectrum import (
    in_transit,
    ivar_coadd,
    load_master_spectrum,
    save_master_spectrum,
    shift_to_rest,
)
from gtellcorr.preprocessing.windows import C_KMS

TW = np.linspace(599.0, 611.0, 24001)
TF = 1.0 - sum(0.6 * np.exp(-0.5 * ((TW - c) / 0.008) ** 2) for c in np.linspace(600.2, 609.8, 40))


def _order(rv_kms: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.linspace(600.0, 610.0, 8000)
    f = np.interp(w / (1 + rv_kms / C_KMS), TW, TF) * (1.0 + 0.02 * (w - 605.0))
    f = f + np.random.default_rng(seed).normal(0.0, 0.002, w.size)
    return w, f, np.full_like(w, 0.002)


def test_ccf_and_rv_recovery() -> None:
    orders = [(o, *_order(-12.5, o)) for o in range(3)]
    result = rv.measure_rv_exposure(
        orders, TW, TF, mask_windows=[], v_lo=-60, v_hi=60, dv=1.0, n_steps=300, n_burn=100, seed=3
    )
    assert result.rv_kms == pytest.approx(-12.5, abs=0.05)
    assert result.n_orders_used == 3 and 0 < result.acceptance < 1
    again = rv.measure_rv_exposure(
        orders, TW, TF, mask_windows=[], v_lo=-60, v_hi=60, dv=1.0, n_steps=300, n_burn=100, seed=3
    )
    assert again.rv_kms == result.rv_kms  # seeded: reproducible
    quick = rv.measure_rv_exposure(
        orders, TW, TF, mask_windows=[], v_lo=-60, v_hi=60, dv=1.0, mcmc=False
    )
    assert quick.rv_kms == quick.rv_master_ccf_kms
    with pytest.raises(DataError, match="no order produced"):
        rv.measure_rv_exposure(orders, TW, TF, mask_windows=[(599.0, 611.0)])
    assert rv.gaussian_fit_ccf(np.arange(5.0), -np.ones(5)) is None


def test_ccf_product_and_tables(tmp_path: Path) -> None:
    orders = [(o, *_order(3.0, o)) for o in range(2)]
    result = rv.measure_rv_exposure(
        orders, TW, TF, mask_windows=[], v_lo=-30, v_hi=30, dv=1.0, mcmc=False
    )
    path = rv.save_ccf_data(
        result, tmp_path / "x" / "ccf.npz", star_name="S", instrument="I", obs_time="T"
    )
    data = np.load(path)
    assert data["orders"].tolist() == [0, 1] and float(data["rv_kms"]) == result.rv_kms
    with pytest.raises(ValueError, match="CCF arrays"):
        rv.save_ccf_data(
            dataclasses.replace(result, master_ccf=None),
            tmp_path / "y.npz",
            star_name="S",
            instrument="I",
            obs_time="T",
        )
    table = tmp_path / "rv.csv"
    with open(table, "w", newline="") as fh:
        csv.writer(fh).writerows([rv.RV_TABLE_COLUMNS, ["a.fits", *[1] * 11]])
    assert rv.read_rv_table(table)[0]["file"] == "a.fits"
    assert rv.rv_table_covers(table, ["a.fits"]) and not rv.rv_table_covers(table, ["b.fits"])
    assert not rv.rv_table_covers(tmp_path / "none.csv", ["a"])
    empty = tmp_path / "empty.csv"
    empty.write_text("file\n")
    with pytest.raises(DataError, match="empty RV table"):
        rv.read_rv_table(empty)


def test_rv_helpers(tmp_path: Path) -> None:
    user = tmp_path / "user.txt"
    user.write_text("501 500\n")
    cfg = RVConfig(
        telluric_mask_molecules=("o2",),
        mask_activity=True,
        pad_kms=0.0,
        user_mask_nm=((1.0, 2.0),),
        user_mask_file=str(user),
    )
    windows = rv.rv_mask_windows(cfg, Molecules.default(), [(393.0, 394.0)])
    assert windows[: len(Molecules.default()["o2"].bands_nm)] == [
        tuple(map(float, b)) for b in Molecules.default()["o2"].bands_nm
    ]
    assert windows[-3:] == [(393.0, 394.0), (1.0, 2.0), (500.0, 501.0)]
    # BJD_TDB from the card each instrument declares
    mid = datetime(2025, 1, 1, 3, 5, 0, tzinfo=UTC)  # JD 2460676.628472 (UTC)
    for name, card in (("NEID", "SSBJD100"), ("PEPSI", "JD-TDB"), ("ESPRESSO", "ESO QC BJD")):
        schema = instruments.get(name).header
        assert rv.exposure_bjd({card: 2460676.6304}, schema, mid) == (2460676.6304, True)
        # without the card: the mid-exposure JD (UTC), flagged as not found
        jd, found = rv.exposure_bjd({"JD-OBS": 1.0}, schema, mid)
        assert not found and jd == pytest.approx(2460676.5 + (3 + 5 / 60) / 24, abs=1e-9)
    assert rv.exposure_bjd({}, None, mid)[1] is False
    assert np.isnan(rv.exposure_bjd({}, None, None)[0])
    flat = rv.normalize_continuum(np.arange(3.0), np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(flat, 1.0)


def test_master_helpers(tmp_path: Path) -> None:
    assert shift_to_rest([600.0], 0.0)[0] == 600.0
    transit = Transit(t0_bjd=100.0, period_d=10.0, duration_d=0.2)
    assert in_transit(110.05, transit) and not in_transit(110.2, transit)
    assert not in_transit(100.0, None) and not in_transit(100.0, Transit())
    w = np.linspace(1.0, 2.0, 11)
    grid_w, f, e, n = ivar_coadd(
        [(w, np.ones_like(w), np.full_like(w, 0.1)), (w, 3 * np.ones_like(w), np.full_like(w, 0.1))]
    )
    np.testing.assert_allclose(f, 2.0)
    np.testing.assert_allclose(e, 0.1 / np.sqrt(2))
    assert n.tolist() == [2] * grid_w.size
    assert ivar_coadd([])[0].size == 0
    orders = {
        5: {
            "wave_nm": w,
            "flux": np.ones_like(w),
            "err": np.ones_like(w),
            "n": np.ones(w.size, int),
            "flux_raw": w,
            "cont": w,
        }
    }
    save_master_spectrum(orders, tmp_path / "m.npz", tmp_path / "m.dat")
    back = load_master_spectrum(tmp_path / "m.npz")
    np.testing.assert_array_equal(back[5]["flux_raw"], w)
    assert (tmp_path / "m.dat").read_text().startswith("# wave_nm_rest")


def test_poly_continuum_and_ispec_fallback(caplog: pytest.LogCaptureFixture) -> None:
    w = np.linspace(600.0, 610.0, 500)
    f = 2.0 + 0.1 * (w - 605.0)
    _, norm = normalise_continuum(w, f, config=MasterContinuum(method="poly"))
    np.testing.assert_allclose(norm, 1.0, rtol=1e-8)
    _, norm = normalise_continuum(w, f, config=MasterContinuum(method="ispec"), ispec_dir=None)
    np.testing.assert_allclose(norm, 1.0, rtol=1e-8)
    assert "using the polynomial" in caplog.text


def test_rv_table_warns_without_bjd(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="gtellcorr.preprocessing.rv"):
        rv._warn_without_bjd([], "LBT/PEPSI")
        assert caplog.text == ""
        rv._warn_without_bjd(["a.fits", "b.fits"], "LBT/PEPSI")
    assert "2 exposure(s) without a BJD_TDB header card" in caplog.text
    assert "JD (UTC) of their epoch" in caplog.text and "a.fits, b.fits" in caplog.text
