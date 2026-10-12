from __future__ import annotations

import csv
import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import gtellcorr as gtc
from gtellcorr.data.products.fits_schema import (
    CORRECTED_WAVE_HDU_NAMES,
    S2D_IMAGE_HDU_NAMES,
    discover_result_orders,
    load_exposure_order,
    read_order_timeseries,
    read_provenance,
    update_order_columns,
    write_exposure_fits,
)
from gtellcorr.data.products.pca import compute_pca_model_maps, compute_telluric_pca_for_run
from gtellcorr.data.products.result import ExposureResult, load_result, summary_rows_from_fits
from gtellcorr.data.products.summary import SUMMARY_COLUMNS, joint_row, single_row, skipped_row
from gtellcorr.fitting.results import OrderFitResult
from gtellcorr.fitting.wave_shift import C_KMS, shifted_grid

T0 = datetime(2022, 2, 13, 7, 35, tzinfo=UTC)


def _result(order: int, n: int, depth: float = 0.5, status: str = "joint_ok") -> OrderFitResult:
    w = np.linspace(600.0 + order, 601.0 + order, n)
    t = 1.0 - depth * np.exp(-0.5 * ((w - w.mean()) / 0.05) ** 2)
    return OrderFitResult(
        order=order, wave_input=w, wave_fit=w, flux_obs=t, err=np.full(n, 0.01), tell_model=t,
        flux_corrected=np.ones(n), err_corrected=np.full(n, 0.01) / t, continuum=np.ones(n),
        exclusion=np.zeros(n, bool), relcol={"h2o": 0.9}, relcol_err={"h2o": 0.01},
        resolution=1.1e5, lsf_kind="gauss", stellar_treatment="template", chi2=1.0,
        chi2_reduced=1.0, dof_mask_removed=n - 7, chi2_reduced_order=1.1,
        dof_mask_removed_order=n - 7,
        chi2_reduced_joint=1.0, dof_mask_removed_joint=2 * n, status=status,
    )  # fmt: skip


def _write(path: Path, depth: float, t: datetime) -> Path:
    return write_exposure_fits(
        path,
        [_result(1, 50, depth), _result(0, 40, depth, status="corrected_not_fitted")],
        exposure=path.name.split("_joint")[0] + ".fits",
        instrument="NEID",
        correction_mode="joint_orders",
        fit_summary={
            "chi2": 2.0,
            "chi2_reduced": 1.0,
            "dof_mask_removed": 90,
            "relcol": {"h2o": 0.9},
        },
        order_metadata={1: {"used_in_fit": True, "resolution_value_source": "fitted_order"}},
        run_metadata={"time_utc": t.isoformat(), "fit_orders": [1]},
        provenance={"CFG_HASH": "abc", "GTC_VERS": "x"},
    )


def test_fits_roundtrip(tmp_path: Path) -> None:
    path = _write(tmp_path / "e1_joint_corrected_all_orders.fits", 0.5, T0)
    with fits.open(path) as h:
        assert h[0].header["GTSCHEMA"] == "S2DIMGV2" and h[0].header["NPIXMAX"] == 50
        assert h["EXCLUSION"].data[0, 40:].tolist() == [1] * 10  # padding excluded
        assert h["ORDER_SUMMARY"].data["order"].tolist() == [0, 1]
    assert read_provenance(path) == {"GTC_VERS": "x", "CFG_HASH": "abc"}
    d = load_exposure_order(path, 1)
    assert d["used_in_fit"] and d["relcol"] == {"h2o": 0.9} and d["wave_fit"].size == 50
    assert d["fit_summary"]["relcol.h2o"] == 0.9 and d["run_metadata"]["fit_orders"] == [1]
    assert d["stellar_model"] is None and d["resolution_value_source"] == "fitted_order"
    assert load_exposure_order(path, 0)["used_in_fit"] is False
    assert d["wave_shift"] is None and d["wave_shift_value_source"] == "not_fitted"
    with pytest.raises(KeyError, match="missing order 7"):
        load_exposure_order(path, 7)
    with pytest.raises(ValueError, match="order must be specified"):
        load_exposure_order(path)
    with pytest.raises(ValueError, match="unsupported correction_mode"):
        write_exposure_fits(
            tmp_path / "x.fits",
            [_result(1, 5)],
            exposure="x",
            instrument="I",
            correction_mode="bad",
        )
    with pytest.raises(ValueError, match="unknown header card"):
        write_exposure_fits(tmp_path / "x.fits", [_result(1, 5)], exposure="x", instrument="I",
                            correction_mode="joint_orders", provenance={"BAD": "1"})  # fmt: skip
    update_order_columns(path, 1, {"EXTRA": np.arange(50.0)}, header_updates={"LONGKEYWORD": 1})
    with fits.open(path) as h:
        assert h["EXTRA"].data[0, 0] != h["EXTRA"].data[0, 0] and h["EXTRA"].data[1, 3] == 3.0
    with pytest.raises(ValueError, match="does not match"):
        update_order_columns(path, 1, {"EXTRA": np.arange(3.0)})
    with pytest.raises(ValueError, match="already exists"):
        update_order_columns(path, 1, {"EXTRA": np.arange(50.0)}, overwrite_columns=False)


def test_pca_and_timeseries(tmp_path: Path) -> None:
    for i, depth in enumerate((0.3, 0.5, 0.7)):
        _write(
            tmp_path / f"e{i}_joint_corrected_all_orders.fits",
            depth,
            T0 + timedelta(minutes=10 * i),
        )
    assert discover_result_orders(tmp_path) == [0, 1]
    rows = read_order_timeseries(tmp_path, 1)
    assert [r["time"] for r in rows] == sorted(r["time"] for r in rows) and len(rows) == 3
    assert read_order_timeseries(tmp_path, 0) == []  # not used in the fit
    out = compute_telluric_pca_for_run(tmp_path, n_components=2)
    assert {r["order"]: r["status"] for r in out} == {0: "skipped", 1: "ok"}
    d = load_exposure_order(tmp_path / "e1_joint_corrected_all_orders.fits", 1)
    assert d["n_pca"] == 2 and d["pca_models"].shape == (2, 50)
    inside = np.isfinite(d["pca_models"][1])  # NaN outside the common grid
    assert inside.sum() > 40
    np.testing.assert_allclose(d["pca_models"][1][inside], d["tell_model"][inside], atol=1e-10)
    maps = compute_pca_model_maps(np.ones((1, 4)), n_components=0)
    assert maps["cumulative_maps"].shape == (0, 1, 4)
    assert len(read_order_timeseries(tmp_path, 1, require_pca=True)) == 3


def test_summary_rows(tmp_path: Path) -> None:
    r = _result(1, 10)
    col = SUMMARY_COLUMNS.index
    row = joint_row("e.fits", r, T0.isoformat(), 1.234)
    assert len(row) == len(SUMMARY_COLUMNS) and row[7] == "110000"
    assert row[col("elapsed_s")] == "1.2"
    assert row[col("wave_shift")] == ""  # no offset
    assert row[col("converged")] == "" and row[col("n_eval")] == ""  # unknown
    r.wave_shift = (0.0412345678, -0.01)
    assert joint_row("e.fits", r, "t", 0.0)[col("wave_shift")] == "[0.04123, -0.01]"
    r.converged, r.n_eval = False, 37
    assert joint_row("e.fits", r, "t", 0.0)[col("converged") :] == ["false", "37"]
    assert single_row("e.fits", r, "t", 0.0, fitted=True)[12:14] == ["", ""]
    assert single_row("e.fits", r, "t", 0.0, fitted=False)[6:16] == [""] * 10
    assert skipped_row("e.fits", 3, "t", error=KeyError())[16] == "ERROR:KeyError"
    assert skipped_row("e.fits", 3, "t", error=None)[16] == "ERROR:ValueError"
    path = _write(tmp_path / "e_joint_corrected_all_orders.fits", 0.5, T0)
    rebuilt = summary_rows_from_fits(path, T0.isoformat())
    assert len(rebuilt) == 1 and rebuilt[0][:2] == ["e.fits", 1] and rebuilt[0][-1] == ""


def test_result(tmp_path: Path) -> None:
    root = tmp_path / "run"
    data = root / "telluric_corr" / "data"
    _write(data / "neidL2_x_joint_corrected_all_orders.fits", 0.5, T0)
    with open(data / "summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(
            [SUMMARY_COLUMNS, joint_row("neidL2_x.fits", _result(1, 5), "t", 0.0)]
        )
    (root / "run_manifest.json").write_text(json.dumps({"exposures": ["neidL2_x.fits"]}))
    result = load_result(root)
    assert len(result.summary) == 1 and result.manifest["exposures"] == ["neidL2_x.fits"]
    ex = result.exposure("neidL2_x.fits")
    assert ex.wave.shape == (2, 50) and ex.orders.tolist() == [0, 1] and ex.name == "neidL2_x.fits"
    assert ex.exclusion.dtype == bool and ex.order_table["status"].tolist()[1] == "joint_ok"
    assert np.isfinite(ex.flux_corrected[1]).all() and ex.flux.shape == ex.telluric.shape
    with pytest.raises(KeyError, match="no result"):
        result.exposure("other")
    with pytest.raises(FileNotFoundError, match="not a run directory"):
        load_result(tmp_path)


def test_workflow_api_and_cli() -> None:
    assert gtc.load_result is load_result and gtc.run.__module__ == "gtellcorr.pipeline"
    assert {"Observation", "FitConfig", "Pipeline", "Result", "Star", "sites"} <= set(dir(gtc))
    from gtellcorr.cli import build_parser

    args = build_parser().parse_args(["run", "run.toml", "--reuse", "any", "--no-plot"])
    assert (args.reuse, args.no_plot, args.no_stellar) == ("any", True, False)


def test_wave_shift_columns(tmp_path: Path) -> None:
    fitted = _result(1, 50)
    fitted.wave_shift, fitted.wave_shift_err = (0.04, -0.01), (0.002, 0.003)
    path = write_exposure_fits(
        tmp_path / "e_joint_corrected_all_orders.fits", [fitted, _result(0, 40)],
        exposure="e.fits", instrument="NEID", correction_mode="joint_orders",
        order_metadata={1: {"used_in_fit": True, "wave_shift_value_source": "fitted_order"}},
    )  # fmt: skip
    d = load_exposure_order(path, 1)
    assert d["wave_shift"] == (0.04, -0.01) and d["wave_shift_err"] == (0.002, 0.003)
    assert d["wave_shift_value_source"] == "fitted_order"
    rows = {row[1]: row for row in summary_rows_from_fits(path, "t")}
    shift = SUMMARY_COLUMNS.index("wave_shift")
    assert rows[1][shift] == "[0.04, -0.01]" and rows[0][shift] == ""
    # results written before the wavelength offset existed have no offset columns
    with fits.open(path) as h:
        table = h["ORDER_SUMMARY"]
        keep = [c for c in table.columns if not c.name.startswith("wave_shift")]
        h["ORDER_SUMMARY"] = fits.BinTableHDU.from_columns(keep, name="ORDER_SUMMARY")
        h.writeto(tmp_path / "old.fits")
    old = load_exposure_order(tmp_path / "old.fits", 1)
    assert old["wave_shift"] is None and old["wave_shift_value_source"] == "not_fitted"
    old_rows = summary_rows_from_fits(tmp_path / "old.fits", "t")
    assert {row[shift] for row in old_rows} == {""}


def test_corrected_wavelength_hdus(tmp_path: Path) -> None:
    fitted = _result(1, 50)
    fitted.wave_input = fitted.wave_fit * (1.0 + 5.0 / C_KMS)  # a different delivered frame
    fitted.wave_shift = (0.5, 0.1)  # km/s: lines in the data redder than the model
    plain = _result(0, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no truncated header cards
        path = write_exposure_fits(
            tmp_path / "e_joint_corrected_all_orders.fits", [fitted, plain],
            exposure="e.fits", instrument="NEID", correction_mode="joint_orders",
        )  # fmt: skip
    with fits.open(path) as h:
        names = [hdu.name for hdu in h[1:]]
        assert names[: len(S2D_IMAGE_HDU_NAMES) + 2] == [
            *S2D_IMAGE_HDU_NAMES,
            *CORRECTED_WAVE_HDU_NAMES,
        ]
        assert h["WAVE_INPUT_CORR"].header["GTFRAME"] == "delivered"
        assert np.isnan(h["WAVE_FIT_CORR"].data[0, 40:]).all()  # padding as WAVE_FIT
    d = load_exposure_order(path, 1)
    np.testing.assert_array_equal(
        d["wave_fit_corr"], shifted_grid(fitted.wave_fit, fitted.wave_shift)
    )
    assert (d["wave_fit_corr"] < d["wave_fit"]).all()  # v > 0: true wavelengths are bluer
    np.testing.assert_allclose(
        d["wave_input_corr"] / d["wave_input"], d["wave_fit_corr"] / d["wave_fit"], rtol=1e-14
    )
    np.testing.assert_array_equal(d["wave_fit_corr"], fitted.wave_fit_corr)
    np.testing.assert_array_equal(d["wave_input_corr"], fitted.wave_input_corr)
    unfitted = load_exposure_order(path, 0)  # no offset: the uncorrected grids
    np.testing.assert_array_equal(unfitted["wave_fit_corr"], plain.wave_fit)
    np.testing.assert_array_equal(unfitted["wave_input_corr"], plain.wave_input)
    assert ExposureResult(path).image("WAVE_INPUT_CORR").shape == (2, 50)
    # no offset fitted in any order: the product keeps its previous layout
    old = _write(tmp_path / "o_joint_corrected_all_orders.fits", 0.5, T0)
    with fits.open(old) as h:
        assert not set(CORRECTED_WAVE_HDU_NAMES) & {hdu.name for hdu in h}
    d_old = load_exposure_order(old, 1)
    assert d_old["wave_fit_corr"] is None and d_old["wave_input_corr"] is None
    assert plain.wave_fit_corr is None and plain.wave_input_corr is None


def test_convergence_columns(tmp_path: Path) -> None:
    fitted = _result(1, 50)
    fitted.converged, fitted.n_eval = False, 41
    path = write_exposure_fits(
        tmp_path / "e_joint_corrected_all_orders.fits", [fitted, _result(0, 40)],
        exposure="e.fits", instrument="NEID", correction_mode="joint_orders",
        order_metadata={1: {"used_in_fit": True}},
    )  # fmt: skip
    assert (load_exposure_order(path, 1)["converged"], load_exposure_order(path, 1)["n_eval"]) == (
        False,
        41,
    )
    assert (load_exposure_order(path, 0)["converged"], load_exposure_order(path, 0)["n_eval"]) == (
        None,
        None,
    )
    col = SUMMARY_COLUMNS.index("converged")
    rows = {row[1]: row for row in summary_rows_from_fits(path, "t")}
    assert rows[1][col:] == ["false", "41"]
    # products written before the columns existed read as unknown
    with fits.open(path) as h:
        table = h["ORDER_SUMMARY"]
        keep = [c for c in table.columns if c.name not in ("converged", "n_eval")]
        h["ORDER_SUMMARY"] = fits.BinTableHDU.from_columns(keep, name="ORDER_SUMMARY")
        h.writeto(tmp_path / "old.fits")
    old = load_exposure_order(tmp_path / "old.fits", 1)
    assert (old["converged"], old["n_eval"]) == (None, None)
    assert summary_rows_from_fits(tmp_path / "old.fits", "t")[0][col:] == ["", ""]
