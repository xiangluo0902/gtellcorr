"""Figures drawn from result files: order fit, parameter time series, backend choice."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gtellcorr.data.products.fits_schema import write_exposure_fits
from gtellcorr.data.products.summary import SUMMARY_COLUMNS, joint_row
from gtellcorr.fitting.results import OrderFitResult
from gtellcorr.plotting import plot_order_fit, plot_param_timeseries

T0 = datetime(2025, 1, 1, 3, 0, tzinfo=UTC)


def _result(order: int = 0, n: int = 60, *, treatment: str = "template") -> OrderFitResult:
    w = np.linspace(650.0, 651.0, n)
    t = 1.0 - 0.5 * np.exp(-0.5 * ((w - 650.5) / 0.05) ** 2)
    m = 1.0 - 0.3 * np.exp(-0.5 * ((w - 650.3) / 0.03) ** 2)
    return OrderFitResult(
        order=order, wave_input=w, wave_fit=w, flux_obs=m * t, err=np.full(n, 0.01),
        tell_model=t, flux_corrected=m, err_corrected=np.full(n, 0.01) / t,
        continuum=np.ones(n), stellar_model=m, exclusion=np.zeros(n, bool),
        relcol={"h2o": 0.9}, relcol_err={"h2o": 0.01}, resolution=1.2e5, lsf_kind="gauss",
        stellar_treatment=treatment, chi2=1.0, chi2_reduced=1.0, dof_mask_removed=n - 7,
        chi2_reduced_order=1.1, dof_mask_removed_order=n - 7, chi2_reduced_joint=1.3,
        dof_mask_removed_joint=2 * n, converged=True, n_eval=9, status="joint_ok",
    )  # fmt: skip


def _write(path: Path, result: OrderFitResult) -> Path:
    return write_exposure_fits(
        path, [result], exposure=path.name.split("_joint")[0] + ".fits",
        instrument="LBT/PEPSI", correction_mode="joint_orders",
        order_metadata={result.order: {"used_in_fit": True}},
    )  # fmt: skip


@pytest.mark.parametrize("treatment", ["template", "model"])
def test_order_fit_draws_the_stellar_component(tmp_path: Path, treatment: str) -> None:
    path = _write(tmp_path / "e_joint_corrected_all_orders.fits", _result(treatment=treatment))
    fig, ax = plot_order_fit(str(path), order=0, out=False)
    labels = [line.get_label() for line in ax[0].get_lines()]
    plt.close(fig)
    assert "stellar  C·M" in labels and "best-fit telluric  C·T" in labels


def test_order_fit_mask_treatment_has_no_stellar_component(tmp_path: Path) -> None:
    path = _write(tmp_path / "e_joint_corrected_all_orders.fits", _result(treatment="mask"))
    fig, ax = plot_order_fit(str(path), order=0, out=False)
    labels = [line.get_label() for line in ax[0].get_lines()]
    plt.close(fig)
    assert "stellar  C·M" not in labels


def test_order_fit_flux_ylim(tmp_path: Path) -> None:
    path = _write(tmp_path / "e_joint_corrected_all_orders.fits", _result())
    fig, ax = plot_order_fit(str(path), order=0, out=False)
    assert ax[0].get_ylim() == pytest.approx((0.0, 1.25))
    plt.close(fig)
    fig, ax = plot_order_fit(str(path), order=0, out=False, flux_ylim=(0.2, 2.0))
    assert ax[0].get_ylim() == pytest.approx((0.2, 2.0))
    plt.close(fig)
    with pytest.raises(ValueError, match="flux_ylim"):
        plot_order_fit(str(path), order=0, out=False, flux_ylim=(1.0, 0.5))


def test_param_timeseries_reads_every_numeric_column(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        stem = f"pepsir.20250101.{1 + i:03d}.dxt"
        result = _result()
        result.chi2_reduced_joint = 1.3 + 0.1 * i
        if i == 0:
            _write(tmp_path / f"{stem}_joint_corrected_all_orders.fits", result)
        t = (T0 + timedelta(minutes=6 * i)).isoformat()
        rows.append(joint_row(f"{stem}.nor", result, t, 1.0))
    with open(tmp_path / "summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)
    params = ["h2o", "chi2_reduced_joint", "chi2_reduced_order", "n_eval"]
    fig, ax = plot_param_timeseries(str(tmp_path), order=0, params=params, out=False)
    values = {p: a.get_lines()[0].get_ydata() for p, a in zip(params, ax, strict=True)}
    title = ax[0].get_title()
    plt.close(fig)
    np.testing.assert_allclose(values["chi2_reduced_joint"], [1.3, 1.4, 1.5])
    np.testing.assert_allclose(values["chi2_reduced_order"], [1.1, 1.1, 1.1])
    np.testing.assert_allclose(values["n_eval"], [9, 9, 9])
    np.testing.assert_allclose(values["h2o"], [0.9, 0.9, 0.9])
    assert "LBT/PEPSI" in title  # from the result FITS, not passed in


def test_importing_plotting_keeps_the_users_backend() -> None:
    code = "import matplotlib, gtellcorr.plotting; print(matplotlib.get_backend())"
    env = {**os.environ, "MPLBACKEND": "pdf"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().lower() == "pdf"
