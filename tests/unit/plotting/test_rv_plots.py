"""RV figures: CCF diagnostics, RV time series, master vs per-order RV."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gtellcorr.data.products.layout import RunLayout
from gtellcorr.pipeline import Pipeline
from gtellcorr.plotting import (
    load_ccf_data,
    plot_ccf_diagnostics,
    plot_rv_master_vs_perorder,
    plot_rv_timeseries,
)
from gtellcorr.preprocessing.config import RVConfig
from gtellcorr.preprocessing.rv import RV_TABLE_COLUMNS

V = np.linspace(-30.0, 30.0, 61)


def _gauss(rv: float) -> Any:
    return 1.0 - 0.4 * np.exp(-0.5 * ((V - rv) / 3.0) ** 2)


def _ccf_file(path: Path, n_orders: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        exposure=np.array("e1.fits"),
        instrument=np.array("NEID"),
        v_grid_kms=V,
        orders=np.arange(80, 80 + n_orders),
        order_ccfs=np.vstack([_gauss(0.1 * i) for i in range(n_orders)]),
        order_weights=np.arange(1.0, n_orders + 1.0),
        master_ccf=_gauss(0.15),
        master_fit_ccf=_gauss(0.15),
        rv_kms=0.15,
        rv_err_kms=0.02,
    )
    return path


def _rv_table(path: Path, *, measured: bool = True) -> Path:
    rows = []
    for i in range(3):
        po = f"{0.1 * i + 0.01:.3f}" if measured else "nan"
        rows.append(
            [
                f"e{i}.fits",
                f"{2459000.5 + 0.01 * i}",
                "1.1",
                f"{0.1 * i}",
                "0.02",
                "nan",
                po,
                "0.01" if measured else "nan",
                "nan",
                "nan",
                "nan",
                "nan",
            ]
        )
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RV_TABLE_COLUMNS)
        writer.writerows(rows)
    return path


def test_ccf_diagnostics(tmp_path: Path) -> None:
    ccf = _ccf_file(tmp_path / "CCF_e1.npz")
    assert load_ccf_data(ccf)["orders"].tolist() == [80, 81, 82, 83]
    fig, ax = plot_ccf_diagnostics(ccf, out=tmp_path / "fig" / "ccf.png", max_orders=2)
    assert (tmp_path / "fig" / "ccf.png").exists()
    assert len(ax[0].lines) == 2 + 1  # the two highest-weight orders + the master CCF
    assert "+0.150 +/- 0.020" in ax[1].get_legend().get_texts()[-1].get_text()
    plt.close(fig)


def test_rv_tables(tmp_path: Path) -> None:
    measured = _rv_table(tmp_path / "T_rv.csv")
    fig, _ = plot_rv_timeseries(measured, out=tmp_path / "ts.png")
    plt.close(fig)
    fig, ax = plot_rv_master_vs_perorder(measured, out=tmp_path / "mvp.png")
    assert ax.get_xlabel().startswith("per-order") and (tmp_path / "mvp.png").exists()
    plt.close(fig)
    user = _rv_table(tmp_path / "U_rv.csv", measured=False)
    fig, ax = plot_rv_timeseries(user)  # only the user RVs
    assert len(ax[0].get_legend().get_texts()) == 1
    plt.close(fig)
    with pytest.raises(ValueError, match="no per-order RVs"):
        plot_rv_master_vs_perorder(user)


@pytest.mark.parametrize(("source", "plot_ccf"), [("generated", True), ("generated", False),
                                                   ("user_file", True)])  # fmt: skip
def test_pipeline_rv_figures(tmp_path: Path, source: str, plot_ccf: bool) -> None:
    layout = RunLayout.for_run(tmp_path, target="T", site_token="S", instrument="I/NEID", date="D")
    exposure = SimpleNamespace(
        path=tmp_path / "e1.fits", time_utc=datetime(2022, 2, 13, 7, 30, 5, tzinfo=UTC)
    )
    _ccf_file(layout.ccf_path("20220213T073005", exposure.path))
    table = _rv_table(layout.rv_table, measured=source != "user_file")
    owner = SimpleNamespace(
        stellar=SimpleNamespace(rv_table=table, status={"rv": source}),
        layout=layout,
        config=SimpleNamespace(rv=RVConfig(plot_ccf=plot_ccf)),
        observation=SimpleNamespace(exposures=[exposure]),
    )
    Pipeline._plot_rv(owner)  # type: ignore[arg-type]
    figures = layout.product_dir("Radial_Velocities", "figure")
    names = sorted(p.relative_to(figures).as_posix() for p in figures.rglob("*.png"))
    expected = ["rv_timeseries.png"]
    if source != "user_file":
        expected.append("rv_master_vs_perorder.png")
        if plot_ccf:
            expected.insert(0, "ccf/e1_ccf.png")
    assert names == sorted(expected)


def test_out_false_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ccf = _ccf_file(tmp_path / "CCF_e1.npz")
    measured = _rv_table(tmp_path / "T_rv.csv")
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    for fig, _ in (
        plot_ccf_diagnostics(ccf, out=False, max_orders=2),
        plot_rv_timeseries(measured, out=False),
        plot_rv_master_vs_perorder(measured, out=False),
    ):
        plt.close(fig)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
