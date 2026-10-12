"""Radial-velocity figures: per-exposure CCF diagnostics and the RV time series.

All three read products written by :mod:`gtellcorr.preprocessing.rv`:

* :func:`plot_ccf_diagnostics` - one exposure's ``CCF_*.npz``: the per-order CCFs
  with the weighted master CCF, and the master CCF with its Gaussian fit and RV;
* :func:`plot_rv_timeseries` - the exposure RV table: master-CCF RV and per-order
  weighted mean against time, and their difference;
* :func:`plot_rv_master_vs_perorder` - the same two RV estimates against each
  other, a quick consistency check.

Each function returns ``(figure, axes)`` and writes a PNG when ``out`` is given.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from numpy.typing import NDArray

__all__ = [
    "load_ccf_data",
    "plot_ccf_diagnostics",
    "plot_rv_master_vs_perorder",
    "plot_rv_timeseries",
]

Array = NDArray[np.float64]


def load_ccf_data(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one exposure's ``CCF_*.npz`` product into a plain dictionary."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _save(fig: Any, out: str | os.PathLike[str] | None) -> None:
    """Write *fig* to *out* (creating its directory); nothing if ``None``."""
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)


def plot_ccf_diagnostics(
    ccf_path: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    title: str | None = None,
    max_orders: int | None = None,
) -> tuple[Any, Any]:
    """Two panels for one exposure's RV measurement.

    Top: the per-order CCFs, coloured blue to red by order, and the weighted master
    CCF. Bottom: the master CCF, its Gaussian fit and the adopted RV with its 1-sigma
    uncertainty. ``max_orders`` keeps only that many highest-weight orders in the top
    panel (``None``: every accepted order).
    """
    data = load_ccf_data(ccf_path)
    v = np.asarray(data["v_grid_kms"], float)
    orders = np.asarray(data["orders"], int)
    ccfs = np.asarray(data["order_ccfs"], float)
    weights = np.asarray(data["order_weights"], float)
    keep = np.arange(orders.size)
    if max_orders is not None and orders.size > int(max_orders):
        keep = np.argsort(weights)[::-1][: int(max_orders)]
        keep = keep[np.argsort(orders[keep])]
    orders, ccfs = orders[keep], ccfs[keep]
    rv, rv_err = float(data["rv_kms"]), float(data["rv_err_kms"])

    fig, ax = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.35]}
    )
    cmap = plt.get_cmap("coolwarm")
    norm = colors.Normalize(orders.min(), orders.max()) if np.unique(orders).size > 1 else None
    for order, ccf in zip(orders, ccfs, strict=True):
        color = cmap(norm(order)) if norm is not None else "tab:blue"
        ax[0].plot(v, ccf, lw=0.7, alpha=0.58, color=color)
    ax[0].plot(v, data["master_ccf"], color="k", lw=1.8, label="weighted master CCF")
    ax[0].set_ylabel("per-order CCF")
    ax[0].legend(fontsize=8, loc="best")
    ax[0].set_title(title or f"{data['exposure']} - {data['instrument']} RV CCF")

    ax[1].plot(v, data["master_ccf"], color="k", lw=1.1, label="weighted master CCF")
    ax[1].plot(v, data["master_fit_ccf"], color="tab:blue", lw=1.2, label="best-fit Gaussian")
    ax[1].axvline(rv, color="tab:red", lw=0.9, ls="--", label=f"RV {rv:+.3f} +/- {rv_err:.3f} km/s")
    ax[1].set_xlabel("velocity [km/s]")
    ax[1].set_ylabel("master CCF")
    ax[1].legend(fontsize=8, loc="best")
    for a in ax:
        a.grid(alpha=0.25)
    shown = (
        f"{orders.size} orders" if max_orders is None else f"{orders.size} highest-weight orders"
    )
    fig.suptitle(f"{data['exposure']}  {shown}", y=0.995, fontsize=10)
    fig.tight_layout()
    _save(fig, out)
    return fig, ax


def _read_rv_table(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """The rows of an RV table (error if empty)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty RV table: {path}")
    return rows


def _column(rows: list[dict[str, str]], key: str) -> Array:
    """One column as floats (NaN where missing or not a number)."""
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, float)


def plot_rv_timeseries(
    rv_table: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Exposure RVs against time (top) and master-CCF minus per-order mean (bottom).

    Time is BJD minus its integer day; without timestamps, the exposure index. For a
    user-supplied RV table (``rv.source = "user_file"``) the per-order columns are
    empty and only the user RVs are drawn.
    """
    rows = sorted(_read_rv_table(rv_table), key=lambda r: (_column([r], "bjd")[0], r["file"]))
    bjd = _column(rows, "bjd")
    rv, rv_err = _column(rows, "rv_kms"), _column(rows, "rv_err_kms")
    po_mean, po_std = _column(rows, "po_wmean_kms"), _column(rows, "po_wstd_kms")
    if np.isfinite(bjd).any():
        zero = np.floor(np.nanmin(bjd))
        t, xlabel = bjd - zero, f"BJD - {int(zero)} [d]"
    else:
        t, xlabel = np.arange(bjd.size, dtype=float), "exposure index"

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].errorbar(t, rv, yerr=rv_err, fmt="o", ms=4, color="C0", label="master-CCF RV")
    if np.isfinite(po_mean).any():
        ax[0].errorbar(
            t, po_mean, yerr=po_std, fmt=".", ms=3, color="C1", alpha=0.65,
            label="per-order weighted mean +/- scatter",
        )  # fmt: skip
    ax[0].set_ylabel("apparent RV [km/s]")
    ax[0].set_title(title or Path(rv_table).name.replace("_rv.csv", " RV time series"))
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].plot(t, rv - po_mean, "o", ms=3, color="C2")
    ax[1].set_ylabel("master - per-order\nmean [km/s]", fontsize=8)
    ax[1].set_xlabel(xlabel)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out)
    return fig, ax


def plot_rv_master_vs_perorder(
    rv_table: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Master-CCF RV against the per-order weighted-mean RV, with the 1:1 line.

    Raises :class:`ValueError` when the table has no per-order RVs (a user RV table).
    """
    rows = _read_rv_table(rv_table)
    rv, rv_err = _column(rows, "rv_kms"), _column(rows, "rv_err_kms")
    po_mean, po_std = _column(rows, "po_wmean_kms"), _column(rows, "po_wstd_kms")
    good = np.isfinite(rv) & np.isfinite(po_mean)
    if not good.any():
        raise ValueError(f"no per-order RVs in {rv_table}")
    rv, rv_err, po_mean, po_std = rv[good], rv_err[good], po_mean[good], po_std[good]

    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo = min(rv.min(), po_mean.min()) - 0.3
    hi = max(rv.max(), po_mean.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.errorbar(po_mean, rv, xerr=po_std, yerr=rv_err, fmt="o", ms=4, color="C0")
    ax.set(
        xlabel="per-order weighted-mean RV [km/s]",
        ylabel="master-CCF RV [km/s]",
        title=title or "master CCF vs per-order RV",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out)
    return fig, ax
