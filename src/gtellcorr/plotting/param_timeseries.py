"""plot_param_timeseries — one order's fit parameters vs time, from summary.csv.

Give it the results directory (``telluric_corr/data``, where ``summary.csv`` is),
an order, and which parameters to show; it plots each parameter's time series over
the night for that order — the molecules that participated in the fit (their
relcol) plus the shared fit parameters (R, χ², masked fraction).

  plot_param_timeseries(results_dir, order=85, params=['h2o','o2','resolution'])

``params=None`` → auto (this order's molecules + resolution + chi2_reduced).
``order=None`` → every processed order overlaid, coloured by central wavelength
(the night-overview mode).  Any ``summary.csv`` column (``chi2_reduced_joint``,
``n_eval`` ...) can be a parameter.  Only successful fits (``ok`` / ``joint_ok``)
are drawn.  Time comes from the ``time_utc`` column, so nothing parses
instrument-specific filenames.  Returns ``(fig, axes)``.
A companion ``plot_param_heatmap`` gives the order × exposure heatmap view.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

_SPECIAL = {
    "resolution": "resolution R",
    "chi2_reduced": "χ² reduced (order)",
    "chi2_reduced_order": "χ² reduced (order)",
    "chi2_reduced_joint": "χ² reduced (joint exposure)",
    "masked_frac": "masked frac",
}


def _rows(results_dir):
    """The ``summary.csv`` rows of successful fits (error if there are none)."""
    summ = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summ):
        raise FileNotFoundError(f"no summary.csv in {results_dir}")
    with open(summ) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") in ("ok", "joint_ok")]
    if not rows:
        raise ValueError(f"{summ}: no successful fit rows")
    return rows


def _time(row):
    """Exposure time of a row (``time_utc``)."""
    return datetime.fromisoformat(row["time_utc"])


def _value(row, key):
    """A parameter value for one row: a ``summary.csv`` column, else a molecule name →
    its relcol (NaN where empty or missing)."""
    if key in row:
        try:
            return float(row[key])
        except (TypeError, ValueError):
            return np.nan
    rc = json.loads(row["relcol"] or "{}")  # molecule relcol
    return float(rc.get(key, np.nan))


def _instrument(results_dir):
    """Instrument of the run (``INSTRUME`` of a result FITS beside ``summary.csv``)."""
    from astropy.io import fits

    from gtellcorr.data.products.fits_schema import EXPOSURE_FITS_PATTERNS

    paths = (
        p
        for pattern in EXPOSURE_FITS_PATTERNS
        for p in glob.glob(os.path.join(results_dir, pattern))
    )
    for path in sorted(paths):
        try:
            return str(fits.getheader(path).get("INSTRUME", ""))
        except OSError:
            continue
    return ""


def _label(key):
    """Axis label of a parameter."""
    return _SPECIAL.get(key, f"{key} relcol")


def plot_param_timeseries(
    results_dir, *, order=None, params=None, instrument=None, out=None, target=""
):
    """Plot fit-parameter time series for ``order`` (int) or, if None, all orders."""
    rows = _rows(results_dir)
    inst = instrument or _instrument(results_dir)
    t0 = min(_time(r) for r in rows)

    if order is not None:  # ---- one order ----
        rr = sorted((r for r in rows if int(r["order"]) == int(order)), key=_time)
        if not rr:
            raise ValueError(f"order {order} not in summary.csv")
        if params is None:
            mols = json.loads(rr[0]["molecules"] or "[]")
            params = [*list(mols), "resolution", "chi2_reduced"]
        th = np.array([(_time(r) - t0).total_seconds() / 3600.0 for r in rr])
        fig, ax = plt.subplots(len(params), 1, figsize=(11, 2.4 * len(params)), sharex=True)
        ax = np.atleast_1d(ax)
        for a, key in zip(ax, params, strict=False):
            v = np.array([_value(r, key) for r in rr])
            a.plot(th, v, "-", marker=".", ms=4, lw=0.8, color="tab:blue")
            a.set_ylabel(_label(key))
            a.grid(alpha=0.25)
        wl = 0.5 * (float(rr[0]["wave_lo"]) + float(rr[0]["wave_hi"]))
        ax[0].set_title(
            f"{target} {inst} — order {order} ({wl:.1f} nm) fit parameters "
            f"vs time ({len(rr)} exposures)"
        )
        ax[-1].set_xlabel(f"hours since {t0:%Y-%m-%d %H:%M} UT")
        fig.tight_layout()
        default = os.path.join(results_dir, "figure", f"param_timeseries_ord{int(order):03d}.png")
    else:  # ---- all orders overlaid ----
        if params is None:
            params = ["resolution", "chi2_reduced", "masked_frac"]
        per = collections.defaultdict(list)
        wl = {}
        for r in rows:
            o = int(r["order"])
            per[o].append(((_time(r) - t0).total_seconds() / 3600.0, r))
            wl[o] = 0.5 * (float(r["wave_lo"]) + float(r["wave_hi"]))
        orders = sorted(per)
        for o in orders:
            per[o].sort(key=lambda z: z[0])
        norm = colors.Normalize(min(wl.values()), max(wl.values()))
        sm = cm.ScalarMappable(norm=norm, cmap="viridis")
        fig, ax = plt.subplots(len(params), 1, figsize=(13, 2.7 * len(params)), sharex=True)
        ax = np.atleast_1d(ax)
        for o in orders:
            th = np.array([h for h, _ in per[o]])
            col = sm.to_rgba(wl[o])
            for a, key in zip(ax, params, strict=False):
                v = np.array([_value(rr, key) for _, rr in per[o]])
                if np.all(np.isnan(v)):
                    continue
                a.plot(th, v, "-", marker=".", ms=3, lw=0.6, color=col, alpha=0.8)
        for a, key in zip(ax, params, strict=False):
            a.set_ylabel(_label(key))
            a.grid(alpha=0.25)
        nexp = len({r["exposure"] for r in rows})
        ax[0].set_title(
            f"{target} {inst} {t0:%Y-%m-%d} — per-order fit parameters "
            f"vs time ({len(orders)} orders × {nexp} exposures)"
        )
        ax[-1].set_xlabel(f"hours since {t0:%H:%M} UT")
        cb = fig.colorbar(sm, ax=ax, location="right", fraction=0.02, pad=0.01)
        cb.set_label("order central wavelength [nm]")
        default = os.path.join(results_dir, "figure", "param_timeseries_all.png")

    if out is not False:
        out = default if out is None else out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig, ax


def plot_param_heatmap(results_dir, *, params=None, instrument=None, out=None, target=""):
    """Order × exposure heatmap of the fit parameters (night overview), from
    summary.csv.  ``params=None`` → the molecules that appear + resolution + χ².
    Time axis uses the ``time_utc`` column (no filename parsing).  Returns
    ``(fig, axes)``; saves unless ``out=False``."""
    rows = _rows(results_dir)
    inst = instrument or _instrument(results_dir)
    t0 = min(_time(r) for r in rows)
    if params is None:
        mols = sorted({m for r in rows for m in json.loads(r["molecules"] or "[]")})
        params = [*mols, "resolution", "chi2_reduced"]
    orders = sorted({int(r["order"]) for r in rows})
    exps = sorted(
        {r["exposure"] for r in rows},
        key=lambda e: min(_time(r) for r in rows if r["exposure"] == e),
    )
    oi = {o: k for k, o in enumerate(orders)}
    ei = {e: k for k, e in enumerate(exps)}
    th = np.array(
        [
            (min(_time(r) for r in rows if r["exposure"] == e) - t0).total_seconds() / 3600.0
            for e in exps
        ]
    )

    ncol = 2 if len(params) > 1 else 1
    nrow = (len(params) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 3.4 * nrow), squeeze=False)
    axf = ax.ravel()
    for a, key in zip(axf, params, strict=False):
        M = np.full((len(orders), len(exps)), np.nan)
        for r in rows:
            M[oi[int(r["order"])], ei[r["exposure"]]] = _value(r, key)
        im = a.imshow(
            M,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[th.min(), th.max(), 0, len(orders)],
        )
        a.set_title(_label(key))
        a.set_xlabel(f"hours since {t0:%H:%M} UT")
        a.set_ylabel("order (↑ wavelength)")
        a.set_yticks(np.arange(len(orders)) + 0.5)
        a.set_yticklabels(orders, fontsize=5)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    for a in axf[len(params) :]:
        a.axis("off")
    fig.suptitle(
        f"{target} {inst} {t0:%Y-%m-%d} — per-order fit parameters "
        f"({len(orders)} orders × {len(exps)} exposures)"
    )
    fig.tight_layout()
    if out is not False:
        out = os.path.join(results_dir, "figure", "param_heatmap.png") if out is None else out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig, ax
