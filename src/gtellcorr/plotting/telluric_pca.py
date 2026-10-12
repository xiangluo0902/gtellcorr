"""Plot the telluric PCA models saved in the result files of a run.

This module is read-only: it does not compute PCA and does not modify result
files.  The PCA models must already have been appended by
:func:`gtellcorr.data.products.pca.compute_telluric_pca_for_run` (run with
``postprocessing.telluric_pca = true``).  One order's time series is read with
:func:`gtellcorr.data.products.fits_schema.read_order_timeseries`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from gtellcorr.data.products.fits_schema import (
    read_order_timeseries as read_single_order_timeseries,
)
from gtellcorr.data.products.pca import (
    common_wavelength_grid,
    interpolate_telluric_map,
)


def _default_out(data_dir: Path, order: int) -> Path:
    """``<product>/figure/telluric_pca_maps/order<NNN>_telluric_pca_map.png``."""
    product_root = data_dir.parent if data_dir.name == "data" else data_dir
    return (
        product_root
        / "figure"
        / "telluric_pca_maps"
        / f"order{int(order):03d}_telluric_pca_map.png"
    )


def _time_edges(times: list[datetime]) -> np.ndarray:
    """Pixel edges (matplotlib dates) around the exposure times; +-1 min for one."""
    t = mdates.date2num(times)
    if len(t) == 1:
        dt = 1.0 / (24.0 * 60.0)
        return np.array([t[0] - dt, t[0] + dt])
    mid = 0.5 * (t[:-1] + t[1:])
    first = t[0] - (mid[0] - t[0])
    last = t[-1] + (t[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _wave_edges(grid: np.ndarray) -> np.ndarray:
    """Pixel edges around the wavelength grid points; +-0.5 nm for one point."""
    if grid.size == 1:
        return np.array([grid[0] - 0.5, grid[0] + 0.5])
    mid = 0.5 * (grid[:-1] + grid[1:])
    first = grid[0] - (mid[0] - grid[0])
    last = grid[-1] + (grid[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def plot_telluric_pca_map(data_dir, *, order, out=None, vmin=None, vmax=None, wave_range=None):
    """Plot original telluric-model map plus residual maps after saved PCA models.

    ``data_dir`` is the ``telluric_corr/data`` directory.  The function reads
    only saved PCA products; it never computes PCA or writes result data.
    ``wave_range`` optionally restricts the plotted overlap to (lo_nm, hi_nm).
    Returns ``(fig, axes)`` and saves unless ``out=False``.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    rows = read_single_order_timeseries(
        data_dir, int(order), only_used_in_fit=True, require_pca=True
    )
    if not rows:
        raise FileNotFoundError(
            f"no order {order} result files with saved PCA models in {data_dir}"
        )
    n_pca = min(row["pca_models"].shape[0] for row in rows)
    if n_pca <= 0:
        raise ValueError(f"order {order}: saved PCA product has no components")
    grid = common_wavelength_grid(rows, wave_range=wave_range)
    image = interpolate_telluric_map(rows, grid)
    residuals = []
    for comp_index in range(n_pca):
        pca_rows = [{**row, "tell_model": row["pca_models"][comp_index]} for row in rows]
        pca_image = interpolate_telluric_map(pca_rows, grid)
        residuals.append(image - pca_image)

    times = [row["time"] for row in rows]
    if any(t is None for t in times):
        base = datetime(2000, 1, 1)
        times = [base + timedelta(minutes=i) for i in range(len(rows))]
    x_edges = _wave_edges(grid)
    y_edges = _time_edges(times)

    n_panel = 1 + n_pca
    fig_height = max(7.0, min(18.0, 2.4 * n_panel + 0.08 * len(rows)))
    fig, axes = plt.subplots(n_panel, 1, figsize=(13.5, fig_height), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    mesh = axes[0].pcolormesh(
        x_edges, y_edges, image, shading="auto", cmap="magma", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("fitted telluric model")
    cb = fig.colorbar(mesh, ax=axes[0], pad=0.01)
    cb.set_label("telluric transmission")

    for comp_index, ax in enumerate(axes[1:], start=1):
        residual = residuals[comp_index - 1]
        scale = np.nanpercentile(np.abs(residual), 99)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        mesh = ax.pcolormesh(
            x_edges, y_edges, residual, shading="auto", cmap="RdBu_r", vmin=-scale, vmax=scale
        )
        ax.set_title(f"fitted model - cumulative PCA model {comp_index}")
        cb = fig.colorbar(mesh, ax=ax, pad=0.01)
        cb.set_label("residual transmission")

    for ax in axes:
        ax.yaxis_date()
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylabel("time UTC")
    axes[-1].set_xlabel("wavelength (air) [nm]")
    title_times = [row["time"] for row in rows if row["time"] is not None]
    if title_times:
        date_lo = title_times[0].date().isoformat()
        date_hi = title_times[-1].date().isoformat()
        date_label = date_lo if date_lo == date_hi else f"{date_lo} to {date_hi}"
    else:
        date_label = "unknown date"
    inst = str(rows[0].get("instrument") or "")
    inst = f"{inst} " if inst else ""
    fig.suptitle(
        f"{inst}order {int(order)} telluric PCA maps | {len(rows)} exposures | {date_label}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    if out is not False:
        out = (
            _default_out(data_dir, int(order)) if out is None else Path(out).expanduser().resolve()
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig, axes
