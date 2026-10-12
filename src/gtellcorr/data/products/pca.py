"""Telluric-model PCA of a run.

For every order, the fitted telluric models of all exposures are interpolated to a
common grid, decomposed (SVD of the mean-subtracted map), and the *cumulative*
models ``mean + PC1``, ``mean + PC1 + PC2``, ... are written back to each exposure's
result FITS on its own grid (``tell_principal_model_###``, ``TPCA_N``,
``TPCA_MODE``). Every run rebuilds these products from scratch.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from gtellcorr.data.products.fits_schema import (
    FITS_PCA_MODE_KEY,
    FITS_PCA_MODE_VALUE,
    FITS_PCA_N_KEY,
    _pca_index,
    _pca_name,
    discover_result_orders,
    read_order_timeseries,
    update_order_columns,
)

__all__ = [
    "common_wavelength_grid",
    "compute_pca_model_maps",
    "compute_telluric_pca_for_run",
    "interpolate_telluric_map",
]


def common_wavelength_grid(
    rows: Sequence[Mapping[str, Any]], wave_range: tuple[float, float] | None = None
) -> NDArray[Any]:
    """Grid at the median input pixel spacing over the rows' common wavelength range."""
    lo = max(float(np.nanmin(row["wave"])) for row in rows)
    hi = min(float(np.nanmax(row["wave"])) for row in rows)
    if wave_range is not None:
        lo = max(lo, float(wave_range[0]))
        hi = min(hi, float(wave_range[1]))
    if not hi > lo:
        raise ValueError(f"no overlapping wavelength range after selection: {lo:.6g}..{hi:.6g} nm")
    steps = []
    for row in rows:
        wave = np.asarray(row["wave"], float)
        dw = np.diff(wave[np.isfinite(wave)])
        dw = np.abs(dw[dw != 0])
        if dw.size:
            steps.append(float(np.nanmedian(dw)))
    if not steps:
        raise ValueError("cannot estimate wavelength pixel spacing")
    step = float(np.nanmedian(steps))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"invalid wavelength pixel spacing: {step}")
    n = max(2, int(np.floor((hi - lo) / step)) + 1)
    return lo + step * np.arange(n)


def interpolate_telluric_map(rows: Sequence[Mapping[str, Any]], grid: NDArray[Any]) -> NDArray[Any]:
    """Each row's telluric model on ``grid`` (NaN outside its range)."""
    image = np.full((len(rows), grid.size), np.nan, float)
    for i, row in enumerate(rows):
        wave = np.asarray(row["wave"], float)
        tell = np.asarray(row["tell_model"], float)
        ok = np.isfinite(wave) & np.isfinite(tell)
        if np.count_nonzero(ok) < 2:
            continue
        wave, tell = wave[ok], tell[ok]
        order = np.argsort(wave)
        image[i] = np.interp(grid, wave[order], tell[order], left=np.nan, right=np.nan)
    return image


def compute_pca_model_maps(image: NDArray[Any], n_components: int = 3) -> dict[str, NDArray[Any]]:
    """Mean, per-component, cumulative and residual maps of a telluric-model image."""
    image = np.asarray(image, float)
    mean = np.nanmean(image, axis=0)
    centered = image - mean
    valid = np.isfinite(centered)
    filled = np.where(valid, centered, 0.0)
    n_components = max(0, min(int(n_components), min(filled.shape)))
    if n_components == 0:
        empty = np.empty((0, *filled.shape), float)
        return {
            "mean": mean,
            "component_maps": empty,
            "cumulative_maps": empty,
            "residual_maps": empty,
        }
    u, s, vt = np.linalg.svd(filled, full_matrices=False)
    component_maps = np.empty((n_components, *filled.shape), float)
    cumulative_maps = np.empty_like(component_maps)
    residual_maps = np.empty_like(component_maps)
    cumulative = np.zeros_like(filled)
    for i in range(n_components):
        component = (u[:, [i]] * s[i]) @ vt[[i], :]
        cumulative = cumulative + component
        residual = filled - cumulative
        component[~valid] = np.nan
        cumulative_out = cumulative.copy()
        cumulative_out[~valid] = np.nan
        residual[~valid] = np.nan
        component_maps[i] = component
        cumulative_maps[i] = cumulative_out
        residual_maps[i] = residual
    return {
        "mean": mean,
        "component_maps": component_maps,
        "cumulative_maps": cumulative_maps,
        "residual_maps": residual_maps,
    }


def _saved_pca_products(path: Path) -> list[str]:
    """Names of the PCA models stored in a result FITS."""
    products = []
    with fits.open(path) as hdul:
        for hdu in hdul[1:]:
            if _pca_index(getattr(hdu, "name", "")) is not None:
                products.append(hdu.name)
            for name in getattr(getattr(hdu, "columns", None), "names", None) or []:
                if _pca_index(name) is not None:
                    products.append(name)
    return sorted(set(products))


def _clear(path_orders: Mapping[Path, int]) -> None:
    """Remove earlier PCA models from the given result files."""
    for path, order in sorted((Path(p), int(o)) for p, o in path_orders.items()):
        remove = _saved_pca_products(path)
        if remove:
            update_order_columns(path, order, {}, remove_columns=remove)


def _write_models(
    rows: Sequence[Mapping[str, Any]],
    grid: NDArray[Any],
    pca: Mapping[str, NDArray[Any]],
    n_components: int,
) -> int:
    """Write the cumulative models back onto each exposure's grid; number of rows written."""
    cumulative = (
        np.asarray(pca["cumulative_maps"], float) + np.asarray(pca["mean"], float)[None, None, :]
    )
    n_save = min(int(n_components), cumulative.shape[0])
    written = 0
    for i, row in enumerate(rows):
        wave = np.asarray(row["wave"], float)
        models = np.full((n_save, wave.size), np.nan, float)
        for c in range(n_save):
            model = cumulative[c, i]
            ok = np.isfinite(grid) & np.isfinite(model)
            if np.count_nonzero(ok) >= 2:
                models[c] = np.interp(wave, grid[ok], model[ok], left=np.nan, right=np.nan)
        update_order_columns(
            row["path"],
            int(row["order"]),
            {_pca_name(c + 1): models[c] for c in range(n_save)},
            header_updates={
                FITS_PCA_N_KEY: (n_save, "number of PCA models"),
                FITS_PCA_MODE_KEY: (FITS_PCA_MODE_VALUE, "model k = mean + first k components"),
            },
        )
        written += 1
    return written


def compute_telluric_pca_for_run(
    data_dir: str | os.PathLike[str], *, orders: Iterable[int] | None = None, n_components: int = 3
) -> list[dict[str, Any]]:
    """Rebuild the cumulative PCA models of all (or the given) orders of a run."""
    root = Path(data_dir).expanduser().resolve()
    use = sorted({int(o) for o in (discover_result_orders(root) if orders is None else orders)})
    rows_by_order = {
        o: read_order_timeseries(root, o, only_used_in_fit=True, require_pca=False) for o in use
    }
    clear: dict[Path, int] = {}
    for rows in rows_by_order.values():
        for row in rows:
            clear.setdefault(Path(row["path"]), int(row["order"]))
    _clear(clear)
    out = []
    for order in use:
        rows = rows_by_order[order]
        if len(rows) < 2:
            out.append(
                {"order": order, "status": "skipped", "reason": "fewer than 2 exposures",
                 "n_exposures": len(rows), "n_written": 0}
            )  # fmt: skip
            continue
        grid = common_wavelength_grid(rows)
        pca = compute_pca_model_maps(
            interpolate_telluric_map(rows, grid), n_components=n_components
        )
        out.append(
            {
                "order": order,
                "status": "ok",
                "n_exposures": len(rows),
                "n_components": min(int(n_components), pca["cumulative_maps"].shape[0]),
                "n_wave": int(grid.size),
                "n_written": _write_models(rows, grid, pca, n_components),
            }
        )
    return out
