"""Out-of-transit, rest-frame, inverse-variance stellar master spectrum.

Every out-of-transit exposure of the RV table is shifted to the stellar rest frame
by its RV and co-added per order (inverse variance, on one common grid); the
continuum is determined once, on the co-added master. The master feeds the
measured stellar-line mask (:mod:`gtellcorr.preprocessing.stellar_lines`). The
exposures are used as delivered, telluric lines included; the line search leaves
those out by its crossmatch with the atomic line list.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.errors import DataError
from gtellcorr.preprocessing.continuum import normalise_continuum
from gtellcorr.preprocessing.windows import C_KMS, Window

if TYPE_CHECKING:
    from gtellcorr.data.observation.frames import VacuumAirFormula
    from gtellcorr.data.observation.instrument import InstrumentSpec
    from gtellcorr.data.observation.target import Transit
    from gtellcorr.preprocessing.config import MasterContinuum

__all__ = [
    "MasterSpectrum",
    "build_master_spectrum",
    "in_transit",
    "ivar_coadd",
    "load_master_spectrum",
    "save_master_spectrum",
    "shift_to_rest",
]

Array = NDArray[np.float64]
#: Per order: ``wave_nm`` (rest, air), ``flux`` and ``err`` (normalised), ``flux_raw``,
#: ``cont`` and ``n`` (number of exposures per pixel).
MasterOrders = dict[int, dict[str, NDArray[Any]]]
MasterSpectrum = MasterOrders


def shift_to_rest(wave_nm: Any, rv_kms: float) -> Array:
    """Observed -> stellar rest frame: ``lambda_rest = lambda_obs / (1 + rv/c)``."""
    return np.asarray(wave_nm, float) / (1.0 + rv_kms / C_KMS)


def in_transit(bjd: float, transit: Transit | None) -> bool:
    """True if ``bjd`` lies within T14/2 of a mid-transit (no ``t0_bjd`` -> False).

    With ``t0_bjd`` set, ``period_d`` and ``duration_d`` must be set too.
    """
    if transit is None or transit.t0_bjd is None:
        return False
    assert transit.period_d is not None and transit.duration_d is not None
    ph = ((bjd - transit.t0_bjd) / transit.period_d + 0.5) % 1.0 - 0.5
    return bool(abs(ph * transit.period_d) <= 0.5 * transit.duration_d)


def ivar_coadd(
    specs: Iterable[tuple[Any, Any, Any]], *, oversample: float = 1.0, grid: Any = None
) -> tuple[Array, Array, Array, NDArray[np.int_]]:
    """Inverse-variance co-add of ``(wave_nm, flux, err)`` spectra on one grid.

    The grid (if not given) spans the union at the median input step / ``oversample``;
    pixels without any contribution are dropped. Returns ``(wave, flux, err, n)``.
    """
    items = [
        (np.asarray(w, float), np.asarray(f, float), np.asarray(e, float))
        for w, f, e in specs
        if len(w) > 1
    ]
    if not items:
        empty = np.array([])
        return empty, empty, empty, empty.astype(int)
    if grid is None:
        lo = min(w.min() for w, _, _ in items)
        hi = max(w.max() for w, _, _ in items)
        step = np.median(np.concatenate([np.diff(w) for w, _, _ in items]))
        grid = np.arange(lo, hi, step / max(oversample, 1e-3))
    sw = np.zeros(grid.size)
    swf = np.zeros(grid.size)
    nc = np.zeros(grid.size)
    for w, f, e in items:
        f_g = np.interp(grid, w, f, left=np.nan, right=np.nan)
        e_g = np.interp(grid, w, e, left=np.nan, right=np.nan)
        m = np.isfinite(f_g) & np.isfinite(e_g) & (e_g > 0)
        wt = np.zeros(grid.size)
        wt[m] = 1.0 / e_g[m] ** 2
        sw += wt
        swf += np.where(m, wt * f_g, 0.0)
        nc += m
    good = sw > 0
    return grid[good], swf[good] / sw[good], 1.0 / np.sqrt(sw[good]), nc[good].astype(int)


def build_master_spectrum(
    rv_rows: Sequence[Mapping[str, str]],
    *,
    directory: str | os.PathLike[str],
    instrument: InstrumentSpec,
    continuum: MasterContinuum,
    broad_windows: Iterable[Window] = (),
    transit: Transit | None = None,
    orders: Sequence[int] | None = None,
    exclude: Iterable[str] = (),
    oversample: float = 1.0,
    formula: VacuumAirFormula = "espresso",
    ispec_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Co-add the out-of-transit exposures of an RV table per order.

    Exposures in *exclude* are left out (``masked``), and so are those in transit
    (:func:`in_transit`). Each order of each remaining exposure is read from
    *directory*, shifted to rest by its RV and co-added (:func:`ivar_coadd`); an order
    that cannot be read or has fewer than 50 pixels is skipped for that exposure with a
    warning. The co-add is normalised with :func:`normalise_continuum`. Orders with no
    data are dropped; :class:`~gtellcorr.errors.DataError` if none is left.

    Returns ``{"orders": {order: {wave_nm, flux, err, flux_raw, cont, n}}, "oot": [...],
    "in_transit": [...], "masked": [...]}``.
    """
    from gtellcorr.data.observation import readers

    reader = readers.get(instrument.spectral.reader)
    use_orders = list(instrument.orders if orders is None else orders)
    excluded = set(exclude)
    table = {r["file"]: (float(r["bjd"]), float(r["rv_kms"])) for r in rv_rows}
    oot: list[tuple[str, float]] = []
    intr: list[tuple[str, float]] = []
    masked: list[str] = []
    for base, (bjd, rv) in sorted(table.items()):
        if base in excluded:
            masked.append(base)
            continue
        (intr if in_transit(bjd, transit) else oot).append((base, rv))
    windows = list(broad_windows)

    out: MasterOrders = {}
    for o in use_orders:
        specs = []
        failures = []
        for base, rv in oot:
            try:
                s = reader.load_order(Path(directory) / base, o, instrument.spectral, formula)
            except DataError as exc:
                failures.append((base, type(exc).__name__, str(exc)))
                continue
            if s.wave_fit.size < 50:
                failures.append((base, "TooFewPixels", f"{s.wave_fit.size} pixels"))
                continue
            specs.append((shift_to_rest(s.wave_fit, rv), s.flux, s.err))
        if failures:
            first = failures[0]
            warnings.warn(
                f"{instrument.name} order {o}: master spectrum used {len(specs)} exposure(s), "
                f"skipped {len(failures)}. First skipped: {first[0]} ({first[1]}: {first[2]})",
                RuntimeWarning,
                stacklevel=2,
            )
        if not specs:
            continue
        w, f_raw, er_raw, n = ivar_coadd(specs, oversample=oversample)
        if not w.size:
            continue
        cont, f_norm = normalise_continuum(
            w, f_raw, config=continuum, broad_windows=windows, err=er_raw, ispec_dir=ispec_dir
        )
        er_norm = np.divide(er_raw, cont, out=np.full_like(er_raw, np.inf), where=cont > 0)
        out[o] = {
            "wave_nm": w,
            "flux": f_norm,
            "err": er_norm,
            "flux_raw": f_raw,
            "cont": cont,
            "n": n,
        }
    if not out:
        raise DataError(f"{instrument.name}: the master spectrum has no usable orders")
    return {
        "orders": out,
        "oot": [b for b, _ in oot],
        "in_transit": [b for b, _ in intr],
        "masked": masked,
    }


def save_master_spectrum(
    orders: MasterOrders, npz_path: Path, s1d_path: Path, *, oversample: float = 1.0
) -> None:
    """Write the per-order master (``.npz``) and an order-merged spectrum (``.dat``,
    the normalised orders co-added again with :func:`ivar_coadd`)."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {"orders": np.array(sorted(orders))}
    for o, d in orders.items():
        blob[f"wave_{o}"] = d["wave_nm"]
        blob[f"flux_{o}"] = d["flux"]
        blob[f"err_{o}"] = d["err"]
        blob[f"n_{o}"] = d["n"]
        blob[f"fluxraw_{o}"] = d["flux_raw"]
        blob[f"cont_{o}"] = d["cont"]
    np.savez(npz_path, **blob)
    sw, sf, se, _ = ivar_coadd(
        [(d["wave_nm"], d["flux"], d["err"]) for d in orders.values()], oversample=oversample
    )
    np.savetxt(
        s1d_path,
        np.column_stack([sw, sf, se]),
        fmt="%.6f %.6e %.6e",
        header="wave_nm_rest flux err  (order-merged S1D, ivar)",
    )


def load_master_spectrum(npz_path: str | os.PathLike[str]) -> MasterOrders:
    """Read a saved master (``flux_raw`` / ``cont`` filled for older files)."""
    z = np.load(npz_path)
    out: MasterOrders = {}
    for o in z["orders"]:
        d = {
            "wave_nm": z[f"wave_{o}"],
            "flux": z[f"flux_{o}"],
            "err": z[f"err_{o}"],
            "n": z[f"n_{o}"],
        }
        d["flux_raw"] = z[f"fluxraw_{o}"] if f"fluxraw_{o}" in z else d["flux"]
        d["cont"] = z[f"cont_{o}"] if f"cont_{o}" in z else np.ones_like(d["flux"])
        out[int(o)] = d
    return out
