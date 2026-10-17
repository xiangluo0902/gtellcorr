"""Per-exposure stellar radial velocity: CCF against the rest template + MCMC.

1. Per order: continuum-normalise, cross-correlate with the rest-frame template on a
   common velocity grid (Pearson CCF), fit a Gaussian; orders failing the quality
   checks are dropped.
2. Co-add the per-order CCFs weighted by ``contrast^2 * N_pix`` (the master CCF).
3. Fit a Gaussian to the master CCF with MCMC (emcee): the exposure RV is the
   posterior median of its centre, the 1-sigma error half the 16-84 % range.

The spectra are in the fit frame (air, topocentric) and the template in air at rest,
so the RV is the apparent topocentric stellar RV (approximately Vsys - BERV). With a
user template in the topocentric frame (``template_frame = "topocentric"``) it is
the shift relative to that template instead.
Masked pixels (telluric bands, activity lines, user windows, each padded by
``RVConfig.pad_kms``) never enter a CCF.
"""

from __future__ import annotations

import csv
import logging
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.data.observation.header import read_scalar
from gtellcorr.errors import DataError
from gtellcorr.preprocessing.windows import C_KMS, Window, good_pixels, pad_window, read_windows

if TYPE_CHECKING:
    from gtellcorr.data.observation.frames import VacuumAirFormula
    from gtellcorr.data.observation.header import HeaderSchema
    from gtellcorr.data.observation.instrument import InstrumentSpec
    from gtellcorr.data.observation.observation import Exposure
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.fitting.config import Molecules
    from gtellcorr.preprocessing.config import RVConfig

__all__ = [
    "RV_TABLE_COLUMNS",
    "ExposureRVResult",
    "ccf_grid_limits",
    "coadd_ccfs",
    "cross_correlate",
    "exposure_bjd",
    "gaussian_fit_ccf",
    "measure_rv_exposure",
    "measure_rv_table",
    "normalize_continuum",
    "read_rv_table",
    "rv_mask_windows",
    "rv_orders",
    "save_ccf_data",
    "write_user_rv_table",
]

Array = NDArray[np.float64]
log = logging.getLogger(__name__)

#: Columns of the RV table: exposure file, BJD_TDB of mid-exposure (see
#: :func:`exposure_bjd`), airmass, MCMC RV and error, Gaussian-fit
#: RV of the master CCF, weighted mean and scatter of the per-order RVs, FWHM and
#: contrast of the master CCF, number of orders used, MCMC acceptance fraction.
RV_TABLE_COLUMNS = (
    "file", "bjd", "airmass", "rv_kms", "rv_err_kms", "rv_master_ccf_kms", "po_wmean_kms",
    "po_wstd_kms", "fwhm_kms", "contrast", "n_orders", "acceptance",
)  # fmt: skip
#: Columns of the per-order RV table (Gaussian fit of each order's CCF).
PER_ORDER_COLUMNS = (
    "file", "bjd", "order", "rv_kms", "rv_err_kms", "weight", "n_good", "contrast",
)  # fmt: skip

#: MCMC walkers, steps and burn-in steps used for the RV table.
MCMC_WALKERS, MCMC_STEPS, MCMC_BURN = 20, 900, 300


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def rv_mask_windows(
    config: RVConfig,
    molecules: Molecules,
    activity_windows: Iterable[Window],
    telluric_windows: Iterable[Window] | None = None,
) -> list[Window]:
    """Windows excluded from the CCFs (not merged).

    Telluric: with ``telluric_mask = "model"`` the model windows *telluric_windows*
    (:func:`~gtellcorr.preprocessing.telluric_mask.ensure_rv_telluric_windows`, already
    padded); with ``"bands"``, or ``"model"`` without *telluric_windows* (a warning),
    the bands of ``config.telluric_mask_molecules`` padded by ``pad_kms``. Then the
    activity windows if ``config.mask_activity`` and the user windows and file, each
    padded by ``pad_kms``.
    """
    windows: list[Window] = []
    padded: list[Window] = []
    if config.telluric_mask == "model" and telluric_windows is not None:
        padded.extend((float(lo), float(hi)) for lo, hi in telluric_windows)
    else:
        if config.telluric_mask == "model":
            log.warning(
                "rv.telluric_mask = 'model' but no telluric model windows were given; "
                "masking the bands of %s instead",
                ", ".join(config.telluric_mask_molecules),
            )
        for m in config.telluric_mask_molecules:
            windows.extend((float(b[0]), float(b[1])) for b in molecules[m].bands_nm)
    if config.mask_activity:
        windows.extend((float(w[0]), float(w[1])) for w in activity_windows)
    windows.extend((float(w[0]), float(w[1])) for w in config.user_mask_nm)
    if config.user_mask_file:
        windows.extend(read_windows(config.user_mask_file, ordered=True))
    return padded + [pad_window(lo, hi, config.pad_kms) for lo, hi in windows]


# ---------------------------------------------------------------------------
# cross-correlation
# ---------------------------------------------------------------------------
def _shift_template(obs_wave: Array, tmpl_wave: Array, tmpl_flux: Array, v_kms: float) -> Array:
    """Rest template in the observed frame at RV ``v_kms``; 1.0 outside the template."""
    rest = obs_wave / (1.0 + v_kms / C_KMS)
    return np.interp(rest, tmpl_wave, tmpl_flux, left=1.0, right=1.0)


def cross_correlate(
    obs_wave: Any,
    obs_flux: Any,
    tmpl_wave: Any,
    tmpl_flux: Any,
    v_grid_kms: Any,
    *,
    good: Any = None,
) -> Array:
    """Mean-subtracted, normalised (Pearson) CCF over ``v_grid_kms``.

    The template is shifted to each velocity and interpolated onto the observed
    pixels (flux 1 outside it). ``good`` selects the observed pixels to use (all if
    not given).
    """
    ow = np.asarray(obs_wave, float)
    of = np.asarray(obs_flux, float)
    if good is None:
        good = np.ones(ow.shape, bool)
    ow, of = ow[good], of[good]
    o = of - of.mean()
    o_norm = np.sqrt(np.dot(o, o))
    ccf = np.empty(len(v_grid_kms))
    for i, v in enumerate(v_grid_kms):
        m = _shift_template(ow, tmpl_wave, tmpl_flux, v)
        m = m - m.mean()
        denom = o_norm * np.sqrt(np.dot(m, m))
        ccf[i] = np.dot(o, m) / denom if denom > 0 else 0.0
    return ccf


def normalize_continuum(
    wave_nm: Any, flux: Any, *, order: int = 3, n_iter: int = 4, hi: float = 3.0, lo: float = 2.0
) -> Array:
    """Upper-envelope sigma-clipped polynomial normalisation -> flux / continuum."""
    w = np.asarray(wave_nm, float)
    f = np.asarray(flux, float)
    finite = np.isfinite(w) & np.isfinite(f)
    if finite.sum() < order + 2 or not (np.nanmax(w[finite]) > np.nanmin(w[finite])):
        med = np.nanmedian(f[finite]) if finite.any() else 1.0
        return f / med if np.isfinite(med) and med != 0 else f
    w0 = w - w.mean()
    good = np.isfinite(f)
    cont = np.ones_like(f)
    for _ in range(n_iter):
        if good.sum() < order + 2:
            break
        try:
            coef = np.polyfit(w0[good], f[good], order)
        except np.linalg.LinAlgError:
            med = np.nanmedian(f[good])
            return f / med if np.isfinite(med) and med != 0 else f
        cont = np.polyval(coef, w0)
        resid = f - cont
        s = np.std(resid[good])
        if not (s > 0):
            break
        good = np.isfinite(f) & (resid < hi * s) & (resid > -lo * s)
    cont = np.where(np.abs(cont) > 1e-6, cont, 1.0)
    return np.asarray(f / cont, dtype=np.float64)


def _gaussian(v: Any, amp: float, mu: float, sig: float, off: float) -> Any:
    """Gaussian of amplitude *amp*, centre *mu* and width *sig* on an offset *off*."""
    return off + amp * np.exp(-0.5 * ((v - mu) / sig) ** 2)


def gaussian_fit_ccf(v_grid_kms: Any, ccf: Any) -> dict[str, float] | None:
    """Gaussian fit of a CCF peak -> ``rv, rv_err, sigma, contrast, offset`` (None on failure)."""
    from scipy.optimize import OptimizeWarning, curve_fit

    v = np.asarray(v_grid_kms, float)
    c = np.asarray(ccf, float)
    off0 = float(np.median(c))
    k = int(np.argmax(c))
    amp0 = float(c[k] - off0)
    if amp0 <= 0:
        return None
    p0 = [amp0, float(v[k]), max(2.0, (v[-1] - v[0]) / 50.0), off0]
    try:
        with warnings.catch_warnings():
            # an undetermined covariance is not a failure: rv_err becomes NaN below
            warnings.simplefilter("ignore", OptimizeWarning)
            p, cov = curve_fit(_gaussian, v, c, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
        return None  # any fit failure rejects the CCF
    amp, mu, sig, off = p
    rv_err = float(np.sqrt(cov[1, 1])) if np.all(np.isfinite(cov)) else float("nan")
    return {
        "rv": float(mu),
        "rv_err": rv_err,
        "sigma": abs(float(sig)),
        "contrast": float(amp),
        "offset": float(off),
    }


def coadd_ccfs(ccf_list: Sequence[Array], weights: Any) -> Array:
    """Weighted mean of CCFs on one grid (plain mean if the weights sum to zero)."""
    ccfs = np.vstack(ccf_list)
    w = np.asarray(weights, float)
    total = w.sum()
    if not (total > 0):
        return np.asarray(ccfs.mean(axis=0), float)
    return np.asarray((ccfs * w[:, None]).sum(axis=0) / total, float)


@dataclass
class ExposureRVResult:
    """One exposure's RV from the co-added master CCF, with per-order QC."""

    rv_kms: float
    """RV [km/s]: MCMC posterior median (the Gaussian fit if MCMC is off)."""
    rv_err_kms: float
    """1-sigma RV error [km/s]."""
    rv_master_ccf_kms: float
    """Centre of the least-squares Gaussian fit of the master CCF [km/s]."""
    fwhm_kms: float = 0.0
    """FWHM of the master-CCF Gaussian [km/s]."""
    contrast: float = 0.0
    """Amplitude of the master-CCF Gaussian."""
    per_order: list[dict[str, Any]] = field(default_factory=list)
    """Per used order: ``order, rv, rv_err, weight, n_good, contrast``."""
    ccf_by_order: list[dict[str, Any]] = field(default_factory=list)
    """Per used order: ``order, ccf, rv, weight, contrast``."""
    per_order_wmean_kms: float = float("nan")
    """Weighted mean of the per-order RVs [km/s]."""
    per_order_wstd_kms: float = float("nan")
    """Weighted scatter of the per-order RVs [km/s]."""
    v_grid_kms: Array | None = None
    """Velocity grid of the CCFs [km/s]."""
    master_ccf: Array | None = None
    """Weighted co-added CCF."""
    n_orders_used: int = 0
    """Orders that passed the quality checks."""
    acceptance: float = 0.0
    """Mean MCMC acceptance fraction (0 without MCMC)."""
    label: str = ""
    """Exposure name, for messages and products."""


def _ccf_gauss_log_prob(
    theta: Any, v: Array, ccf: Array, sigma_ccf: float, bounds: Sequence[tuple[float, float]]
) -> float:
    """Log-probability of a Gaussian CCF model: flat priors within *bounds*, Gaussian
    likelihood with per-point error *sigma_ccf*."""
    amp, mu, sig, off = theta
    (alo, ahi), (mlo, mhi), (slo, shi), (olo, ohi) = bounds
    if not (alo < amp < ahi and mlo < mu < mhi and slo < sig < shi and olo < off < ohi):
        return -np.inf
    model = off + amp * np.exp(-0.5 * ((v - mu) / sig) ** 2)
    r = (ccf - model) / sigma_ccf
    return -0.5 * float(np.dot(r, r))


def measure_rv_exposure(
    orders: Iterable[tuple[int, Any, Any, Any]],
    tmpl_wave_nm: Any,
    tmpl_flux: Any,
    *,
    mask_windows: Sequence[Window],
    v_lo: float = -150.0,
    v_hi: float = 150.0,
    dv: float = 0.5,
    normalize: bool = True,
    min_good: int = 50,
    min_contrast: float = 0.02,
    mcmc: bool = True,
    n_walkers: int = 24,
    n_steps: int = 1500,
    n_burn: int = 400,
    seed: int | None = None,
    label: str = "",
) -> ExposureRVResult:
    """RV of one exposure from ``(order, wave_nm, flux, err)`` tuples.

    An order is used if it has at least *min_good* unmasked pixels and its CCF has a
    Gaussian fit with contrast >= *min_contrast*, centre inside the grid, width below
    half the grid and a finite error. The defaults here are generic; the pipeline
    passes the ``RVConfig`` grid and ``MCMC_WALKERS/STEPS/BURN``
    (:func:`measure_rv_table`).

    The MCMC walkers start around the master-CCF fit (fixed generator); *seed* seeds
    the sampler, ``None`` starts it from a copy of numpy's global generator state. The
    CCF noise is the scatter of the master CCF about its fit beyond 4 sigma (of the
    fitted Gaussian) from the centre, or over the whole grid if fewer than 6 points lie
    there.
    """
    v_grid = np.arange(v_lo, v_hi + dv, dv)
    ccf_list, weights, per_order = [], [], []
    for oid, w, f, e in orders:
        w = np.asarray(w, float)
        f = np.asarray(f, float)
        e = np.asarray(e, float)
        good = (
            np.isfinite(w)
            & np.isfinite(f)
            & np.isfinite(e)
            & (e > 0)
            & good_pixels(w, mask_windows)
        )
        if good.sum() < min_good:
            continue
        wo, fo = w[good], f[good]
        if normalize:
            fo = normalize_continuum(wo, fo)
        ccf = cross_correlate(wo, fo, tmpl_wave_nm, tmpl_flux, v_grid)
        g = gaussian_fit_ccf(v_grid, ccf)
        if (
            g is None
            or g["contrast"] < min_contrast
            or not (v_lo < g["rv"] < v_hi)
            or not (0.0 < g["sigma"] < 0.5 * (v_hi - v_lo))
            or not np.isfinite(g["rv_err"])
        ):
            continue
        wgt = g["contrast"] ** 2 * int(good.sum())
        ccf_list.append(ccf)
        weights.append(wgt)
        per_order.append(
            {
                "order": oid,
                "rv": g["rv"],
                "rv_err": g["rv_err"],
                "weight": wgt,
                "n_good": int(good.sum()),
                "contrast": g["contrast"],
            }
        )
    if not ccf_list:
        raise DataError("no order produced a usable CCF (all masked / too shallow)")

    weight_arr = np.asarray(weights, float)
    master = coadd_ccfs(ccf_list, weight_arr)
    rv_arr = np.array([p["rv"] for p in per_order])
    total = weight_arr.sum()
    wmean = float(np.dot(weight_arr, rv_arr) / total)
    wstd = float(np.sqrt(np.dot(weight_arr, (rv_arr - wmean) ** 2) / total))

    gm = gaussian_fit_ccf(v_grid, master)
    if gm is None:
        raise DataError("master-CCF Gaussian fit failed")
    rv_seed, sig, amp, off = gm["rv"], gm["sigma"], gm["contrast"], gm["offset"]
    rv, rv_err, accept = rv_seed, gm["rv_err"], 0.0

    if mcmc:
        import emcee

        model0 = off + amp * np.exp(-0.5 * ((v_grid - rv_seed) / sig) ** 2)
        wing = np.abs(v_grid - rv_seed) > 4 * sig
        sigma_ccf = (
            float(np.std((master - model0)[wing]))
            if wing.sum() > 5
            else float(np.std(master - model0))
        )
        sigma_ccf = max(sigma_ccf, 1e-6)
        bounds = ((0.0, 5 * amp), (v_lo, v_hi), (0.5 * sig, 3 * sig), (off - 0.5, off + 0.5))
        rng = np.random.default_rng(0)
        p0 = np.column_stack(
            [
                np.clip(amp + rng.normal(0, 0.02 * amp, n_walkers), 1e-3, 5 * amp),
                rv_seed + rng.normal(0, max(0.05, 0.1 * sig), n_walkers),
                np.clip(sig + rng.normal(0, 0.05 * sig, n_walkers), 0.51 * sig, 2.9 * sig),
                off + rng.normal(0, 0.2 * sigma_ccf, n_walkers),
            ]
        )
        sampler = emcee.EnsembleSampler(
            n_walkers, 4, _ccf_gauss_log_prob, args=(v_grid, master, sigma_ccf, bounds)
        )
        if seed is not None:
            sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(p0, n_steps, progress=False)
        flat = sampler.get_chain(discard=n_burn, flat=True)
        lo16, med, hi84 = np.percentile(flat[:, 1], [16, 50, 84])
        rv, rv_err = float(med), float(0.5 * (hi84 - lo16))
        accept = float(np.mean(sampler.acceptance_fraction))

    return ExposureRVResult(
        rv_kms=rv,
        rv_err_kms=rv_err,
        rv_master_ccf_kms=float(rv_seed),
        fwhm_kms=float(2.3548 * sig),
        contrast=float(amp),
        per_order=per_order,
        per_order_wmean_kms=wmean,
        per_order_wstd_kms=wstd,
        ccf_by_order=[
            {
                "order": p["order"],
                "ccf": np.asarray(c, float),
                "rv": p["rv"],
                "weight": p["weight"],
                "contrast": p["contrast"],
            }
            for p, c in zip(per_order, ccf_list, strict=True)
        ],
        v_grid_kms=v_grid,
        master_ccf=master,
        n_orders_used=len(ccf_list),
        acceptance=accept,
        label=label,
    )


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------
def save_ccf_data(
    result: ExposureRVResult,
    path: str | os.PathLike[str],
    *,
    star_name: str,
    instrument: str,
    obs_time: str,
    exposure: str | os.PathLike[str] | None = None,
    bjd: float = np.nan,
    airmass: float = np.nan,
) -> Path:
    """Write one exposure's CCF product (``.npz``): the velocity grid, per-order CCFs
    with their weights, RVs and contrasts, the master CCF with its Gaussian fit, and
    the exposure's RV results."""
    if result.v_grid_kms is None or result.master_ccf is None or not result.ccf_by_order:
        raise ValueError("result does not carry CCF arrays")
    v = np.asarray(result.v_grid_kms, float)
    items = sorted(result.ccf_by_order, key=lambda d: d["order"])
    orders = np.array([d["order"] for d in items], dtype=int)
    by_order = {p["order"]: p for p in result.per_order}
    fit = gaussian_fit_ccf(v, result.master_ccf)
    if fit is None:
        profile = np.full_like(v, np.nan, dtype=float)
        fit_rv = fit_err = fit_sigma = fit_amp = fit_offset = np.nan
    else:
        profile = _gaussian(v, fit["contrast"], fit["rv"], fit["sigma"], fit["offset"])
        fit_rv, fit_err, fit_sigma = fit["rv"], fit["rv_err"], fit["sigma"]
        fit_amp, fit_offset = fit["contrast"], fit["offset"]
    out = Path(os.path.abspath(str(path)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        data_type=np.array("CCF"),
        star_name=np.array(str(star_name)),
        instrument=np.array(str(instrument)),
        obs_time=np.array(str(obs_time)),
        exposure=np.array(os.path.basename(str(exposure or result.label))),
        bjd=float(bjd),
        airmass=float(airmass),
        v_grid_kms=v,
        orders=orders,
        order_ccfs=np.vstack([np.asarray(d["ccf"], float) for d in items]),
        order_weights=np.array([d["weight"] for d in items], dtype=float),
        order_rv_kms=np.array([d["rv"] for d in items], dtype=float),
        order_rv_err_kms=np.array([by_order[int(o)]["rv_err"] for o in orders], dtype=float),
        order_n_good=np.array([by_order[int(o)]["n_good"] for o in orders], dtype=int),
        order_contrast=np.array([d["contrast"] for d in items], dtype=float),
        master_ccf=np.asarray(result.master_ccf, float),
        master_fit_ccf=profile,
        fit_rv_kms=float(fit_rv),
        fit_rv_err_kms=float(fit_err),
        fit_sigma_kms=float(fit_sigma),
        fit_contrast=float(fit_amp),
        fit_offset=float(fit_offset),
        rv_kms=float(result.rv_kms),
        rv_err_kms=float(result.rv_err_kms),
        rv_master_ccf_kms=float(result.rv_master_ccf_kms),
        per_order_wmean_kms=float(result.per_order_wmean_kms),
        per_order_wstd_kms=float(result.per_order_wstd_kms),
        fwhm_kms=float(result.fwhm_kms),
        contrast=float(result.contrast),
        n_orders_used=int(result.n_orders_used),
        acceptance=float(result.acceptance),
    )
    return out


def read_rv_table(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Rows of an RV table (strings, as written)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise DataError(f"empty RV table: {path}")
    return rows


def write_user_rv_table(
    exposures: Sequence[Exposure],
    user_file: str | os.PathLike[str],
    layout: RunLayout,
    *,
    instrument: InstrumentSpec | None = None,
) -> Path:
    """Write the run's RV table from a user file (``rv.source = "user_file"``).

    The user file is a CSV with ``file`` (exposure file name or stem) and ``rv_kms``
    columns, optionally ``rv_err_kms``. The run table keeps the measured-table layout
    (``bjd`` from the *instrument*'s BJD card, see :func:`exposure_bjd`; ``airmass``
    from the exposure; measurement-only columns ``nan``), so every consumer reads it
    unchanged. Raises if an exposure has no row.
    """
    from astropy.io import fits

    path = Path(user_file).expanduser()
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {"file", "rv_kms"} <= set(rows[0]):
        raise DataError(f"{path}: an RV file needs the columns 'file' and 'rv_kms'")
    by_name: dict[str, Mapping[str, str]] = {}
    for row in rows:
        name = os.path.basename(str(row["file"]).strip())
        by_name[name] = row
        by_name.setdefault(Path(name).stem, row)
    table, missing, without_bjd = [], [], []
    schema = None if instrument is None else instrument.header
    for exposure in exposures:
        base = exposure.path.name
        match = by_name.get(base) or by_name.get(exposure.path.stem)
        if match is None:
            missing.append(base)
            continue
        rv = float(match["rv_kms"])
        if not np.isfinite(rv):
            raise DataError(f"{path}: non-finite rv_kms for {base}")
        err = float(match.get("rv_err_kms") or "nan")
        with fits.open(exposure.path, memmap=True) as hdul:
            bjd, found = exposure_bjd(hdul[0].header, schema, exposure.time_utc)
        if not found:
            without_bjd.append(base)
        nan = float("nan")
        table.append((base, bjd, float(exposure.airmass), rv, err, *([nan] * 7)))
    if missing:
        raise DataError(f"{path}: no RV for {len(missing)} exposure(s): {', '.join(missing)}")
    _warn_without_bjd(without_bjd, "the instrument" if instrument is None else instrument.name)
    layout.rv_table.parent.mkdir(parents=True, exist_ok=True)
    with open(layout.rv_table, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RV_TABLE_COLUMNS)
        writer.writerows(table)
    with open(layout.rv_per_order_table, "w", newline="") as fh:
        csv.writer(fh).writerow(PER_ORDER_COLUMNS)
    return layout.rv_table


def rv_table_covers(path: str | os.PathLike[str], names: Iterable[str]) -> bool:
    """True if an existing table has a row for every exposure file name."""
    try:
        rows = read_rv_table(path)
    except (OSError, DataError, KeyError):
        return False
    have = {os.path.basename(str(r.get("file", ""))) for r in rows}
    return set(names).issubset(have)


def exposure_bjd(
    header: Mapping[str, Any], schema: HeaderSchema | None, mid_exposure_utc: datetime | None
) -> tuple[float, bool]:
    """BJD_TDB of an exposure, and whether the header gave it.

    The instrument declares the header cards holding the BJD_TDB of mid-exposure
    (``header.bjd`` of its definition). If none is declared or present, the JD (UTC)
    of *mid_exposure_utc* (the exposure epoch, ``Exposure.time_utc``) is returned
    instead, NaN without an epoch.
    """
    if schema is not None:
        value = read_scalar(header, schema.bjd)
        if value is not None:
            return float(value), True
    if mid_exposure_utc is None:
        return float("nan"), False
    from astropy.time import Time

    utc = mid_exposure_utc.astimezone(UTC).replace(tzinfo=None)
    return float(Time(utc, scale="utc").jd), False


def _warn_without_bjd(names: Sequence[str], instrument: str) -> None:
    """Warn that the RV table holds plain JDs (UTC) for *names*."""
    if names:
        log.warning(
            "%s: %d exposure(s) without a BJD_TDB header card (instrument header.bjd); the "
            "RV table's bjd column holds the JD (UTC) of their epoch instead: %s",
            instrument,
            len(names),
            ", ".join(names),
        )


def rv_orders(instrument: InstrumentSpec, config: RVConfig) -> list[int]:
    """Orders used for the RV: ``config.orders``, else every instrument order (order 0
    for a merged PEPSI spectrum without order windows)."""
    if config.orders is not None:
        return list(config.orders)
    spectral = instrument.spectral
    if spectral.reader == "pepsi_dxt" and getattr(spectral, "order_windows_nm", None) is None:
        return [0]
    return list(instrument.orders)


def ccf_grid_limits(config: RVConfig, vsini_kms: float | None = None) -> tuple[float, float]:
    """Ends of the CCF velocity grid [km/s]: ``ccf_v_lo``-``ccf_v_hi``, each widened by
    ``ccf_vsini_pad * vsini`` when *vsini_kms* is known (a rotationally broadened
    CCF reaches ``vsini`` from its centre and needs its wings on the grid)."""
    pad = 0.0
    if vsini_kms is not None and np.isfinite(vsini_kms) and vsini_kms > 0:
        pad = float(config.ccf_vsini_pad) * float(vsini_kms)
    return float(config.ccf_v_lo) - pad, float(config.ccf_v_hi) + pad


def measure_rv_table(
    exposures: Sequence[Exposure],
    template: tuple[Any, Any],
    *,
    instrument: InstrumentSpec,
    layout: RunLayout,
    config: RVConfig,
    mask_windows: Sequence[Window],
    formula: VacuumAirFormula = "espresso",
    seed: int | None = None,
    mcmc: bool = True,
    vsini_kms: float | None = None,
) -> Path:
    """Measure every exposure's RV and write the RV tables and CCF products.

    Orders with fewer than 50 fit-frame pixels are skipped. *vsini_kms* (the target's)
    widens the CCF grid (:func:`ccf_grid_limits`). Returns the exposure table path;
    whether an existing table is reused is decided by
    :func:`gtellcorr.preprocessing.prepare.ensure_rv_table`.
    """
    from astropy.io import fits

    from gtellcorr.data.observation import readers

    if not exposures:
        raise DataError(f"{instrument.name}: no exposures available for RV measurement")
    if not config.ccf_v_lo < config.ccf_v_hi:
        raise DataError(f"invalid RV CCF grid: v_lo={config.ccf_v_lo} >= v_hi={config.ccf_v_hi}")
    if not config.ccf_dv > 0:
        raise DataError(f"invalid RV CCF grid: dv={config.ccf_dv} must be > 0")
    reader = readers.get(instrument.spectral.reader)
    tw, tf = template
    orders = rv_orders(instrument, config)
    v_lo, v_hi = ccf_grid_limits(config, vsini_kms)
    exp_rows, po_rows, without_bjd = [], [], []
    for exposure in exposures:
        base = exposure.path.name
        with fits.open(exposure.path, memmap=True) as hdul:
            bjd, found = exposure_bjd(hdul[0].header, instrument.header, exposure.time_utc)
        if not found:
            without_bjd.append(base)
        airmass = float(exposure.airmass)
        order_specs = []
        for order in orders:
            try:
                spectrum = reader.load_order(exposure.path, order, instrument.spectral, formula)
            except DataError:
                continue  # an unreadable or empty order is skipped
            if spectrum.wave_fit.size >= 50:
                order_specs.append((order, spectrum.wave_fit, spectrum.flux, spectrum.err))
        if not order_specs:
            raise DataError(f"{base}: no usable orders for RV measurement")
        result = measure_rv_exposure(
            order_specs,
            tw,
            tf,
            mask_windows=mask_windows,
            v_lo=v_lo,
            v_hi=v_hi,
            dv=float(config.ccf_dv),
            mcmc=mcmc,
            n_walkers=MCMC_WALKERS,
            n_steps=MCMC_STEPS,
            n_burn=MCMC_BURN,
            seed=seed,
            label=base,
        )
        obs_time = exposure.time_utc.strftime("%Y%m%dT%H%M%S")
        save_ccf_data(
            result,
            layout.ccf_path(obs_time, exposure.path),
            star_name=layout.target,
            instrument=layout.instrument,
            obs_time=obs_time,
            exposure=exposure.path,
            bjd=bjd,
            airmass=airmass,
        )
        exp_rows.append(
            (
                base, bjd, airmass, result.rv_kms, result.rv_err_kms, result.rv_master_ccf_kms,
                result.per_order_wmean_kms, result.per_order_wstd_kms, result.fwhm_kms,
                result.contrast, result.n_orders_used, result.acceptance,
            )
        )  # fmt: skip
        for item in result.per_order:
            po_rows.append(
                (
                    base, bjd, item["order"], item["rv"], item["rv_err"], item["weight"],
                    item["n_good"], item["contrast"],
                )
            )  # fmt: skip
    _warn_without_bjd(without_bjd, instrument.name)
    with open(layout.rv_table, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RV_TABLE_COLUMNS)
        writer.writerows(exp_rows)
    with open(layout.rv_per_order_table, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PER_ORDER_COLUMNS)
        writer.writerows(po_rows)
    return layout.rv_table
