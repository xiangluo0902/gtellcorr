"""Stellar-line masks and the per-order fit exclusion mask.

Convention: ``True`` = excluded from the telluric fit.

Stellar-line windows (measured on the master, :mod:`gtellcorr.preprocessing.stellar_lines`;
from the stellar template, :func:`build_stellar_linemask`; or user-supplied) are
placed on every exposure by its RV and written per exposure
(:func:`write_exposure_linemasks`). :func:`fit_exclusion` projects the per-exposure
windows, the activity windows, the bad regions and the strong-telluric windows onto
an order grid.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.preprocessing.windows import (
    C_KMS,
    Window,
    exclusion_pixels,
    normalise_windows,
    read_windows,
    shift_windows,
)

if TYPE_CHECKING:
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.preprocessing.config import StellarLineMask

__all__ = [
    "build_stellar_linemask",
    "fit_exclusion",
    "write_exposure_linemasks",
]


def build_stellar_linemask(
    template: tuple[Any, Any],
    config: StellarLineMask,
    *,
    wave_range_nm: tuple[float, float] | None = None,
) -> list[Window]:
    """Windows where the template flux is below ``depth_below``, padded by ``pad_kms``
    on each side and merged across gaps up to ``merge_gap_nm`` (in the template's frame)."""
    w = np.asarray(template[0], float)
    f = np.asarray(template[1], float)
    if wave_range_nm is not None:
        m = (w >= wave_range_nm[0]) & (w <= wave_range_nm[1])
        w, f = w[m], f[m]
    below = np.isfinite(f) & (f < config.depth_below)
    windows: list[Window] = []
    if below.any():
        idx = np.where(below)[0]
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            lo, hi = float(w[run[0]]), float(w[run[-1]])
            d = (config.pad_kms / C_KMS) * 0.5 * (lo + hi)
            windows.append((lo - d, hi + d))
    windows.sort()
    merged: list[Window] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1] + config.merge_gap_nm:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def write_exposure_linemasks(
    rv_rows: Sequence[Mapping[str, str]],
    windows: Iterable[Window],
    source: str,
    layout: RunLayout,
    *,
    shift: bool = True,
) -> int:
    """Write one topocentric mask per RV-table exposure; returns the count.

    Rest-frame *windows* are shifted by each exposure's RV; ``shift=False`` writes
    windows that are already topocentric (from a topocentric template) unchanged.
    """
    rest = list(windows)
    n = 0
    for row in rv_rows:
        exposure, rv = row["file"], float(row["rv_kms"])
        if shift:
            shifted = np.asarray(shift_windows(rest, rv), float).reshape(-1, 2)
            frame = (
                "reference frame: TOPOCENTRIC (observed) — REST windows shifted by "
                f"apparent stellar RV = {rv:+.4f} km/s (lambda_obs = lambda_rest*(1+rv/c))\n"
                f"rest-window source: {source}"
            )
        else:
            shifted = np.asarray(rest, float).reshape(-1, 2)
            frame = (
                "reference frame: TOPOCENTRIC (observed) — windows of a topocentric "
                f"template, not shifted\nwindow source: {source}"
            )
        header = (
            f"stellar line mask for {exposure}\n"
            f"{frame}\n"
            "columns: lo_nm hi_nm   (windows to EXCLUDE in the telluric fit)"
        )
        np.savetxt(layout.exposure_linemask(exposure), shifted, fmt="%.5f", header=header)
        n += 1
    return n


def missing_exposure_linemasks(rv_path: str | os.PathLike[str], layout: RunLayout) -> list[str]:
    """RV-table exposures without a per-exposure mask file."""
    with open(rv_path) as fh:
        names = [row["file"] for row in csv.DictReader(fh)]
    return [n for n in names if not layout.exposure_linemask(n).exists()]


def fit_exclusion(
    wave_nm: Any,
    *,
    stellar_mask_file: str | os.PathLike[str] | None = None,
    activity_windows: Iterable[Window] = (),
    bad_regions: Iterable[Window] = (),
    strong_windows: Any = None,
) -> NDArray[np.bool_]:
    """Exclusion mask of one order (True = exclude).

    ``stellar_mask_file``: the exposure's topocentric stellar-line mask (``mask``
    treatment). A missing file raises :class:`FileNotFoundError` rather than fitting
    the order without its stellar lines excluded.
    """
    w = np.asarray(wave_nm, float)
    excl = np.zeros(w.shape, bool)
    if stellar_mask_file is not None:
        path = Path(stellar_mask_file)
        if not path.exists():
            raise FileNotFoundError(f"stellar-line mask missing: {path}")
        excl |= exclusion_pixels(w, read_windows(path))
    activity = [(float(a), float(b)) for a, b in activity_windows]
    if activity:
        excl |= exclusion_pixels(w, activity)
    bad = [(float(a), float(b)) for a, b in bad_regions]
    if bad:
        excl |= exclusion_pixels(w, bad)
    strong = normalise_windows(strong_windows)
    if strong:
        excl |= exclusion_pixels(w, strong)
    return excl
