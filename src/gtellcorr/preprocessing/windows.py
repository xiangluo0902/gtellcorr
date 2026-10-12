"""Wavelength windows ``[(lo_nm, hi_nm), ...]``: padding, Doppler shift, merging, pixels.

Windows are closed intervals in nm, in whatever frame and medium the caller uses.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "C_KMS",
    "Window",
    "boolean_mask_windows",
    "exclusion_pixels",
    "good_pixels",
    "merge_windows",
    "normalise_windows",
    "pad_window",
    "read_windows",
    "shift_windows",
]

#: Speed of light [km/s].
C_KMS = 299792.458
#: A wavelength window ``(lo_nm, hi_nm)``.
Window = tuple[float, float]


def pad_window(lo: float, hi: float, pad_kms: float) -> Window:
    """Widen ``(lo, hi)`` by ``pad_kms`` on each side (velocity -> nm at the centre)."""
    if pad_kms <= 0:
        return (lo, hi)
    d = (pad_kms / C_KMS) * 0.5 * (lo + hi)
    return (lo - d, hi + d)


def shift_windows(windows: Iterable[Window], rv_kms: float) -> list[Window]:
    """Rest -> observed frame: ``lambda_obs = lambda_rest * (1 + rv/c)``."""
    fac = 1.0 + rv_kms / C_KMS
    return [(lo * fac, hi * fac) for lo, hi in windows]


def exclusion_pixels(wave_nm: Any, windows: Iterable[Window]) -> NDArray[np.bool_]:
    """True where ``wave_nm`` lies inside any window (ends included)."""
    w = np.asarray(wave_nm, float)
    m = np.zeros(w.shape, bool)
    for lo, hi in windows:
        m |= (w >= lo) & (w <= hi)
    return m


def good_pixels(wave_nm: Any, windows: Iterable[Window]) -> NDArray[np.bool_]:
    """True where ``wave_nm`` lies outside every window."""
    w = np.asarray(wave_nm, float)
    keep = np.ones(w.shape, bool)
    for lo, hi in windows:
        keep &= ~((w >= lo) & (w <= hi))
    return keep


def normalise_windows(windows: Any) -> list[Window]:
    """Windows as a list of ``(lo, hi)`` tuples (``None`` / empty -> ``[]``)."""
    if windows is None:
        return []
    arr = np.asarray(windows, float)
    if arr.size == 0:
        return []
    return [(float(row[0]), float(row[1])) for row in arr.reshape(-1, 2)]


def merge_windows(
    windows: Iterable[Sequence[float]], *, merge_gap_nm: float = 0.0, min_width_nm: float = 0.0
) -> list[Window]:
    """Sort, merge (gaps <= ``merge_gap_nm``) and drop windows narrower than ``min_width_nm``."""
    clean = []
    for lo, hi in windows:
        lo_f, hi_f = float(lo), float(hi)
        if np.isfinite(lo_f) and np.isfinite(hi_f):
            clean.append((min(lo_f, hi_f), max(lo_f, hi_f)))
    if not clean:
        return []
    clean.sort()
    merged = [clean[0]]
    gap = float(merge_gap_nm)
    for lo_f, hi_f in clean[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo_f <= prev_hi + gap:
            merged[-1] = (prev_lo, max(prev_hi, hi_f))
        else:
            merged.append((lo_f, hi_f))
    min_width = float(min_width_nm)
    return [(lo_f, hi_f) for lo_f, hi_f in merged if (hi_f - lo_f) >= min_width]


def boolean_mask_windows(
    wave_nm: Any,
    mask: Any,
    *,
    pad_nm: float = 0.0,
    merge_gap_nm: float = 0.0,
    min_width_nm: float = 0.0,
) -> list[Window]:
    """Contiguous runs of a boolean mask on a wavelength grid -> merged windows."""
    wave = np.asarray(wave_nm, float)
    flag = np.asarray(mask, bool)
    idx = np.flatnonzero(np.isfinite(wave) & flag)
    if idx.size == 0:
        return []
    chunks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    pad = float(pad_nm)
    windows = []
    for chunk in chunks:
        lo = float(np.nanmin(wave[chunk])) - pad
        hi = float(np.nanmax(wave[chunk])) + pad
        if hi >= lo:
            windows.append((lo, hi))
    return merge_windows(windows, merge_gap_nm=merge_gap_nm, min_width_nm=min_width_nm)


def read_windows(path: str | os.PathLike[str], *, ordered: bool = False) -> list[Window]:
    """Two-column ``lo_nm hi_nm`` file (``#`` comments) -> windows.

    Only the first two columns are read. ``ordered`` returns ``(min, max)`` of each
    row (user RV masks); otherwise rows are returned as written (line and strong
    masks).
    """
    windows = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            lo, hi = (float(v) for v in line.split()[:2])
            windows.append((min(lo, hi), max(lo, hi)) if ordered else (lo, hi))
    return windows
