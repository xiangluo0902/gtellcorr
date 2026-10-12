"""Stellar-line catalogue and rest-frame line mask measured on the master spectrum.

For each order of the rest-frame, normalised master, iSpec ``find_linemasks``
locates and fits the absorption lines and crossmatches them to the atomic line list;
lines of the selected elements deeper than ``min_depth`` are kept. Telluric
features do not crossmatch a stellar atomic line and are therefore left out.
Needs iSpec (``ispec_dir``) and its atomic line list.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gtellcorr.preprocessing._ispec import import_ispec
from gtellcorr.preprocessing.windows import Window

__all__ = ["measure_stellar_lines", "write_stellar_lines"]

log = logging.getLogger(__name__)


def measure_stellar_lines(
    master: Mapping[int, Mapping[str, Any]],
    *,
    elements: Iterable[str],
    min_depth: float,
    ispec_dir: str | os.PathLike[str] | None,
    linelist: str,
    smooth_r: float = 70000.0,
    max_atomic_wave_diff: float = 0.005,
) -> tuple[dict[str, list[dict[str, Any]]], list[Window]]:
    """Measure the master's stellar lines -> ``(per-element catalogue, merged rest windows)``.

    ``linelist`` names an iSpec transition list (``IspecSettings.linelist``); its lines
    with theoretical depth >= 0.01 are the candidates. The search runs on the master
    smoothed to ``smooth_r`` with the continuum fixed at 1; a line must lie within
    ``max_atomic_wave_diff`` nm of its atomic line. Orders with fewer than 100 usable
    pixels, without candidate lines, or where iSpec fails are skipped (a warning for
    the last). The windows are the lines' ``wave_base``-``wave_top`` ranges, merged
    where they overlap.
    """
    wanted = set(elements)
    ispec = import_ispec(ispec_dir, fork=True)
    llf = os.path.join(
        str(Path(str(ispec_dir)).expanduser()),
        "input", "linelists", "transitions", linelist, "atomic_lines.tsv",
    )  # fmt: skip
    per: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    windows: list[Window] = []
    for o in sorted(master):
        w = np.asarray(master[o]["wave_nm"], float)
        f = np.asarray(master[o]["flux"], float)
        e = np.asarray(master[o].get("err", np.ones_like(f)), float)
        m = np.isfinite(w) & np.isfinite(f) & (f > 0)
        if m.sum() < 100:
            continue
        atomic = ispec.read_atomic_linelist(
            llf, wave_base=float(w[m].min()), wave_top=float(w[m].max())
        )
        atomic = atomic[atomic["theoretical_depth"] >= 0.01]
        if len(atomic) == 0:
            continue
        spec = ispec.create_spectrum_structure(w[m])
        spec["flux"] = f[m]
        spec["err"] = e[m]
        cont = ispec.fit_continuum(spec, fixed_value=1.0, model="Fixed value")
        smoothed = ispec.convolve_spectrum(spec, smooth_r)
        try:
            lm = ispec.find_linemasks(
                spec,
                cont,
                atomic_linelist=atomic,
                max_atomic_wave_diff=max_atomic_wave_diff,
                minimum_depth=min_depth,
                maximum_depth=1.0,
                smoothed_spectrum=smoothed,
                discard_gaussian=False,
                discard_voigt=True,
                closest_match=False,
            )
        except Exception as exc:  # an order where iSpec fails is skipped
            log.warning("order %s: find_linemasks failed (%s)", o, type(exc).__name__)
            continue
        lm = lm[(lm["wave_nm"] != 0) & (lm["ew"] != 0)]
        for line in lm:
            element = str(line["element"]).strip()
            symbol = element.split()[0] if element else ""
            if symbol not in wanted:
                continue
            per[symbol].append(
                {
                    "element": element,
                    "wave_linelist_nm": round(float(line["wave_nm"]), 5),
                    "wave_obs_nm": round(float(line["mu"]), 5),
                    "ew_mA": round(float(line["ew"]), 3),
                    "fwhm_nm": round(float(line["fwhm"]), 5),
                    "depth": round(float(line["depth"]), 4),
                    "wave_base_nm": round(float(line["wave_base"]), 5),
                    "wave_top_nm": round(float(line["wave_top"]), 5),
                }
            )
            windows.append((float(line["wave_base"]), float(line["wave_top"])))
    windows.sort()
    merged: list[Window] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return {k: per[k] for k in sorted(per)}, merged


def write_stellar_lines(
    catalogue: Mapping[str, Any],
    windows: Iterable[Window],
    *,
    catalog_path: Path,
    rest_path: Path,
    target: str,
    elements: Iterable[str],
    min_depth: float,
    master_path: str | os.PathLike[str],
) -> None:
    """Write the JSON catalogue (per element) and the rest-frame mask (``lo_nm hi_nm``
    lines under a commented header)."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as fh:
        json.dump(catalogue, fh, indent=1)
    header = (
        f"stellar line mask (REST frame) for {target} — measured on the master with iSpec\n"
        f"find_linemasks; elements {list(elements)}, depth>={min_depth}\n"
        f"master: {os.path.abspath(str(master_path))}\n"
        "columns: lo_nm hi_nm  (wave_base..wave_top endpoints; shift by RV → observed frame)"
    )
    np.savetxt(
        rest_path, np.asarray(list(windows), float).reshape(-1, 2), fmt="%.5f", header=header
    )
