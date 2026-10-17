"""Stellar-line catalogue and rest-frame line mask measured on the master spectrum.

For each order of the rest-frame, normalised master, iSpec ``find_linemasks``
locates and fits the absorption lines and crossmatches them to the atomic line list;
lines of the selected elements (all if none are given) deeper than ``min_depth`` are
kept. The master contains the telluric lines, and some of them fall close enough to
an atomic line to be taken for it. Given the telluric model on the master's grid,
every line whose window the model absorbs in is classified (:func:`judge_line`): a
telluric feature is dropped, and a blend of a stellar and a telluric line is masked
or kept by the ``blend`` policy. Needs iSpec (``ispec_dir``) and its atomic line list.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from gtellcorr.preprocessing._ispec import import_ispec
from gtellcorr.preprocessing.windows import Window

__all__ = ["judge_line", "measure_stellar_lines", "write_stellar_lines"]

log = logging.getLogger(__name__)

Array = NDArray[np.float64]
BlendPolicy = Literal["off", "mask", "ratio", "keep"]
#: Model transmission below which the master is not divided by it (saturated cores,
#: excluded from the fit by the strong-telluric mask).
_MIN_TRANSMISSION = 0.2


def judge_line(
    line: Mapping[str, Any],
    wave_nm: Array,
    flux: Array,
    telluric: Array,
    *,
    min_depth: float,
    blend: BlendPolicy,
    blend_ratio: float,
    telluric_depth: float,
) -> dict[str, Any]:
    """Class of a measured line against the telluric model, and whether it is masked.

    *line* is a catalogue entry (``depth`` the iSpec depth, ``wave_obs_nm`` its centre,
    ``wave_base_nm``-``wave_top_nm`` its window); *wave_nm*, *flux* and *telluric* are
    the master's order and the model on the same grid. Classes: ``clean`` (model
    telluric depth in the window below *telluric_depth*; masked), ``telluric`` (the
    depth left after dividing by the model at the centre is below *min_depth*, or the
    model is below 0.2 there; not masked) and ``blend`` (masked by *blend*: always for
    ``mask``, never for ``keep``, and for ``ratio`` if the stellar equivalent width in
    the window, of the master divided by the model where the model is >= 0.2, is at
    least *blend_ratio* times the model's telluric one).
    """
    window = (wave_nm >= line["wave_base_nm"]) & (wave_nm <= line["wave_top_nm"])
    t = telluric[window]
    d_tell = float(1.0 - t.min()) if t.size else 0.0
    out: dict[str, Any] = {"telluric_depth": round(d_tell, 4)}
    if d_tell < telluric_depth:
        return {**out, "class": "clean", "masked": True}
    t_centre = float(np.interp(line["wave_obs_nm"], wave_nm, telluric))
    d_star = 1.0 - (1.0 - float(line["depth"])) / max(t_centre, _MIN_TRANSMISSION)
    out["stellar_depth"] = round(d_star, 4)
    if t_centre < _MIN_TRANSMISSION or d_star < min_depth:
        return {**out, "class": "telluric", "masked": False}
    step = np.gradient(wave_nm)[window]
    usable = t >= _MIN_TRANSMISSION
    ew_star = float(np.sum(np.clip(1.0 - flux[window][usable] / t[usable], 0, None) * step[usable]))
    ew_tell = float(np.sum((1.0 - t) * step))
    ratio = ew_star / ew_tell if ew_tell > 0 else np.inf
    out["ew_ratio"] = round(ratio, 4)
    masked = blend == "mask" or (blend == "ratio" and ratio >= blend_ratio)
    return {**out, "class": "blend", "masked": masked}


def measure_stellar_lines(
    master: Mapping[int, Mapping[str, Any]],
    *,
    elements: Iterable[str],
    min_depth: float,
    ispec_dir: str | os.PathLike[str] | None,
    linelist: str,
    smooth_r: float = 70000.0,
    max_atomic_wave_diff: float = 0.005,
    telluric: Callable[[Array], Array] | None = None,
    blend: BlendPolicy = "off",
    blend_ratio: float = 0.3,
    telluric_depth: float = 0.02,
) -> tuple[dict[str, list[dict[str, Any]]], list[Window]]:
    """Measure the master's stellar lines -> ``(per-element catalogue, merged rest windows)``.

    ``linelist`` names an iSpec transition list (``IspecSettings.linelist``); its lines
    with theoretical depth >= 0.01 are the candidates. The search runs on the master
    smoothed to ``smooth_r`` with the continuum fixed at 1; a line must lie within
    ``max_atomic_wave_diff`` nm of its atomic line. Orders with fewer than 100 usable
    pixels, without candidate lines, or where iSpec fails are skipped (a warning for
    the last). *elements* empty keeps every element.

    With *blend* other than ``off``, *telluric* maps the master's rest wavelengths to
    the model telluric transmission on the master, and each line is judged by
    :func:`judge_line` (its class and numbers are added to its catalogue entry). The
    windows are the masked lines' ``wave_base``-``wave_top`` ranges, merged where they
    overlap.
    """
    if blend != "off" and telluric is None:
        raise ValueError(f"blend = {blend!r} needs the telluric model of the master")
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
        tell = telluric(w[m]) if blend != "off" and telluric is not None else None
        for line in lm:
            element = str(line["element"]).strip()
            symbol = element.split()[0] if element else ""
            if wanted and symbol not in wanted:
                continue
            entry: dict[str, Any] = {
                "element": element,
                "wave_linelist_nm": round(float(line["wave_nm"]), 5),
                "wave_obs_nm": round(float(line["mu"]), 5),
                "ew_mA": round(float(line["ew"]), 3),
                "fwhm_nm": round(float(line["fwhm"]), 5),
                "depth": round(float(line["depth"]), 4),
                "wave_base_nm": round(float(line["wave_base"]), 5),
                "wave_top_nm": round(float(line["wave_top"]), 5),
            }
            if tell is not None:
                entry.update(
                    judge_line(
                        entry, w[m], f[m], tell, min_depth=min_depth, blend=blend,
                        blend_ratio=blend_ratio, telluric_depth=telluric_depth,
                    )
                )  # fmt: skip
            per[symbol].append(entry)
            if entry.get("masked", True):
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
    names = list(elements) or "all"
    header = (
        f"stellar line mask (REST frame) for {target} — measured on the master with iSpec\n"
        f"find_linemasks; elements {names}, depth>={min_depth}\n"
        f"master: {os.path.abspath(str(master_path))}\n"
        "columns: lo_nm hi_nm  (wave_base..wave_top endpoints; shift by RV → observed frame)"
    )
    np.savetxt(
        rest_path, np.asarray(list(windows), float).reshape(-1, 2), fmt="%.5f", header=header
    )
