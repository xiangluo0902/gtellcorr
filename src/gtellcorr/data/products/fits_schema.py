"""Exposure-level result FITS (schema ``S2DIMGV2``).

One file per exposure holds:

* image HDUs ``WAVE_FIT``, ``WAVE_INPUT``, ``FLUX_OBS``, ``ERR``, ``TELL_MODEL``,
  ``FLUX_CORRECTED``, ``ERR_CORRECTED``, ``CONTINUUM``, ``STELLAR_MODEL`` (float64,
  NaN-padded) and ``EXCLUSION`` (uint8, 1 = excluded, padding excluded), one row per
  order;
* only when a wavelength offset was fitted: ``WAVE_FIT_CORR``
  and ``WAVE_INPUT_CORR``, the same pixels' wavelengths anchored on the telluric lines
  (``wave / (1 + v / c)``), the uncorrected grid for orders without an offset;
* ``ORDER_SUMMARY`` (per-order parameters; ``row_index`` maps rows to orders),
  ``FIT_SUMMARY`` (exposure-level fit) and ``RUN_METADATA`` key/value tables;
* optional cumulative PCA models ``tell_principal_model_###`` (``TPCA_N``, ``TPCA_MODE``).

Primary-header cards record the provenance used to decide whether a product can be
reused: ``GTC_VERS`` (package version), ``CFG_HASH`` (fit configuration),
``RES_FP`` (LBLRTM resource fingerprint) and ``IN_HASH`` (input spectrum file).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from gtellcorr.errors import DataError

__all__ = [
    "CORRECTED_WAVE_HDU_NAMES",
    "EXPOSURE_FITS_PATTERNS",
    "PROVENANCE_CARDS",
    "S2D_IMAGE_HDU_NAMES",
    "S2D_SCHEMA_NAME",
    "discover_result_orders",
    "load_exposure_order",
    "read_order_timeseries",
    "read_provenance",
    "update_order_columns",
    "write_exposure_fits",
]

FITS_PCA_MODEL_PREFIX = "tell_principal_model"
FITS_PCA_N_KEY = "TPCA_N"
FITS_PCA_MODE_KEY = "TPCA_MODE"
FITS_PCA_MODE_VALUE = "cumulative_model"

S2D_SCHEMA_NAME = "S2DIMGV2"
S2D_IMAGE_HDU_NAMES = (
    "WAVE_FIT", "WAVE_INPUT", "FLUX_OBS", "ERR", "TELL_MODEL", "FLUX_CORRECTED",
    "ERR_CORRECTED", "CONTINUUM", "STELLAR_MODEL", "EXCLUSION",
)  # fmt: skip
_FLOAT_HDUS = tuple(n for n in S2D_IMAGE_HDU_NAMES if n != "EXCLUSION")
#: Optional image HDUs, written only when a wavelength offset was fitted.
CORRECTED_WAVE_HDU_NAMES = ("WAVE_FIT_CORR", "WAVE_INPUT_CORR")
EXPOSURE_FITS_PATTERNS = ("*_single_order_corrected.fits", "*_joint_corrected_all_orders.fits")

#: ROWMAP card of every image HDU: image row i holds the order whose row_index is i.
_ROWMAP = ("ORDER_SUMMARY.row_index", "order of each image row")

#: Provenance header cards (name -> comment).
PROVENANCE_CARDS = {
    "GTC_VERS": "gtellcorr version",
    "CFG_HASH": "SHA-256 of the fit configuration",
    "RES_FP": "LBLRTM resource fingerprint",
    "IN_HASH": "SHA-256 of the input spectrum file",
}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _string_column(name: str, values: Iterable[Any], *, min_width: int = 1) -> fits.Column:
    """Fixed-width string column wide enough for every value."""
    vals = np.asarray([str(v) for v in values])
    width = max([min_width] + [len(v) for v in vals])
    return fits.Column(name=name, format=f"{width}A", array=vals)


def _cell(value: Any) -> str:
    """A table cell: containers as JSON, ``None`` as ``""``."""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return "" if value is None else str(value)


def _float(value: Any) -> float:
    """``None`` -> NaN."""
    return float(value) if value is not None else np.nan


def _int(value: Any) -> int:
    """``None`` -> -1."""
    return int(value) if value is not None else -1


def _flag(value: Any) -> str:
    """``true`` / ``false``, ``""`` for ``None``."""
    return "" if value is None else ("true" if value else "false")


def _coefficients(values: Any) -> str:
    """JSON list of offset coefficients (``[]`` = no offset)."""
    return json.dumps([] if values is None else [float(v) for v in values])


def _fit_summary_hdu(fit_summary: Mapping[str, Any] | None) -> fits.BinTableHDU:
    """``FIT_SUMMARY``: parameter/value rows; nested dicts become ``key.subkey``."""
    rows = []
    for key, value in sorted((fit_summary or {}).items()):
        if isinstance(value, dict):
            for subkey, subvalue in sorted(value.items()):
                rows.append((f"{key}.{subkey}", subvalue))
        else:
            rows.append((str(key), value))
    return fits.BinTableHDU.from_columns(
        [
            _string_column("parameter", [r[0] for r in rows]),
            _string_column("value", [_cell(r[1]) for r in rows]),
        ],
        name="FIT_SUMMARY",
    )


def _metadata_hdu(name: str, metadata: Mapping[str, Any] | None) -> fits.BinTableHDU:
    """A key/value table HDU."""
    metadata = metadata or {}
    return fits.BinTableHDU.from_columns(
        [
            _string_column("key", sorted(metadata)),
            _string_column("value", [_cell(metadata[k]) for k in sorted(metadata)]),
        ],
        name=name,
    )


def _summary_rows(
    results: Sequence[Any], metadata: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``ORDER_SUMMARY`` rows, one per order result (missing values as NaN / -1)."""
    rows = []
    for row_index, r in enumerate(results):
        meta = metadata.get(int(r.order), {})
        x = np.asarray(r.wave_fit, float)
        finite = np.isfinite(x)
        relcol = getattr(r, "relcol", {}) or {}
        relcol_err = getattr(r, "relcol_err", {}) or {}
        rows.append(
            {
                "row_index": int(row_index),
                "order": int(r.order),
                "used_in_fit": bool(
                    meta.get(
                        "used_in_fit", str(getattr(r, "status", "")).startswith(("ok", "joint"))
                    )
                ),
                "status": str(getattr(r, "status", "")),
                "wave_lo": float(np.nanmin(x[finite])) if np.any(finite) else np.nan,
                "wave_hi": float(np.nanmax(x[finite])) if np.any(finite) else np.nan,
                "n_pixels": int(x.size),
                "molecules": json.dumps(list(relcol.keys())),
                "treatment": str(getattr(r, "stellar_treatment", "") or ""),
                "relcol": json.dumps({m: float(v) for m, v in relcol.items()}),
                "relcol_err": json.dumps({m: float(v) for m, v in relcol_err.items()}),
                "resolution": _float(getattr(r, "resolution", None)),
                "resolution_err": _float(getattr(r, "resolution_err", None)),
                "resolution_value_source": str(meta.get("resolution_value_source", "unknown")),
                "lsf_kind": str(getattr(r, "lsf_kind", "") or ""),
                "lsf_shape": _float(getattr(r, "lsf_shape", None)),
                "lsf_shape_err": _float(getattr(r, "lsf_shape_err", None)),
                "lsf_shape_value_source": str(meta.get("lsf_shape_value_source", "unknown")),
                "chi2_reduced": _float(getattr(r, "chi2_reduced", None)),
                "dof_mask_removed": _int(getattr(r, "dof_mask_removed", None)),
                "chi2_reduced_order": _float(getattr(r, "chi2_reduced_order", None)),
                "dof_mask_removed_order": _int(getattr(r, "dof_mask_removed_order", None)),
                "chi2_reduced_joint": _float(getattr(r, "chi2_reduced_joint", None)),
                "dof_mask_removed_joint": _int(getattr(r, "dof_mask_removed_joint", None)),
                "masked_frac": (
                    float(np.asarray(r.exclusion, bool).mean()) if r.exclusion is not None else 0.0
                ),
                "wave_shift": _coefficients(getattr(r, "wave_shift", None)),
                "wave_shift_err": _coefficients(getattr(r, "wave_shift_err", None)),
                "wave_shift_value_source": str(meta.get("wave_shift_value_source", "not_fitted")),
                "converged": _flag(getattr(r, "converged", None)),
                "n_eval": _int(getattr(r, "n_eval", None)),
            }
        )
    return rows


_TABLE_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("row_index", "J"), ("order", "J"), ("used_in_fit", "L"), ("status", None),
    ("wave_lo", "D"), ("wave_hi", "D"), ("n_pixels", "J"), ("molecules", None),
    ("treatment", None), ("relcol", None), ("relcol_err", None), ("resolution", "D"),
    ("resolution_err", "D"), ("resolution_value_source", None), ("lsf_kind", None),
    ("lsf_shape", "D"), ("lsf_shape_err", "D"), ("lsf_shape_value_source", None),
    ("chi2_reduced", "D"), ("dof_mask_removed", "J"), ("chi2_reduced_order", "D"),
    ("dof_mask_removed_order", "J"), ("chi2_reduced_joint", "D"),
    ("dof_mask_removed_joint", "J"), ("masked_frac", "D"),
    # wavelength-offset coefficients [km/s] as JSON (absent from older result files)
    ("wave_shift", None), ("wave_shift_err", None), ("wave_shift_value_source", None),
    # optimiser outcome: "true" / "false" / "" and evaluations, -1 if not fitted
    # (absent from older result files)
    ("converged", None), ("n_eval", "J"),
)  # fmt: skip


def _order_summary_hdu(rows: Sequence[Mapping[str, Any]]) -> fits.BinTableHDU:
    """``ORDER_SUMMARY`` table from the rows (string columns sized to fit)."""
    columns = []
    for name, fmt in _TABLE_COLUMNS:
        values = [row[name] for row in rows]
        columns.append(
            _string_column(name, values)
            if fmt is None
            else fits.Column(name, format=fmt, array=values)
        )
    return fits.BinTableHDU.from_columns(columns, name="ORDER_SUMMARY")


def _row_array(result: Any, attribute: str, *, n_pixels: int, optional: bool) -> NDArray[Any]:
    """One result array, checked against the order length (NaN if optional and absent)."""
    values = getattr(result, attribute, None)
    if values is None:
        if optional:
            return np.full(n_pixels, np.nan, dtype=np.float64)
        raise ValueError(f"order {int(result.order)}: missing required result array {attribute!r}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size != int(n_pixels):
        raise ValueError(
            f"order {int(result.order)} {attribute}: shape {arr.shape} does not match "
            f"wave_fit length {int(n_pixels)}"
        )
    return arr


_ATTRIBUTES = {
    "WAVE_FIT": "wave_fit", "WAVE_INPUT": "wave_input", "FLUX_OBS": "flux_obs", "ERR": "err",
    "TELL_MODEL": "tell_model", "FLUX_CORRECTED": "flux_corrected",
    "ERR_CORRECTED": "err_corrected", "CONTINUUM": "continuum", "STELLAR_MODEL": "stellar_model",
}  # fmt: skip
_OPTIONAL = {"ERR_CORRECTED", "CONTINUUM", "STELLAR_MODEL"}


def _stack_images(
    results: Sequence[Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, NDArray[Any]], int]:
    """The image HDUs as ``(n_order, n_pixel_max)`` arrays, and ``n_pixel_max``."""
    n_pix = max(int(row["n_pixels"]) for row in rows)
    images = {n: np.full((len(results), n_pix), np.nan, dtype=np.float64) for n in _FLOAT_HDUS}
    images["EXCLUSION"] = np.ones((len(results), n_pix), dtype=np.uint8)
    for i, result in enumerate(results):
        n = int(rows[i]["n_pixels"])
        for hdu, attribute in _ATTRIBUTES.items():
            images[hdu][i, :n] = _row_array(
                result, attribute, n_pixels=n, optional=hdu in _OPTIONAL
            )
        exclusion = getattr(result, "exclusion", None)
        exclusion = np.zeros(n, dtype=bool) if exclusion is None else np.asarray(exclusion, bool)
        if exclusion.ndim != 1 or exclusion.size != n:
            raise ValueError(
                f"order {int(result.order)} exclusion: expected length {n}, got {exclusion.shape}"
            )
        images["EXCLUSION"][i, :n] = exclusion.astype(np.uint8)
    return images, n_pix


def _corrected_wave_images(
    results: Sequence[Any], rows: Sequence[Mapping[str, Any]], n_pix: int
) -> dict[str, NDArray[Any]]:
    """``WAVE_FIT_CORR`` / ``WAVE_INPUT_CORR``, or nothing when no order has an offset.

    The result's ``wave_fit_corr`` / ``wave_input_corr`` are used where present; an
    order without an offset keeps its uncorrected grid.
    """
    if all(getattr(r, "wave_fit_corr", None) is None for r in results):
        return {}
    images = {
        n: np.full((len(results), n_pix), np.nan, dtype=np.float64)
        for n in CORRECTED_WAVE_HDU_NAMES
    }
    for i, result in enumerate(results):
        n = int(rows[i]["n_pixels"])
        for hdu, corrected, original in (
            ("WAVE_FIT_CORR", "wave_fit_corr", "wave_fit"),
            ("WAVE_INPUT_CORR", "wave_input_corr", "wave_input"),
        ):
            attribute = original if getattr(result, corrected, None) is None else corrected
            images[hdu][i, :n] = _row_array(result, attribute, n_pixels=n, optional=False)
    return images


_IMAGE_CARDS: dict[str, dict[str, tuple[str, str]]] = {
    "WAVE_INPUT": {
        "BUNIT": ("nm", "wavelength unit"),
        "GTFRAME": ("delivered", "frame/medium as delivered by the input data"),
        "GTDESC": ("delivered wavelength grid", "HDU meaning"),
    },
    "WAVE_FIT": {
        "BUNIT": ("nm", "wavelength unit"),
        "GTFRAME": ("air_topocentric", "telluric fitting wavelength frame"),
        "GTDESC": ("telluric fit wavelength grid", "HDU meaning"),
    },
    "WAVE_FIT_CORR": {
        "BUNIT": ("nm", "wavelength unit"),
        "GTFRAME": ("air_topocentric", "telluric fitting wavelength frame"),
        "GTDESC": ("WAVE_FIT anchored on telluric lines", "HDU meaning"),
        "GTFORM": ("WAVE_FIT/(1+v/c), v from wave_shift", "ORDER_SUMMARY.wave_shift"),
        "GTNOTE": ("includes line-list systematics", "uncorrected if not fitted"),
    },
    "WAVE_INPUT_CORR": {
        "BUNIT": ("nm", "wavelength unit"),
        "GTFRAME": ("delivered", "frame/medium as delivered by the input data"),
        "GTDESC": ("WAVE_INPUT anchored on telluric lines", "HDU meaning"),
        "GTFORM": ("WAVE_INPUT/(1+v/c), v on WAVE_FIT", "ORDER_SUMMARY.wave_shift"),
        "GTNOTE": ("includes line-list systematics", "uncorrected if not fitted"),
    },
    "FLUX_OBS": {
        "GTDESC": ("observed flux; not resampled", "HDU meaning"),
        "GTWAVE": ("WAVE_INPUT,WAVE_FIT", "same pixels; fit frame=WAVE_FIT"),
    },
    "ERR": {
        "GTDESC": ("observed flux 1-sigma error", "HDU meaning"),
        "GTWAVE": ("WAVE_INPUT,WAVE_FIT", "same pixels; fit frame=WAVE_FIT"),
    },
    "TELL_MODEL": {
        "GTDESC": ("best-fit telluric model", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "wavelength grid for this product"),
    },
    "FLUX_CORRECTED": {
        "GTDESC": ("FLUX_OBS/TELL_MODEL", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "science wavelength grid"),
    },
    "ERR_CORRECTED": {
        "GTDESC": ("ERR/TELL_MODEL", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "science wavelength grid"),
    },
    "CONTINUUM": {
        "GTDESC": ("best-fit continuum on WAVE_FIT", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "wavelength grid for this product"),
    },
    "STELLAR_MODEL": {
        "GTDESC": ("stellar model; NaN if absent", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "wavelength grid for this product"),
    },
    "EXCLUSION": {
        "BUNIT": ("bool_uint8", "1=excluded from fit, 0=available"),
        "GTDESC": ("union fit-exclusion mask", "HDU meaning"),
        "GTWAVE": ("WAVE_FIT", "wavelength grid for this mask"),
    },
}


def _image_hdu(name: str, image: NDArray[Any]) -> fits.ImageHDU:
    """Image HDU with its descriptive cards."""
    arr = image.astype(np.uint8 if name == "EXCLUSION" else np.float64, copy=False)
    hdu = fits.ImageHDU(data=arr, name=name)
    hdu.header["ROWMAP"] = _ROWMAP
    for key, (value, comment) in _IMAGE_CARDS.get(name, {}).items():
        hdu.header[key] = (value, comment)
    return hdu


def write_exposure_fits(
    path: str | os.PathLike[str],
    order_results: Iterable[Any],
    *,
    exposure: str,
    instrument: str,
    correction_mode: str,
    fit_summary: Mapping[str, Any] | None = None,
    order_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write one exposure's results.

    ``correction_mode`` is ``single_order`` or ``joint_orders``; ``provenance`` fills
    the :data:`PROVENANCE_CARDS`.
    """
    if correction_mode not in ("single_order", "joint_orders"):
        raise ValueError(f"unsupported correction_mode: {correction_mode!r}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = sorted(order_results, key=lambda r: int(r.order))
    if not results:
        raise ValueError("write_exposure_fits requires at least one order result")
    rows = _summary_rows(results, order_metadata or {})
    images, n_pix = _stack_images(results, rows)
    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["GTSCHEMA"] = (S2D_SCHEMA_NAME, "gtellcorr result file format")
    hdr["CORRMODE"] = (correction_mode, "telluric correction mode")
    hdr["INSTRUME"] = (str(instrument), "instrument")
    hdr["EXPOSURE"] = (str(exposure)[:68], "input spectrum file")
    hdr["NORDER"] = (len(results), "orders in this file (image rows)")
    hdr["NFITORD"] = (sum(bool(row["used_in_fit"]) for row in rows), "orders used in the fit")
    hdr["NPIXMAX"] = (int(n_pix), "pixels per image row (longest order)")
    summary = fit_summary or {}
    if summary.get("chi2") is not None:
        hdr["CHI2"] = (float(summary["chi2"]), "chi-square of the fit")
    if summary.get("chi2_reduced") is not None:
        hdr["RCHI2"] = (float(summary["chi2_reduced"]), "reduced chi-square of the fit")
    if summary.get("dof_mask_removed") is not None:
        hdr["DOFMR"] = (int(summary["dof_mask_removed"]), "unmasked pixels minus fitted parameters")
    for key, value in (provenance or {}).items():
        if key not in PROVENANCE_CARDS:
            raise ValueError(f"internal error: unknown header card {key!r}")
        # a SHA-256 value leaves no room for a comment
        hdr[key] = (str(value), PROVENANCE_CARDS[key]) if key == "GTC_VERS" else str(value)
    hdus: list[Any] = [primary]
    hdus.extend(_image_hdu(n, images[n]) for n in S2D_IMAGE_HDU_NAMES)
    hdus.extend(
        _image_hdu(n, image) for n, image in _corrected_wave_images(results, rows, n_pix).items()
    )
    hdus.extend(
        [
            _order_summary_hdu(rows),
            _fit_summary_hdu(summary),
            _metadata_hdu("RUN_METADATA", run_metadata),
        ]
    )
    fits.HDUList(hdus).writeto(out, overwrite=True)
    return out


def read_provenance(path: str | os.PathLike[str]) -> dict[str, str]:
    """The provenance cards present in a result's primary header."""
    header = fits.getheader(path)
    return {k: str(header[k]) for k in PROVENANCE_CARDS if k in header}


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _nan_to_none(values: NDArray[Any]) -> NDArray[Any] | None:
    """``None`` for an all-NaN row (an optional product that was not written)."""
    arr = np.asarray(values, float)
    return None if arr.size and np.all(~np.isfinite(arr)) else arr


def _json_cell(value: Any, default: Any) -> Any:
    """A JSON table cell, or *default* if it does not parse."""
    try:
        return json.loads(str(value).strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _table_to_dict(hdu: Any) -> dict[str, Any]:
    """A key/value (or parameter/value) table as a dict."""
    if hdu is None or hdu.data is None:
        return {}
    names = list(hdu.columns.names)
    key_name = "key" if "key" in names else "parameter"
    out = {}
    for row in hdu.data:
        key = str(row[key_name]).strip()
        value = str(row["value"]).strip()
        out[key] = _json_cell(value, value)
    return out


def _is_s2d(hdul: Any) -> bool:
    """Whether an open FITS file is a gtellcorr exposure result."""
    return "ORDER_SUMMARY" in hdul and all(
        n in hdul for n in ("WAVE_FIT", "TELL_MODEL", "EXCLUSION")
    )


def _summary_row(hdul: Any, order: int) -> Any:
    """``ORDER_SUMMARY`` row of *order*, or ``None``."""
    data = hdul["ORDER_SUMMARY"].data if "ORDER_SUMMARY" in hdul else None
    if data is None:
        return None
    matches = [row for row in data if int(row["order"]) == int(order)]
    return matches[0] if matches else None


def _row_index(row: Any) -> int:
    """Image row of an order (older files without ``row_index``: the order number)."""
    return int(row["row_index"]) if "row_index" in (row.array.names or []) else int(row["order"])


def _row_pixels(row: Any, wave_row: NDArray[Any]) -> int:
    """Number of real (unpadded) pixels of an order."""
    if "n_pixels" in (row.array.names or []):
        n = int(row["n_pixels"])
        if n > 0:
            return min(n, len(wave_row))
    return int(np.count_nonzero(np.isfinite(np.asarray(wave_row, float))))


def _row_values(
    hdul: Any, name: str, index: int, n: int, *, optional: bool = False
) -> NDArray[Any] | None:
    """The first *n* pixels of row *index* of image *name*."""
    if name not in hdul:
        if optional:
            return None
        raise KeyError(f"missing required FITS image HDU {name!r}")
    arr = np.asarray(hdul[name].data)
    if arr.ndim != 2 or not 0 <= index < arr.shape[0]:
        raise ValueError(f"{name}: row {index} outside image shape {arr.shape}")
    return np.asarray(arr[index, :n], float)


def _pca_name(index: int) -> str:
    """HDU name of the cumulative PCA model with *index* components."""
    return f"{FITS_PCA_MODEL_PREFIX}_{int(index):03d}"


def _pca_index(name: str) -> int | None:
    """Component count of a PCA model HDU name, ``None`` for other names."""
    prefix = f"{FITS_PCA_MODEL_PREFIX}_"
    lower = str(name).lower()
    if not lower.startswith(prefix):
        return None
    try:
        return int(lower[len(prefix) :])
    except ValueError:
        return None


def _summary_fields(row: Any) -> dict[str, Any]:
    """Typed fields of an ``ORDER_SUMMARY`` row (-1 counts become ``None``)."""

    def opt_int(name: str) -> int | None:
        value = int(row[name])
        return value if value >= 0 else None

    return {
        "used_in_fit": bool(row["used_in_fit"]),
        "status": str(row["status"]).strip(),
        "wave_lo": float(row["wave_lo"]),
        "wave_hi": float(row["wave_hi"]),
        "molecules": _json_cell(row["molecules"], []),
        "treatment": str(row["treatment"]).strip(),
        "relcol": _json_cell(row["relcol"], {}),
        "relcol_err": _json_cell(row["relcol_err"], {}),
        "resolution": float(row["resolution"]),
        "resolution_err": float(row["resolution_err"]),
        "resolution_value_source": str(row["resolution_value_source"]).strip(),
        "lsf_kind": str(row["lsf_kind"]).strip(),
        "lsf_shape": float(row["lsf_shape"]),
        "lsf_shape_err": float(row["lsf_shape_err"]),
        "lsf_shape_value_source": str(row["lsf_shape_value_source"]).strip(),
        "chi2_reduced": float(row["chi2_reduced"]),
        "dof_mask_removed": opt_int("dof_mask_removed"),
        "chi2_reduced_order": float(row["chi2_reduced_order"]),
        "dof_mask_removed_order": opt_int("dof_mask_removed_order"),
        "chi2_reduced_joint": float(row["chi2_reduced_joint"]),
        "dof_mask_removed_joint": opt_int("dof_mask_removed_joint"),
        "masked_frac": float(row["masked_frac"]),
        **_shift_fields(row),
        **_convergence_fields(row),
    }


def _shift_fields(row: Any) -> dict[str, Any]:
    """Wavelength-offset fields (``None`` / ``not_fitted`` in older result files)."""
    if "wave_shift" not in (row.array.names or []):
        return {"wave_shift": None, "wave_shift_err": None, "wave_shift_value_source": "not_fitted"}
    shift = _json_cell(row["wave_shift"], [])
    err = _json_cell(row["wave_shift_err"], [])
    return {
        "wave_shift": tuple(shift) if shift else None,
        "wave_shift_err": tuple(err) if err else None,
        "wave_shift_value_source": str(row["wave_shift_value_source"]).strip(),
    }


def _convergence_fields(row: Any) -> dict[str, Any]:
    """Optimiser outcome (``None`` when not fitted and in older result files)."""
    if "converged" not in (row.array.names or []):
        return {"converged": None, "n_eval": None}
    flag = str(row["converged"]).strip()
    n_eval = int(row["n_eval"])
    return {
        "converged": {"true": True, "false": False}.get(flag),
        "n_eval": n_eval if n_eval >= 0 else None,
    }


def load_exposure_order(path: str | os.PathLike[str], order: int | None = None) -> dict[str, Any]:
    """One order of a result FITS as a dict: arrays, ``ORDER_SUMMARY`` fields and metadata.

    ``order=None`` is allowed for a file holding a single order.
    """
    with fits.open(path) as hdul:
        if not _is_s2d(hdul):
            raise DataError(f"{path}: not a gtellcorr result file")
        if order is None:
            data = hdul["ORDER_SUMMARY"].data
            if data is None or len(data) != 1:
                raise ValueError(f"{path}: order must be specified")
            order = int(data[0]["order"])
        row = _summary_row(hdul, int(order))
        if row is None:
            available = [int(r["order"]) for r in hdul["ORDER_SUMMARY"].data]
            raise KeyError(f"{path}: missing order {int(order)}; available orders={available}")
        index = _row_index(row)
        n = _row_pixels(row, hdul["WAVE_FIT"].data[index])
        hdr = hdul[0].header
        # HDUList has no .get()
        fit_summary = _table_to_dict(hdul["FIT_SUMMARY"] if "FIT_SUMMARY" in hdul else None)  # noqa: SIM401
        run_metadata = _table_to_dict(hdul["RUN_METADATA"] if "RUN_METADATA" in hdul else None)  # noqa: SIM401

        def values(name: str) -> NDArray[Any]:
            got = _row_values(hdul, name, index, n)
            assert got is not None
            return got

        def optional(name: str) -> NDArray[Any] | None:
            got = _row_values(hdul, name, index, n, optional=True)
            return None if got is None else _nan_to_none(got)

        out: dict[str, Any] = {
            "wave_fit": values("WAVE_FIT"),
            "wave_input": values("WAVE_INPUT"),
            "flux_obs": values("FLUX_OBS"),
            "err": values("ERR"),
            "tell_model": values("TELL_MODEL"),
            "flux_corrected": values("FLUX_CORRECTED"),
            "err_corrected": optional("ERR_CORRECTED"),
            "continuum": optional("CONTINUUM"),
            "stellar_model": optional("STELLAR_MODEL"),
            "exclusion": np.asarray(values("EXCLUSION"), bool),
            "order": int(order),
            "row_index": index,
            "instrument": str(hdr.get("INSTRUME", "")),
            "exposure": str(run_metadata.get("exposure") or hdr.get("EXPOSURE", "")),
            "correction_mode": str(hdr.get("CORRMODE", "")),
            "fit_summary": fit_summary,
            "run_metadata": run_metadata,
        }
        n_pca = int(hdr.get(FITS_PCA_N_KEY, 0) or 0)
        if n_pca > 0 and all(_pca_name(i) in hdul for i in range(1, n_pca + 1)):
            out["pca_models"] = np.vstack([values(_pca_name(i)) for i in range(1, n_pca + 1)])
            out["n_pca"] = n_pca
            out["pca_mode"] = str(hdr.get(FITS_PCA_MODE_KEY, "") or "")
        out.update(_summary_fields(row))
        # present only when a wavelength offset was fitted
        out["wave_fit_corr"] = _row_values(hdul, "WAVE_FIT_CORR", index, n, optional=True)
        out["wave_input_corr"] = _row_values(hdul, "WAVE_INPUT_CORR", index, n, optional=True)
    return out


def _result_files(data_dir: Path) -> list[Path]:
    """Exposure result FITS files of a data directory."""
    paths: set[Path] = set()
    for pattern in EXPOSURE_FITS_PATTERNS:
        paths.update(p for p in data_dir.glob(pattern) if p.is_file())
    return sorted(paths)


def discover_result_orders(data_dir: str | os.PathLike[str]) -> list[int]:
    """All order numbers stored in the result FITS files of a data directory."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"missing result data dir: {root}")
    orders: set[int] = set()
    for path in _result_files(root):
        with fits.open(path) as hdul:
            if not _is_s2d(hdul):
                raise DataError(f"{path}: not a gtellcorr result file")
            summary = hdul["ORDER_SUMMARY"].data
            if summary is None or len(summary) == 0:
                raise ValueError(f"{path}: ORDER_SUMMARY is empty")
            orders.update(int(row["order"]) for row in summary)
    return sorted(orders)


def read_order_timeseries(
    data_dir: str | os.PathLike[str],
    order: int,
    *,
    only_used_in_fit: bool = True,
    require_pca: bool = False,
) -> list[dict[str, Any]]:
    """One order across the exposures of a run, sorted by time.

    Only successfully fitted orders are returned; ``require_pca`` keeps only those
    that already carry PCA models.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"missing result data dir: {root}")
    rows = []
    for path in _result_files(root):
        try:
            data = load_exposure_order(path, order=order)
        except KeyError:
            continue
        if only_used_in_fit and data.get("used_in_fit") is False:
            continue
        if data.get("status") not in ("ok", "joint_ok"):
            continue
        has_pca = "pca_models" in data
        if require_pca and not has_pca:
            continue
        time_value = None
        raw_time = data.get("run_metadata", {}).get("time_utc")
        if raw_time:
            try:
                time_value = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            except ValueError:
                time_value = None
        row = {
            "path": path,
            "time": time_value,
            "exposure": str(data.get("exposure", "")),
            "order": int(order),
            "wave": np.asarray(data["wave_fit"], float),
            "tell_model": np.asarray(data["tell_model"], float),
            "correction_mode": str(data.get("correction_mode", "")),
            "instrument": str(data.get("instrument", "")),
            "used_in_fit": data.get("used_in_fit"),
            "treatment": str(data.get("treatment", "")),
        }
        if has_pca:
            row["pca_models"] = np.asarray(data["pca_models"], float)
            row["n_pca"] = int(data["n_pca"])
            row["pca_mode"] = str(data.get("pca_mode", ""))
        rows.append(row)
    rows.sort(key=lambda r: (r["time"] is None, r["time"] or datetime.min, r["exposure"]))
    return rows


# ---------------------------------------------------------------------------
# post-processing updates
# ---------------------------------------------------------------------------
def update_order_columns(
    path: str | os.PathLike[str],
    order: int,
    columns: Mapping[str, Any],
    *,
    header_updates: Mapping[str, Any] | None = None,
    overwrite_columns: bool = True,
    remove_columns: Iterable[str] | None = None,
) -> Path:
    """Add or replace per-order image products in a result FITS (one order's row)."""
    target = Path(os.path.abspath(str(path)))
    remove = list(remove_columns or [])
    if not columns and not header_updates and not remove:
        return target
    with fits.open(target, mode="update") as hdul:
        if not _is_s2d(hdul):
            raise DataError(f"{target}: not a gtellcorr result file")
        row = _summary_row(hdul, int(order))
        if row is None:
            available = [int(r["order"]) for r in hdul["ORDER_SUMMARY"].data]
            raise KeyError(f"{target}: missing order {int(order)}; available orders={available}")
        index = _row_index(row)
        n = _row_pixels(row, hdul["WAVE_FIT"].data[index])
        shape = np.asarray(hdul["WAVE_FIT"].data).shape
        for name in remove:
            if str(name) in hdul:
                del hdul[hdul.index_of(str(name))]
        for name, values in columns.items():
            arr = np.asarray(values, float)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(
                    f"{name}: product length {arr.shape} does not match order {order} length {n}"
                )
            if name in hdul:
                if not overwrite_columns:
                    raise ValueError(f"{target}: FITS product {name!r} already exists")
                image = np.asarray(hdul[name].data, dtype=np.float64).copy()
                header = hdul[name].header.copy()
            else:
                image = np.full(shape, np.nan, dtype=np.float64)
                header = fits.Header()
            image[index, :] = np.nan
            image[index, :n] = arr
            hdu = fits.ImageHDU(data=image, name=name, header=header)
            hdu.header["ROWMAP"] = _ROWMAP
            for key, value in (header_updates or {}).items():
                hdu.header[f"HIERARCH {key}" if len(key) > 8 else key] = value
            if hdu.name in hdul:
                hdul[hdul.index_of(hdu.name)] = hdu
            else:
                hdul.append(hdu)
        for key, value in (header_updates or {}).items():
            hdul[0].header[f"HIERARCH {key}" if len(key) > 8 else key] = value
        hdul.flush()
    return target
