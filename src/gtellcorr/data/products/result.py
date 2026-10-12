"""Results of a run: :class:`Result`, :class:`ExposureResult`, :func:`load_result`."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from gtellcorr.data.products.fits_schema import (
    CORRECTED_WAVE_HDU_NAMES,
    EXPOSURE_FITS_PATTERNS,
    S2D_IMAGE_HDU_NAMES,
    _is_s2d,
)
from gtellcorr.data.products.layout import PRODUCT_TELLURIC
from gtellcorr.data.products.summary import shift_cell
from gtellcorr.errors import DataError

__all__ = ["ExposureResult", "Result", "load_result", "summary_rows_from_fits"]


@dataclass(frozen=True)
class ExposureResult:
    """One exposure's result FITS, as arrays of shape ``(n_order, n_pixel)`` (NaN-padded)."""

    path: Path
    """The result FITS file."""

    @cached_property
    def _data(self) -> dict[str, Any]:
        """Images, order numbers, ``ORDER_SUMMARY`` and primary header, read once."""
        with fits.open(self.path) as hdul:
            if not _is_s2d(hdul):
                raise DataError(f"{self.path}: not a gtellcorr result file")
            images = {
                name: np.asarray(hdul[name].data)
                for name in S2D_IMAGE_HDU_NAMES + CORRECTED_WAVE_HDU_NAMES
                if name in hdul
            }
            table = hdul["ORDER_SUMMARY"].data
            orders = np.asarray(table["order"], int)
            header = dict(hdul[0].header)
        return {"images": images, "orders": orders, "table": table, "header": header}

    @property
    def name(self) -> str:
        """Name of the input exposure."""
        return str(self._data["header"].get("EXPOSURE", self.path.name))

    @property
    def orders(self) -> NDArray[Any]:
        """Order numbers, one per image row."""
        return np.asarray(self._data["orders"])

    def image(self, name: str) -> NDArray[Any]:
        """Any image product (``WAVE_FIT``, ``TELL_MODEL``, ...; ``WAVE_FIT_CORR`` /
        ``WAVE_INPUT_CORR`` only when a wavelength offset was fitted)."""
        return np.asarray(self._data["images"][name])

    @property
    def wave(self) -> NDArray[Any]:
        """``WAVE_FIT``: air wavelengths [nm] in the topocentric frame."""
        return self.image("WAVE_FIT")

    @property
    def flux(self) -> NDArray[Any]:
        """``FLUX_OBS``: the observed, normalised flux."""
        return self.image("FLUX_OBS")

    @property
    def flux_corrected(self) -> NDArray[Any]:
        """``FLUX_CORRECTED``: the flux divided by the telluric model."""
        return self.image("FLUX_CORRECTED")

    @property
    def telluric(self) -> NDArray[Any]:
        """``TELL_MODEL``: the best-fit telluric transmission."""
        return self.image("TELL_MODEL")

    @property
    def exclusion(self) -> NDArray[Any]:
        """``EXCLUSION`` as booleans: ``True`` = pixel left out of the fit."""
        return self.image("EXCLUSION").astype(bool)

    @property
    def order_table(self) -> Any:
        """``ORDER_SUMMARY`` as a pandas DataFrame."""
        import pandas as pd

        table = self._data["table"]
        return pd.DataFrame({name: np.asarray(table[name]) for name in table.names})


@dataclass(frozen=True)
class Result:
    """A run directory (``<output>/<target>/<site>_<instrument>/<date>``)."""

    root: Path
    """The run directory."""

    @property
    def data_dir(self) -> Path:
        """``telluric_corr/data``: result FITS files and ``summary.csv``."""
        return self.root / PRODUCT_TELLURIC / "data"

    @property
    def figure_dir(self) -> Path:
        """``telluric_corr/figure``."""
        return self.root / PRODUCT_TELLURIC / "figure"

    @cached_property
    def summary(self) -> Any:
        """``summary.csv`` as a pandas DataFrame."""
        import pandas as pd

        return pd.read_csv(self.data_dir / "summary.csv")

    @cached_property
    def exposures(self) -> list[ExposureResult]:
        """One :class:`ExposureResult` per result FITS, sorted by file name."""
        paths: set[Path] = set()
        for pattern in EXPOSURE_FITS_PATTERNS:
            paths.update(self.data_dir.glob(pattern))
        return [ExposureResult(p) for p in sorted(paths)]

    def exposure(self, name: str) -> ExposureResult:
        """The result of the exposure whose file name starts with ``name``."""
        stem = Path(name).stem
        for ex in self.exposures:
            if ex.path.name.startswith(stem):
                return ex
        raise KeyError(f"no result for exposure {name!r} in {self.data_dir}")

    @property
    def manifest(self) -> dict[str, Any]:
        """``run_manifest.json`` (empty if absent)."""
        path = self.root / "run_manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}


def load_result(path: str | os.PathLike[str]) -> Result:
    """Open a run directory (the one holding ``telluric_corr/``)."""
    root = Path(path).expanduser().resolve()
    if not (root / PRODUCT_TELLURIC / "data").is_dir():
        raise FileNotFoundError(f"{root} is not a run directory (no {PRODUCT_TELLURIC}/data)")
    return Result(root)


def _shift_cell(row: Any) -> str:
    """``wave_shift`` summary cell of an ``ORDER_SUMMARY`` row (empty if absent)."""
    if "wave_shift" not in (row.array.names or []):
        return ""
    values = json.loads(str(row["wave_shift"]).strip() or "[]")
    return shift_cell(values or None)


def _outcome_cells(row: Any) -> list[str]:
    """``converged`` and ``n_eval`` summary cells of an ``ORDER_SUMMARY`` row (empty in
    older result files)."""
    if "converged" not in (row.array.names or []):
        return ["", ""]
    n_eval = int(row["n_eval"])
    return [str(row["converged"]).strip(), str(n_eval) if n_eval >= 0 else ""]


def summary_rows_from_fits(path: str | os.PathLike[str], time_iso: str) -> list[list[Any]]:
    """Summary rows of the fitted orders of an existing result (``elapsed_s`` empty)."""
    rows = []
    with fits.open(path) as hdul:
        exposure = str(hdul[0].header.get("EXPOSURE", ""))
        for r in hdul["ORDER_SUMMARY"].data:
            if not bool(r["used_in_fit"]):
                continue
            relcol = json.loads(str(r["relcol"]))
            shape = float(r["lsf_shape"])
            rows.append(
                [
                    exposure, int(r["order"]),
                    f"{float(r['wave_lo']):.1f}", f"{float(r['wave_hi']):.1f}",
                    time_iso, json.dumps(list(relcol)), json.dumps(relcol),
                    f"{float(r['resolution']):.0f}",
                    float(r["chi2_reduced_order"]), int(r["dof_mask_removed_order"]),
                    float(r["chi2_reduced_order"]), int(r["dof_mask_removed_order"]),
                    float(r["chi2_reduced_joint"]), int(r["dof_mask_removed_joint"]),
                    f"{float(r['masked_frac']):.3f}",
                    "" if not np.isfinite(shape) else f"{shape:.4f}",
                    str(r["status"]).strip(), "", _shift_cell(r), *_outcome_cells(r),
                ]
            )  # fmt: skip
    return rows
