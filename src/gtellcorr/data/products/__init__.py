"""Products of a run: output layout, result files, summaries and post-processing."""

from __future__ import annotations

from gtellcorr.data.products.config import Postprocessing
from gtellcorr.data.products.fits_schema import (
    load_exposure_order,
    read_order_timeseries,
    write_exposure_fits,
)
from gtellcorr.data.products.layout import RunLayout
from gtellcorr.data.products.pca import compute_telluric_pca_for_run
from gtellcorr.data.products.result import ExposureResult, Result, load_result

__all__ = [
    "ExposureResult",
    "Postprocessing",
    "Result",
    "RunLayout",
    "compute_telluric_pca_for_run",
    "load_exposure_order",
    "load_result",
    "read_order_timeseries",
    "write_exposure_fits",
]
