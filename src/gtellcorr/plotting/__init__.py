"""Figures of a run, drawn from its products only (result FITS, ``summary.csv``, RV
table and CCF files, strong-mask transmission). Nothing is refitted.
"""

from __future__ import annotations

from gtellcorr.plotting.order_fit import plot_order_fit
from gtellcorr.plotting.param_timeseries import plot_param_heatmap, plot_param_timeseries
from gtellcorr.plotting.rv import (
    load_ccf_data,
    plot_ccf_diagnostics,
    plot_rv_master_vs_perorder,
    plot_rv_timeseries,
)
from gtellcorr.plotting.strong_mask import plot_strong_telluric_transmission
from gtellcorr.plotting.telluric_pca import plot_telluric_pca_map

__all__ = [
    "load_ccf_data",
    "plot_ccf_diagnostics",
    "plot_order_fit",
    "plot_param_heatmap",
    "plot_param_timeseries",
    "plot_rv_master_vs_perorder",
    "plot_rv_timeseries",
    "plot_strong_telluric_transmission",
    "plot_telluric_pca_map",
]
