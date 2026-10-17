"""Atmospheric profiles: temperature, pressure and water vapour against altitude."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

__all__ = ["plot_profiles"]

#: Profile fields drawn, with their axis labels and scales.
_PANELS = (
    ("T_K", "Temperature [K]", "linear"),
    ("P_hPa", "Pressure [hPa]", "log"),
    ("H2O_ppmv", r"H$_2$O [ppmv]", "log"),
)


def plot_profiles(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    site_alt_km: float | None = None,
    onsite: Any = None,
    z_max_km: float = 120.0,
    inset_z_km: tuple[float, float] | None = (0.0, 20.0),
    out: str | Path | Literal[False] | None = None,
) -> tuple[Any, Any]:
    """Temperature, pressure and H2O of one or more profiles against altitude.

    *profiles* maps a label to a profile with ``z_km``, ``T_K``, ``P_hPa`` and
    ``H2O_ppmv`` (as :func:`~gtellcorr.data.atmosphere.full_profile_for_exposure`
    returns it). *site_alt_km* draws the observatory altitude; *onsite* (an
    exposure's ``weather``, with ``T_K``, ``P_hpa`` and ``RH_pct``) marks the on-site
    measurements there, the water vapour derived from the relative humidity.
    *inset_z_km* adds a linear-altitude zoom of the water vapour (``None``: none).
    *out* saves the figure there (``None`` or ``False``: no file). Returns ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.0), sharey=True)
    for label, profile in profiles.items():
        z = np.asarray(profile["z_km"], float)
        for ax, (field, _, _) in zip(axes, _PANELS, strict=True):
            values = np.asarray(profile[field], float)
            if field == "H2O_ppmv":
                values = np.clip(values, 1e-4, None)
            ax.plot(values, z, lw=1.2, label=label)
    for ax, (_, xlabel, scale) in zip(axes, _PANELS, strict=True):
        ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, z_max_km)
        ax.yaxis.set_major_locator(MultipleLocator(20.0))
        ax.tick_params(direction="in", top=True, right=True)
        if site_alt_km is not None:
            ax.axhline(site_alt_km, color="0.55", ls=":", lw=0.8)
    axes[0].set_ylabel("Altitude [km]")
    axes[2].set_xlim(1e-3, 1e5)
    if onsite is not None and site_alt_km is not None:
        _mark_onsite(axes, onsite, site_alt_km)
    if inset_z_km is not None:
        inset = axes[2].inset_axes([0.5, 0.5, 0.46, 0.46])
        lo, hi = inset_z_km
        shown = []
        for line in axes[2].get_lines():
            if line.get_linestyle() == "-":
                x, z = (np.asarray(v, float) for v in line.get_data())
                inset.plot(x, z, color=line.get_color(), lw=0.9)
                shown.append(x[(z >= lo) & (z <= hi)])
        values = np.concatenate(shown) if shown else np.array([])
        if values.size:
            inset.set_xlim(values.min() / 1.5, values.max() * 1.5)
        inset.set_ylim(lo, hi)
        inset.set_xscale("log")
        inset.tick_params(labelsize=7, direction="in")
        if site_alt_km is not None:
            inset.axhline(site_alt_km, color="0.55", ls=":", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 1:
        axes[1].legend(handles, labels, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    if out is not None and out is not False:
        fig.savefig(out, dpi=200)
    return fig, axes


def _mark_onsite(axes: Any, onsite: Any, site_alt_km: float) -> None:
    """Stars at the site altitude for the on-site temperature, pressure and humidity."""
    from gtellcorr.data.atmosphere.profile import rh_to_vmr_ppmv

    style = {"marker": "*", "ms": 8, "color": "0.15", "ls": "none", "zorder": 6}
    T, P, rh = (getattr(onsite, name, None) for name in ("T_K", "P_hpa", "RH_pct"))
    if T is not None:
        axes[0].plot(T, site_alt_km, label="on site", **style)
    if P is not None:
        axes[1].plot(P, site_alt_km, **style)
    if T is not None and P is not None and rh is not None:
        h2o = rh_to_vmr_ppmv(np.asarray([rh]), np.asarray([T]), np.asarray([P]))[0]
        axes[2].plot(h2o, site_alt_km, markerfacecolor="white", **style)
