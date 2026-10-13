"""Plot model-derived strong telluric transmission masks.

The input is the ``*_strong_telluric_transmission.npz`` product written by
:mod:`gtellcorr.preprocessing.telluric_mask`.  The output defaults to the sibling
``figure`` directory:

    <run_root>/mask_spectral_line/data/<stem>.npz
    <run_root>/mask_spectral_line/figure/<stem>.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

TELLURIC_MOLECULE_COLORS = {
    "h2o": "#0ead69",
    "ch4": "#EA2F14",
    "co": "#FB9E3A",
    "co2": "#F0E76F",
    "o2": "#008BFF",
    "o3": "#56B4E9",
    "nh3": "#C2D9FF",
    "n2o": "#8E8FFA",
    "no": "#9D4EDD",
    "no2": "#b5179e",
    "hno3": "#7A08FA",
    "so2": "#8C510A",
}


def _default_out(spectrum_path):
    """``<product>/figure/<stem>.png`` beside the product's ``data`` directory."""
    data_dir, base = os.path.split(os.path.abspath(str(spectrum_path)))
    stem = os.path.splitext(base)[0]
    product_dir = os.path.dirname(data_dir) if os.path.basename(data_dir) == "data" else data_dir
    return os.path.join(product_dir, "figure", stem + ".png")


def _metadata_threshold(metadata):
    """The mask threshold from the product's ``metadata`` string (the repr of a
    dict), or ``None``."""
    if "'threshold':" not in metadata:
        return None
    try:
        return float(metadata.split("'threshold':", 1)[1].split(",", 1)[0].strip())
    except Exception:
        return None


def plot_strong_telluric_transmission(spectrum_path, *, out=None, threshold=None):
    """Plot one strong-telluric-mask transmission product.

    ``out=None`` writes to the sibling ``figure`` directory; a string
    writes there; ``out=False`` draws and returns without saving.

    Returns ``(fig, ax, out_path)`` where ``out_path`` is None when ``out=False``.
    """
    with np.load(spectrum_path, allow_pickle=False) as data:
        wave = np.asarray(data["wave_nm"], float)
        total = np.asarray(data["transmission"], float)
        mask = (
            np.asarray(data["mask"], bool) if "mask" in data.files else np.zeros(wave.shape, bool)
        )
        molecule_names = (
            [str(v).lower() for v in data["molecule_names"]]
            if "molecule_names" in data.files
            else []
        )
        trans_by_molecule = (
            np.asarray(data["transmission_by_molecule"], float)
            if "transmission_by_molecule" in data.files
            else np.empty((0, wave.size), float)
        )
        metadata = str(data["metadata"]) if "metadata" in data.files else ""

    if threshold is None:
        threshold = _metadata_threshold(metadata)

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    if mask.any():
        ax.fill_between(
            wave,
            -0.05,
            1.05,
            where=mask,
            color="lightgrey",
            alpha=0.3,
            step="mid",
            zorder=0,
            label="strong telluric mask",
        )

    ax.plot(wave, total, color="#111111", lw=0.5, zorder=1, alpha=0.8, label="total transmission")
    for i, molecule in enumerate(molecule_names):
        if i >= trans_by_molecule.shape[0]:
            break
        color = TELLURIC_MOLECULE_COLORS.get(molecule, f"C{i + 1}")
        ax.plot(
            wave,
            trans_by_molecule[i],
            color=color,
            lw=1,
            alpha=0.8,
            zorder=i + 2,
            label=f"{molecule.upper()} transmission",
        )

    if threshold is not None:
        ax.axhline(
            float(threshold),
            color="0.25",
            lw=0.5,
            ls="--",
            zorder=4,
            label=f"threshold={float(threshold):.3g}",
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("wavelength (air, topocentric) [nm]", fontsize=9)
    ax.set_ylabel("transmission", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=True, borderpad=0.3, handlelength=1.6)
    fig.tight_layout(pad=0.4)

    out_path = None
    if out is not False:
        out_path = _default_out(spectrum_path) if out is None else os.path.abspath(str(out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
    return fig, ax, out_path
