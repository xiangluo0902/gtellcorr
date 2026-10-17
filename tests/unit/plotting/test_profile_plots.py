"""Atmospheric-profile figure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gtellcorr.plotting import plot_profiles


def _profile(water: float = 1.0) -> dict[str, np.ndarray]:
    z = np.arange(0.0, 121.0)
    return {
        "z_km": z,
        "T_K": 288.0 - 6.5 * np.minimum(z, 11.0),
        "P_hPa": 1013.0 * np.exp(-z / 8.0),
        "H2O_ppmv": water * (5000.0 * np.exp(-z / 2.0) + 5.0),
    }


def test_profiles_with_site_onsite_and_inset(tmp_path: Path) -> None:
    out = tmp_path / "profiles.png"
    onsite = SimpleNamespace(T_K=285.0, P_hpa=700.0, RH_pct=20.0)
    fig, axes = plot_profiles(
        {"model": _profile(), "drier": _profile(0.5)}, site_alt_km=3.2, onsite=onsite, out=out
    )
    assert out.is_file()
    assert [ax.get_xscale() for ax in axes] == ["linear", "log", "log"]
    assert len(axes[2].child_axes) == 1  # the water-vapour inset
    legend = axes[1].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["model", "drier", "on site"]
    stars = [ln for ax in axes for ln in ax.get_lines() if ln.get_marker() == "*"]
    assert len(stars) == 3
    plt.close(fig)


def test_profiles_minimal() -> None:
    fig, axes = plot_profiles({"model": _profile()}, inset_z_km=None)
    assert axes[2].child_axes == [] and axes[1].get_legend() is None
    plt.close(fig)


def test_profiles_out_false_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fig, _ = plot_profiles({"model": _profile()}, out=False)
    plt.close(fig)
    assert list(tmp_path.iterdir()) == []
