"""Full-band templates: VALD slicing per chunk and stitching of the chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gtellcorr.constants import C_KMS
from gtellcorr.preprocessing import synthesis


def _vald(path: Path, lines: list[tuple[str, float]]) -> Path:
    """A minimal long-format VALD file: 2 header lines, 4 lines per transition."""
    body = []
    for species, wave_nm in lines:
        body += [
            f"'{species}',       {wave_nm * 10:.5f},  -1.000,  2.000, 1.0, 2.0, 0.0\n",
            "  LS  x\n",
            "  LS  y\n",
            "'ref'\n",
        ]
    path.write_text("header 1\nheader 2\n" + "".join(body) + "* oscillator strengths\n")
    return path


def _waves(path: str) -> list[float]:
    text = Path(path).read_text().splitlines()
    return [float(t.split(",")[1]) / 10 for t in text if t.startswith("'") and "," in t]


def test_broad_lines_reach_every_chunk_within_50_nm(tmp_path: Path) -> None:
    full = _vald(
        tmp_path / "full.vald",
        [("Na 1", 589.0), ("Fe 1", 630.0), ("Fe 1", 652.0), ("H 1", 656.28), ("Fe 1", 700.0)],
    )
    chunks = [(625, 650), (650, 675)]
    paths = synthesis._slice_vald(str(full), chunks, tmp_path / "slices")
    first, second = _waves(paths[(625, 650)]), _waves(paths[(650, 675)])
    assert first == [589.0, 630.0, 652.0, 656.28]  # Na D and H-alpha within 50 nm; 652 within 3
    assert second == [652.0, 656.28]  # Na D is 61 nm away; the Fe lines 20-25 nm away


def test_stitch_keeps_each_chunk_range(tmp_path: Path) -> None:
    chunks = [(600, 605), (605, 610)]
    step = 0.0013
    parts = []
    for k, (lo, hi) in enumerate(chunks):  # the two chunks disagree by a constant
        w = np.arange(lo - 2.0 + 0.0004 * k, hi + 2.0, step)
        parts.append(np.column_stack([w, np.full_like(w, 1.0 + 0.001 * k)]))
    reach = 5.0  # km/s
    wave, flux = synthesis._stitch(parts, chunks, reach_kms=reach)
    assert np.all(np.diff(wave) > 0)  # sorted, no interleaving
    assert np.all(flux[wave < 605] == 1.0) and np.all(flux[wave >= 605] == 1.001)
    # at the join each side lies within one step of 605 nm: no gap beyond two steps
    assert np.max(np.diff(wave)) <= 2 * step + 1e-12
    # outer margins kept, less the kernel's reach at the two array ends
    assert wave[0] >= parts[0][0, 0] * (1 + reach / C_KMS) and wave[0] < 598.02
    assert wave[-1] <= parts[1][-1, 0] * (1 - reach / C_KMS) and wave[-1] > 611.97
