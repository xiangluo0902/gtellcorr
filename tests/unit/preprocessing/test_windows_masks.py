from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gtellcorr.data.products.layout import RunLayout, date_token, safe_component
from gtellcorr.preprocessing.config import StellarLineMask
from gtellcorr.preprocessing.stellar_mask import (
    build_stellar_linemask,
    fit_exclusion,
    missing_exposure_linemasks,
    write_exposure_linemasks,
)
from gtellcorr.preprocessing.windows import (
    C_KMS,
    boolean_mask_windows,
    exclusion_pixels,
    good_pixels,
    merge_windows,
    normalise_windows,
    pad_window,
    read_windows,
    shift_windows,
)


def test_window_helpers(tmp_path: Path) -> None:
    lo, hi = pad_window(600.0, 601.0, 30.0)
    d = 30.0 / C_KMS * 600.5
    assert (lo, hi) == (600.0 - d, 601.0 + d)
    assert pad_window(1.0, 2.0, 0.0) == (1.0, 2.0)
    assert shift_windows([(600.0, 601.0)], 10.0) == [
        (600.0 * (1 + 10 / C_KMS), 601.0 * (1 + 10 / C_KMS))
    ]
    wave = np.array([599.0, 600.0, 600.5, 601.0, 602.0])
    assert exclusion_pixels(wave, [(600.0, 601.0)]).tolist() == [False, True, True, True, False]
    assert good_pixels(wave, [(600.0, 601.0)]).tolist() == [True, False, False, False, True]
    assert merge_windows([(3, 4), (1, 2), (2.05, 2.5), (np.nan, 1)], merge_gap_nm=0.1) == [
        (1.0, 2.5),
        (3.0, 4.0),
    ]
    assert merge_windows([(1, 1.001), (2, 3)], min_width_nm=0.5) == [(2.0, 3.0)]
    assert merge_windows([]) == []
    mask = np.array([False, True, True, False, True])
    assert boolean_mask_windows(wave, mask, pad_nm=0.1) == [(599.9, 600.6), (601.9, 602.1)]
    assert boolean_mask_windows(wave, np.zeros(5, bool)) == []
    assert normalise_windows(None) == [] and normalise_windows(np.empty((0, 2))) == []
    assert normalise_windows([[1, 2], [3, 4]]) == [(1.0, 2.0), (3.0, 4.0)]
    f = tmp_path / "w.txt"
    f.write_text("# comment\n2.0 1.0  extra\n\n3 4 # tail\n")
    assert read_windows(f) == [(2.0, 1.0), (3.0, 4.0)]
    assert read_windows(f, ordered=True) == [(1.0, 2.0), (3.0, 4.0)]


def test_layout_names(tmp_path: Path) -> None:
    layout = RunLayout.for_run(
        tmp_path, target="TOI 2076", site_token="KPNO", instrument="WIYN/NEID", date="20220213"
    )
    assert layout.root == tmp_path / "TOI_2076" / "KPNO_NEID" / "20220213"
    assert layout.stem == "TOI2076_NEID_20220213"
    assert layout.rv_table.name == "TOI2076_NEID_20220213_rv.csv"
    assert layout.rv_table.parent == layout.root / "Radial_Velocities" / "data"
    assert (
        layout.exposure_linemask("/x/neidL2_1.fits").name
        == "neidL2_1_stellarlinemask_topocentric.txt"
    )
    assert (
        layout.ccf_path("20220213T073505", "a/neidL2_x.fits").name
        == "CCF_TOI_2076_NEID_20220213T073505_neidL2_x.npz"
    )
    assert layout.master_npz.name.endswith("_master.npz") and layout.lines_catalog.suffix == ".json"
    with pytest.raises(ValueError, match="unknown product"):
        layout.product_dir("x")
    with pytest.raises(ValueError, match="kind"):
        layout.product_dir("telluric_corr", "other")  # type: ignore[arg-type]
    assert safe_component(" a/b ") == "a_b" and safe_component("//") == "unknown"
    t = datetime(2022, 2, 13, tzinfo=UTC)
    assert date_token([t, None]) == "20220213"
    assert date_token([t, datetime(2022, 2, 15, tzinfo=UTC)]) == "20220213-20220215"
    assert date_token([]) == "unknown-date"


def test_stellar_linemask_and_exclusion(tmp_path: Path) -> None:
    w = np.linspace(600.0, 601.0, 1001)
    f = np.ones_like(w)
    f[100:110] = 0.5
    f[112:115] = 0.5
    f[800:805] = 0.5
    cfg = StellarLineMask(depth_below=0.97, pad_kms=0.0, merge_gap_nm=0.01)
    windows = build_stellar_linemask((w, f), cfg)
    assert windows == [(w[100], w[114]), (w[800], w[804])]
    assert build_stellar_linemask((w, f), cfg, wave_range_nm=(600.5, 601.0)) == [(w[800], w[804])]

    layout = RunLayout.for_run(tmp_path, target="T", site_token="S", instrument="I", date="D")
    source = "template:t.dat"
    rows = [{"file": "a.fits", "rv_kms": "10.0"}, {"file": "b.fits", "rv_kms": "-5.0"}]
    rv_csv = tmp_path / "rv.csv"
    rv_csv.write_text("file,rv_kms\na.fits,10.0\nb.fits,-5.0\n")
    assert missing_exposure_linemasks(rv_csv, layout) == ["a.fits", "b.fits"]
    assert write_exposure_linemasks(rows, windows, source, layout) == 2
    assert missing_exposure_linemasks(rv_csv, layout) == []
    shifted = read_windows(layout.exposure_linemask("a.fits"))
    np.testing.assert_allclose(shifted, shift_windows(windows, 10.0), atol=5e-6)
    # windows of a topocentric template are written unchanged, without an RV
    write_exposure_linemasks(rows, windows, source, layout, shift=False)
    np.testing.assert_allclose(read_windows(layout.exposure_linemask("a.fits")), windows, atol=5e-6)
    assert "apparent stellar RV" not in layout.exposure_linemask("a.fits").read_text()

    grid = np.array([600.0, 600.2, 600.4, 600.6])
    excl = fit_exclusion(
        grid,
        stellar_mask_file=layout.exposure_linemask("a.fits"),
        activity_windows=[(600.55, 600.65)],
        bad_regions=[(599.0, 600.01)],
        strong_windows=np.array([[600.35, 600.45]]),
    )
    assert excl.tolist() == [True, False, True, True]
    assert not fit_exclusion(grid).any()
    with pytest.raises(FileNotFoundError, match="stellar-line mask missing"):
        fit_exclusion(grid, stellar_mask_file=tmp_path / "none.txt")
