"""User-supplied stellar inputs: RV file, template (medium, frame, resolution), line windows."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from astropy.io import fits

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.frames import vac_to_air
from gtellcorr.data.products.layout import RunLayout
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.preprocessing import master_spectrum, prepare, stellar_lines
from gtellcorr.preprocessing import rv as rvmod
from gtellcorr.preprocessing.config import (
    FitMasks,
    RVConfig,
    StellarLineMask,
    StellarModelConfig,
)
from gtellcorr.preprocessing.prepare import StellarProducts
from gtellcorr.preprocessing.stellar_mask import write_exposure_linemasks
from gtellcorr.preprocessing.synthesis import broaden_template, stellar_model_for_order
from gtellcorr.preprocessing.treatments import TemplateTreatment
from gtellcorr.preprocessing.windows import C_KMS, read_windows, shift_windows
from gtellcorr.settings import Settings

NEID = instruments.get("NEID")


def _layout(tmp_path: Path) -> RunLayout:
    return RunLayout.for_run(tmp_path, target="T", site_token="S", instrument="I", date="D")


def _exposures(tmp_path: Path, names: list[str]) -> list[Any]:
    out = []
    for i, name in enumerate(names):
        path = tmp_path / "raw" / name
        path.parent.mkdir(exist_ok=True)
        fits.PrimaryHDU(header=fits.Header({"JD-OBS": 2459000.5 + i})).writeto(path)
        time = datetime(2020, 5, 31, 0, 0, tzinfo=UTC) + timedelta(days=i)
        out.append(SimpleNamespace(path=path, airmass=1.1 + i, time_utc=time))
    return out


def _obs(tmp_path: Path, exposures: list[Any]) -> Any:
    return SimpleNamespace(
        instrument=NEID, target=None, transit=None, directory=tmp_path, exposures=exposures
    )


def _rv_file(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _measure_must_not_run(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the CCF RV measurement must not run")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_validation() -> None:
    with pytest.raises(ConfigError, match="'barycentric' is not supported"):
        StellarModelConfig(template_file="t.dat", template_frame="barycentric")
    with pytest.raises(ConfigError, match="template_medium"):
        StellarModelConfig(template_medium="vacuum")  # no template_file
    with pytest.raises(ConfigError, match="template_frame"):
        StellarModelConfig(template_frame="topocentric")
    with pytest.raises(ConfigError, match="must be > 0"):
        StellarModelConfig(template_file="t.dat", template_resolution=-1.0)
    with pytest.raises(ConfigError, match="template_medium"):
        StellarModelConfig(template_file="t.dat", template_medium="vac")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="required when source = 'user_file'"):
        RVConfig(source="user_file")
    with pytest.raises(ConfigError, match="set source = 'user_file'"):
        RVConfig(user_rv_file="rv.csv")
    defaults = StellarModelConfig()
    assert (defaults.template_medium, defaults.template_frame) == ("air", "rest")
    assert defaults.template_resolution is None and RVConfig().source == "measure"
    assert StellarLineMask().user_file is None


# ---------------------------------------------------------------------------
# RV
# ---------------------------------------------------------------------------
def test_user_rv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = _layout(tmp_path)
    exposures = _exposures(tmp_path, ["e1.fits", "e2.fits"])
    obs = _obs(tmp_path, exposures)
    user = _rv_file(
        tmp_path / "rv.csv",
        [["file", "rv_kms", "rv_err_kms"], ["e1", "-12.5", "0.01"], ["/x/e2.fits", "30000.0", ""]],
    )
    cfg = FitConfig(rv=RVConfig(source="user_file", user_rv_file=str(user)))
    monkeypatch.setattr(rvmod, "measure_rv_table", _measure_must_not_run)
    path, status = prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))
    assert (path, status) == (layout.rv_table, "user_file")
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == rvmod.RV_TABLE_COLUMNS
        rows = list(reader)
    assert [r["file"] for r in rows] == ["e1.fits", "e2.fits"]
    assert [float(r["rv_kms"]) for r in rows] == [-12.5, 30000.0]
    assert float(rows[0]["rv_err_kms"]) == 0.01 and np.isnan(float(rows[1]["rv_err_kms"]))
    assert [float(r["bjd"]) for r in rows] == [2459000.5, 2459001.5]
    assert [float(r["airmass"]) for r in rows] == [1.1, 2.1]
    assert np.isnan(float(rows[0]["acceptance"]))
    # an exposure without a row is an error naming it
    _rv_file(user, [["file", "rv_kms"], ["e1.fits", "1.0"]])
    with pytest.raises(DataError, match=r"no RV for 1 exposure\(s\): e2.fits"):
        prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))
    _rv_file(user, [["name", "rv"], ["e1.fits", "1.0"]])
    with pytest.raises(DataError, match="'file' and 'rv_kms'"):
        prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))


def test_measure_does_not_reuse_a_user_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    obs = _obs(tmp_path, _exposures(tmp_path, ["e1.fits"]))
    user = _rv_file(tmp_path / "rv.csv", [["file", "rv_kms"], ["e1.fits", "5.0"]])
    cfg = FitConfig(rv=RVConfig(source="user_file", user_rv_file=str(user)))
    prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))
    calls = []

    def measured(*args: Any, **kwargs: Any) -> Path:
        calls.append(1)
        return layout.rv_table

    monkeypatch.setattr(rvmod, "measure_rv_table", measured)
    # the table covers every exposure, but came from the user file: measure afresh
    assert prepare.ensure_rv_table(obs, FitConfig(), layout, template=(None, None))[1] == (
        "generated"
    )
    assert calls == [1]
    # a measured table is reused as before
    assert prepare.ensure_rv_table(obs, FitConfig(), layout, template=(None, None))[1] == ("reused")


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------
def test_template_medium(tmp_path: Path) -> None:
    obs = _obs(tmp_path, [])
    wave = np.linspace(600.0, 601.0, 11)
    path = tmp_path / "t.dat"
    np.savetxt(path, np.column_stack([wave, np.linspace(0.5, 1.0, 11)]))
    air = FitConfig(stellar_model=StellarModelConfig(template_file=str(path)))
    w_air, f_air = prepare.load_stellar_template(path, obs, air)
    assert np.array_equal(w_air, np.loadtxt(path)[:, 0])
    vac = air.replace(
        stellar_model=StellarModelConfig(template_file=str(path), template_medium="vacuum")
    )
    w_vac, f_vac = prepare.load_stellar_template(path, obs, vac)
    np.testing.assert_array_equal(w_vac, vac_to_air(w_air, vac.frames.vacuum_air_formula))
    assert np.all(w_vac < w_air) and np.array_equal(f_vac, f_air)
    # a synthesised template (no template_file) is used as written
    assert np.array_equal(prepare.load_stellar_template(path, obs, FitConfig())[0], w_air)


def _fwhm_kms(wave: Any, flux: Any) -> float:
    depth = 1.0 - np.asarray(flux)
    half = depth >= depth.max() / 2.0
    i0, i1 = np.flatnonzero(half)[[0, -1]]

    def edge(i: int, j: int) -> float:  # interpolate the half-depth crossing
        return float(np.interp(depth.max() / 2.0, [depth[j], depth[i]], [wave[j], wave[i]]))

    lo, hi = edge(i0, i0 - 1), edge(i1, i1 + 1)
    return (hi - lo) / ((hi + lo) / 2.0) * C_KMS


def test_template_resolution(tmp_path: Path) -> None:
    r_template, r_instrument = 300_000.0, float(NEID.resolution.default)
    wave = np.exp(np.linspace(np.log(599.0), np.log(601.0), 40001))
    sigma_v = C_KMS / r_template / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    v = (wave / 600.0 - 1.0) * C_KMS
    flux = 1.0 - 0.6 * np.exp(-0.5 * (v / sigma_v) ** 2)  # a line at the template resolution
    assert _fwhm_kms(wave, flux) == pytest.approx(C_KMS / r_template, rel=1e-3)
    broad = broaden_template(wave, flux, from_resolution=r_template, to_resolution=r_instrument)
    assert _fwhm_kms(wave, broad) == pytest.approx(C_KMS / r_instrument, rel=5e-3)
    assert broad.sum() == pytest.approx(flux.sum(), rel=1e-6)  # flux conserving
    path = tmp_path / "t.dat"
    np.savetxt(path, np.column_stack([wave, flux]), fmt="%.10f")
    obs = _obs(tmp_path, [])
    cfg = FitConfig(
        stellar_model=StellarModelConfig(template_file=str(path), template_resolution=r_template)
    )
    assert _fwhm_kms(*prepare.load_stellar_template(path, obs, cfg)) == pytest.approx(
        C_KMS / r_instrument, rel=5e-3
    )
    # the exposures' header R wins over the instrument default
    obs_50k = _obs(tmp_path, [SimpleNamespace(resolution=50_000.0)] * 2)
    assert _fwhm_kms(*prepare.load_stellar_template(path, obs_50k, cfg)) == pytest.approx(
        C_KMS / 50_000.0, rel=5e-3
    )
    mixed = _obs(tmp_path, [SimpleNamespace(resolution=r) for r in (50_000.0, 130_000.0)])
    with pytest.raises(ConfigError, match="different resolving powers"):
        prepare.load_stellar_template(path, mixed, cfg)
    # lsf.resolution, the start (or fixed) R of the fit, wins over the header R
    pinned = cfg.replace(lsf=cfg.lsf.replace(resolution=80_000.0))
    for observation in (obs_50k, mixed):
        assert _fwhm_kms(
            *prepare.load_stellar_template(path, observation, pinned)
        ) == pytest.approx(C_KMS / 80_000.0, rel=5e-3)
    low = cfg.replace(
        stellar_model=StellarModelConfig(template_file=str(path), template_resolution=50_000.0)
    )
    with pytest.raises(ConfigError, match="must exceed"):
        prepare.load_stellar_template(path, obs, low)


def test_template_frame(tmp_path: Path) -> None:
    tpl = (np.array([599.0, 600.0, 601.0]), np.array([1.0, 0.3, 1.0]))
    wave = np.array([599.9, 600.0, 600.1])
    base: dict[str, Any] = dict(
        layout=_layout(tmp_path), template_path=tmp_path / "t.dat", rv_table=tmp_path / "rv.csv",
        rv_kms={"a.fits": 30.0}, activity_windows=[], bad_regions=[], _template=tpl,
    )  # fmt: skip
    rest = StellarProducts(**base)
    model, _ = TemplateTreatment().order_model(rest, "a.fits", wave)
    np.testing.assert_array_equal(model, stellar_model_for_order(wave, 30.0, tpl))
    topo = StellarProducts(**base, template_frame="topocentric")
    model, _ = TemplateTreatment().order_model(topo, "a.fits", wave)
    np.testing.assert_array_equal(model, stellar_model_for_order(wave, 0.0, tpl))
    assert topo.exposure_rv("a.fits") == 30.0 and topo.placement_rv("a.fits") == 0.0


# ---------------------------------------------------------------------------
# stellar-line windows
# ---------------------------------------------------------------------------
def test_user_line_windows_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = _layout(tmp_path)
    obs = _obs(tmp_path, _exposures(tmp_path, ["e1.fits", "e2.fits"]))
    rv_path = _rv_file(
        tmp_path / "rv.csv", [["file", "rv_kms"], ["e1.fits", "10.0"], ["e2.fits", "-20.0"]]
    )
    windows_file = tmp_path / "windows.txt"
    np.savetxt(windows_file, [[600.0, 600.2], [650.0, 650.5]])
    for module, name in (
        (master_spectrum, "build_master_spectrum"),
        (stellar_lines, "measure_stellar_lines"),
    ):
        monkeypatch.setattr(module, name, _measure_must_not_run)  # no iSpec / master spectrum
    cfg = FitConfig(
        stellar="mask",
        on_missing_stellar_data="error",
        rv=RVConfig(source="user_file", user_rv_file=str(rv_path)),
        masks=FitMasks(stellar_linemask=StellarLineMask(user_file=str(windows_file))),
    )
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)
    table, status = prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))
    products = StellarProducts(
        layout=layout, template_path=tmp_path / "t.dat", rv_table=table,
        rv_kms={}, activity_windows=[], bad_regions=[], status={"rv": status},
    )  # fmt: skip
    assert prepare.ensure_linemasks(obs, cfg, settings, layout, products) == "user_file"
    rest = read_windows(windows_file)
    for name, rv in (("e1.fits", 10.0), ("e2.fits", -20.0)):
        mask = layout.exposure_linemask(name)
        np.testing.assert_allclose(np.loadtxt(mask), shift_windows(rest, rv), atol=1e-5)
        assert "rest-window source: user_file:windows.txt" in mask.read_text()
    # back to the measured windows: masks built from user windows are not reused
    measured = cfg.replace(
        masks=FitMasks(stellar_linemask=StellarLineMask(blend="off")),
        on_missing_stellar_data="generate",
    )
    products.status["rv"] = "reused"
    with pytest.raises(AssertionError, match="must not run"):
        prepare.ensure_linemasks(obs, measured, settings, layout, products)


def test_broadening_ignores_near_duplicate_points() -> None:
    # stitched templates hold near-duplicate wavelengths at chunk joins (1e-6 nm apart)
    wave = np.linspace(600.0, 640.0, 200_001)
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wave - 620.0) / 0.003) ** 2)
    joined_w = np.insert(wave, 100_001, wave[100_000] + 1e-6)
    joined_f = np.insert(flux, 100_001, flux[100_000])
    plain = broaden_template(wave, flux, from_resolution=300_000.0, to_resolution=50_000.0)
    joined = broaden_template(joined_w, joined_f, from_resolution=300_000.0, to_resolution=50_000.0)
    np.testing.assert_allclose(np.delete(joined, 100_001), plain, atol=1e-6)


def test_stale_mask_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = _layout(tmp_path)
    obs = _obs(tmp_path, _exposures(tmp_path, ["e1.fits", "e2.fits"]))
    rv_path = _rv_file(
        tmp_path / "rv.csv", [["file", "rv_kms"], ["e1.fits", "10.0"], ["e2.fits", "-20.0"]]
    )
    strict = FitConfig(
        stellar="mask",
        on_missing_stellar_data="error",
        rv=RVConfig(source="user_file", user_rv_file=str(rv_path)),
        masks=FitMasks(stellar_linemask=StellarLineMask(blend="off")),
    )
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)

    def products() -> StellarProducts:
        table, status = prepare.ensure_rv_table(obs, strict, layout, template=(None, None))
        return StellarProducts(
            layout=layout, template_path=tmp_path / "t.dat", rv_table=table, rv_kms={},
            activity_windows=[], bad_regions=[], status={"rv": status},
        )  # fmt: skip

    # no masks yet: "not found"
    with pytest.raises(FileNotFoundError, match="stellar-line masks not found"):
        prepare.ensure_linemasks(obs, strict, settings, layout, products())
    # measured-window masks without a record of their settings are not reused
    rows = rvmod.read_rv_table(layout.rv_table)
    write_exposure_linemasks(rows, [(600.0, 600.2)], "measured:rest.txt", layout)
    with pytest.raises(DataError, match=r"no record of how they were built.*'generate'"):
        prepare.ensure_linemasks(obs, strict, settings, layout, products())
    # with the record of the same settings, and the same (user) RVs, they are reused
    record = prepare._linemask_settings(strict, products())
    layout.linemask_settings.write_text(json.dumps(record))
    assert prepare.ensure_linemasks(obs, strict, settings, layout, products()) == "reused"
    # other settings: the masks exist but are stale, and the message names the change
    deeper = strict.replace(stellar_model=StellarModelConfig(line_min_depth=0.2))
    with pytest.raises(DataError, match=r"other settings \(line_min_depth 0.1 -> 0.2\)"):
        prepare.ensure_linemasks(obs, deeper, settings, layout, products())
    # the RVs changed: the masks exist but are stale, and the message says why
    _rv_file(rv_path, [["file", "rv_kms"], ["e1.fits", "11.0"], ["e2.fits", "-20.0"]])
    with pytest.raises(DataError, match=r"shifted by RV \+10.0000 km/s.*now gives \+11.0000"):
        prepare.ensure_linemasks(obs, strict, settings, layout, products())
    # masks built from user windows are stale for the measured windows
    layout.linemask_settings.write_text(json.dumps({**record, "source": "user_file"}))
    with pytest.raises(DataError, match=r"source 'user_file' -> 'master'"):
        prepare.ensure_linemasks(obs, strict, settings, layout, products())


# ---------------------------------------------------------------------------
# stellar-line masks: from_spectrum
# ---------------------------------------------------------------------------
def _mask_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    layout = _layout(tmp_path)
    obs = _obs(tmp_path, _exposures(tmp_path, ["e1.fits", "e2.fits"]))
    rv_path = _rv_file(
        tmp_path / "rv.csv", [["file", "rv_kms"], ["e1.fits", "10.0"], ["e2.fits", "-20.0"]]
    )
    w = np.linspace(600.0, 601.0, 1001)
    f = np.ones_like(w)
    f[400:410] = 0.5  # one stellar line
    calls: list[str] = []
    monkeypatch.setattr(prepare, "_measure_master_lines", lambda *a, **k: calls.append("m"))
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)
    return layout, obs, rv_path, (w, f), calls, settings


def _products(
    layout: RunLayout, obs: Any, cfg: FitConfig, template: Any, frame: str = "rest"
) -> StellarProducts:
    table, status = prepare.ensure_rv_table(obs, cfg, layout, template=(None, None))
    return StellarProducts(
        layout=layout, template_path=Path("/t/tmpl.dat"), rv_table=table, rv_kms={},
        activity_windows=[], bad_regions=[], status={"rv": status},
        template_frame=frame, _template=template,
    )  # fmt: skip


def test_linemask_from_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    layout, obs, rv_path, template, calls, settings = _mask_setup(tmp_path, monkeypatch)
    lm = StellarLineMask(from_spectrum="template", pad_kms=0.0)
    cfg = FitConfig(
        stellar="mask",
        rv=RVConfig(source="user_file", user_rv_file=str(rv_path)),
        masks=FitMasks(stellar_linemask=lm),
    )
    products = _products(layout, obs, cfg, template)
    with caplog.at_level("WARNING", logger="gtellcorr.preprocessing.prepare"):
        assert prepare.ensure_linemasks(obs, cfg, settings, layout, products) == "generated"
    assert calls == [] and "not suitable" not in caplog.text  # no master, no warning
    rest = [(template[0][400], template[0][409])]
    for name, rv in (("e1.fits", 10.0), ("e2.fits", -20.0)):
        mask = layout.exposure_linemask(name)
        np.testing.assert_allclose(
            np.loadtxt(mask).reshape(-1, 2), shift_windows(rest, rv), atol=1e-5
        )
        assert "rest-window source: template:tmpl.dat" in mask.read_text()
    assert json.loads(layout.linemask_settings.read_text())["source"] == "template"
    assert prepare.ensure_linemasks(obs, cfg, settings, layout, products) == "reused"
    # a user-supplied template: used, with a warning about observed spectra
    user = cfg.replace(stellar_model=StellarModelConfig(template_file="/t/tmpl.dat"))
    layout.linemask_settings.unlink()
    with caplog.at_level("WARNING", logger="gtellcorr.preprocessing.prepare"):
        assert prepare.ensure_linemasks(obs, user, settings, layout, products) == "generated"
    assert "without telluric correction is not suitable" in caplog.text
    # a topocentric template: the windows are not shifted
    topo = _products(layout, obs, cfg, template, frame="topocentric")
    assert prepare.ensure_linemasks(obs, cfg, settings, layout, topo) == "generated"
    for name in ("e1.fits", "e2.fits"):
        got = np.loadtxt(layout.exposure_linemask(name)).reshape(-1, 2)
        np.testing.assert_allclose(got, rest, atol=1e-5)


def test_linemask_from_master_reuses_matching_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, obs, rv_path, template, calls, settings = _mask_setup(tmp_path, monkeypatch)
    cfg = FitConfig(
        stellar="mask",
        rv=RVConfig(source="user_file", user_rv_file=str(rv_path)),
        masks=FitMasks(stellar_linemask=StellarLineMask(blend="off")),
    )
    products = _products(layout, obs, cfg, template)
    np.savetxt(layout.linemask_rest, [[600.4, 600.41]])  # as the measurement writes it
    # no record yet: the lines are measured again
    assert prepare.ensure_linemasks(obs, cfg, settings, layout, products) == "generated"
    assert calls == ["m"]
    # masks deleted, same settings: the measured windows are placed again, not re-measured
    for name in ("e1.fits", "e2.fits"):
        layout.exposure_linemask(name).unlink()
    assert prepare.ensure_linemasks(obs, cfg, settings, layout, products) == "generated"
    assert calls == ["m"]
    # other line settings: measured again
    deeper = cfg.replace(stellar_model=StellarModelConfig(line_min_depth=0.3))
    assert prepare.ensure_linemasks(obs, deeper, settings, layout, products) == "generated"
    assert calls == ["m", "m"]
