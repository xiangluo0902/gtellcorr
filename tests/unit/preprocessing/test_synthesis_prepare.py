from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.target import Star
from gtellcorr.data.products.layout import RunLayout
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.preprocessing import prepare, telluric_mask
from gtellcorr.preprocessing.config import StellarModelConfig
from gtellcorr.preprocessing.prepare import StellarProducts
from gtellcorr.preprocessing.synthesis import (
    StellarParameters,
    Synthesis,
    generate_template,
    load_template,
    stellar_model_for_order,
    stellar_template,
    template_filename,
)
from gtellcorr.preprocessing.treatments import MaskTreatment, TemplateTreatment, stellar_treatments
from gtellcorr.settings import Settings

STAR = Star(name="TOI-2076", teff=5180.0, logg=4.608, feh=0.0, vsini_kms=5.0, vsys_kms=-12.5)
NEID = instruments.get("NEID")


def test_synthesis_parameters_and_names(tmp_path: Path) -> None:
    settings = Settings(
        resource_root=tmp_path,
        cache_dir=tmp_path,
        stellar_linelist=tmp_path / "ll",
        ispec_dir=tmp_path / "is",
    )
    syn = Synthesis.from_config(STAR, StellarModelConfig(), settings)
    assert syn.star == StellarParameters("TOI-2076", 5180.0, 4.608, 0.0, 5.0, -12.5)
    assert syn.linelist == str(tmp_path / "ll") and syn.pysme["atmosphere"] == "marcs2012"
    name = template_filename(syn, wave_range_nm=(357.09, 1105.0), resolution=110000.0)
    assert (
        name
        == "TOI-2076_pySME-VALD_Teff5180_logg4.608_MH+0.0_vsini5.0kms_R110000_357-1105nm_rest.dat"
    )
    with pytest.raises(ConfigError, match="teff"):
        StellarParameters.from_star(Star(name="x"))
    with pytest.raises(ConfigError, match="unknown stellar-model backend"):
        stellar_template(syn, (600.0, 601.0), backend="other")
    ispec_syn = Synthesis(star=syn.star, backend="ispec", linelist=None, ispec_dir=None)
    with pytest.raises(ConfigError, match="ispec_dir setting"):
        generate_template(ispec_syn, tmp_path / "t.dat", wave_range_nm=(600, 610), resolution=1e5)
    no_list = Synthesis(star=syn.star, backend="pysme", linelist=None, ispec_dir=None)
    with pytest.raises(ConfigError, match="stellar_linelist setting"):
        generate_template(no_list, tmp_path / "t.dat", wave_range_nm=(600, 610), resolution=1e5)
    ispec_config = Synthesis.from_config(STAR, StellarModelConfig(backend="ispec"), settings)
    assert template_filename(
        ispec_config, wave_range_nm=(357.09, 1105.0), resolution=110000.0
    ).startswith("TOI-2076_iSpec-spectrum_Teff5180_")
    existing = tmp_path / "exists.dat"
    existing.write_text("1 1\n")
    assert (
        generate_template(ispec_syn, existing, wave_range_nm=(600, 610), resolution=1e5) == existing
    )


def test_template_model(tmp_path: Path) -> None:
    path = tmp_path / "t.dat"
    np.savetxt(path, np.column_stack([[600.0, 601.0, 602.0], [1.0, 0.5, 1.0]]))
    template = load_template(path)
    model = stellar_model_for_order([599.0, 601.0, 603.0], 0.0, template)
    assert model.tolist() == [1.0, 0.5, 1.0]


def _products(tmp_path: Path) -> StellarProducts:
    layout = RunLayout.for_run(tmp_path, target="T", site_token="S", instrument="I", date="D")
    np.savetxt(layout.exposure_linemask("a.fits"), [[600.0, 600.1]])
    tpl = tmp_path / "t.dat"
    np.savetxt(tpl, np.column_stack([[599.0, 600.05, 601.0], [1.0, 0.3, 1.0]]))
    return StellarProducts(
        layout=layout, template_path=tpl, rv_table=tmp_path / "rv.csv",
        rv_kms={"a.fits": 0.0}, activity_windows=[(600.5, 600.6)], bad_regions=[],
        strong_windows=[(600.8, 600.9)],
    )  # fmt: skip


def test_treatments(tmp_path: Path) -> None:
    products = _products(tmp_path)
    wave = np.array([600.05, 600.3, 600.55, 600.85])
    model, excl = MaskTreatment().order_model(products, "a.fits", wave)
    assert model is None and excl.tolist() == [True, False, True, True]
    model, excl = TemplateTreatment().order_model(products, "/raw/a.fits", wave)
    assert (
        excl.tolist() == [False, False, True, True]
        and model is not None
        and model[0] == pytest.approx(0.3)
    )
    assert products.exposure_rv("a") == 0.0
    with pytest.raises(KeyError, match="no RV-table row"):
        products.exposure_rv("b.fits")
    assert set(stellar_treatments.names()) == {"mask", "template"}


class _Obs:
    def __init__(self, tmp_path: Path) -> None:
        self.instrument = NEID
        self.target = STAR
        self.transit = None
        self.directory = tmp_path
        self.exposures: list[Any] = []


def test_prepare_reuse_and_policies(tmp_path: Path) -> None:
    settings = Settings(
        resource_root=tmp_path, cache_dir=tmp_path, stellar_linelist=tmp_path / "ll"
    )
    layout = RunLayout.for_run(
        tmp_path, target="TOI-2076", site_token="KPNO", instrument="WIYN/NEID", date="D"
    )
    obs = _Obs(tmp_path)
    fit = FitConfig(on_missing_stellar_data="error")
    path = prepare.template_path(obs, fit, settings, layout)  # type: ignore[arg-type]
    assert path.parent == layout.template_dir and path.name.endswith("_rest.dat")
    with pytest.raises(FileNotFoundError, match="not found"):
        prepare.ensure_template(obs, fit, settings, layout)  # type: ignore[arg-type]
    np.savetxt(path, np.column_stack([[500.0, 700.0], [1.0, 1.0]]))
    assert prepare.ensure_template(obs, fit, settings, layout) == (path, "reused")  # type: ignore[arg-type]
    explicit = fit.replace(stellar_model=StellarModelConfig(template_file=str(tmp_path / "x.dat")))
    assert prepare.template_path(obs, explicit, settings, layout) == tmp_path / "x.dat"  # type: ignore[arg-type]
    # a missing user template is an error, never synthesised into the user's path
    for policy in ("generate", "error"):
        missing = explicit.replace(on_missing_stellar_data=policy)
        with pytest.raises(FileNotFoundError, match="template_file not found"):
            prepare.ensure_template(obs, missing, settings, layout)  # type: ignore[arg-type]
    assert not (tmp_path / "x.dat").exists()
    with open(layout.rv_table, "w", newline="") as fh:
        csv.writer(fh).writerows([["file", "bjd", "rv_kms"], ["a.fits", "1", "2.5"]])
    products = prepare.prepare_stellar(obs, fit.replace(stellar="template"), settings, layout)  # type: ignore[arg-type]
    assert products.status == {"template": "reused", "rv": "reused"} and products.rv_kms == {
        "a.fits": 2.5
    }
    with pytest.raises(FileNotFoundError, match="stellar-line masks not found"):
        prepare.prepare_stellar(obs, fit.replace(stellar="mask"), settings, layout)  # type: ignore[arg-type]
    np.savetxt(layout.exposure_linemask("a.fits"), [[1.0, 2.0]])
    # a mask without a record of how it was built is not reused
    with pytest.raises(DataError, match="no record of how they were built"):
        prepare.prepare_stellar(obs, fit.replace(stellar="mask"), settings, layout)  # type: ignore[arg-type]
    record = prepare._linemask_settings(fit, products)
    layout.linemask_settings.write_text(json.dumps(record))
    products = MaskTreatment().prepare(obs, fit, settings, layout)  # type: ignore[arg-type]
    assert products.status["linemask"] == "reused"
    assert TemplateTreatment().prepare(obs, fit, settings, layout).status["rv"] == "reused"  # type: ignore[arg-type]
    obs.target = None
    with pytest.raises(ConfigError, match="target"):
        prepare.template_path(obs, fit, settings, layout)  # type: ignore[arg-type]


def test_strong_mask_helpers() -> None:
    token = telluric_mask.filename_token(
        wavelength_range_nm=(357.09, 1105.0), resolution=110000.0, backend="era5_cdsapi",
        upper_source="msis00", time=datetime(2022, 2, 13, 7, 35, 5, tzinfo=UTC), airmass=1.899,
        threshold=0.05,
    )  # fmt: skip
    assert token == "w357p09-1105nm_R110000_era5_cdsapi-msis00-20220213T073505_am1p899_thr0p05"
    assert telluric_mask._chunk_ranges(0.0, 70.0, 30.0) == [(0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]
    with pytest.raises(ValueError, match="positive"):
        telluric_mask._chunk_ranges(0.0, 1.0, 0.0)
    assert telluric_mask.component_molecules(NEID) == ("h2o", "o2")
    ref = dict.fromkeys(
        ("h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "no", "so2", "no2", "nh3", "hno3"), 2.0
    )
    total = telluric_mask._reference_abundances(ref, relcol=0.9)
    assert total["h2o"] == 0.9 and total["o2"] == 2.0
    only = telluric_mask._reference_abundances(ref, relcol=0.5, molecule="o2")
    assert only["o2"] == 1.0 and only["h2o"] == 0.0
    with pytest.raises(ValueError, match="unsupported"):
        telluric_mask._reference_abundances(ref, molecule="xx")
    wave = np.array([1.0, 2.0, 3.0, 4.0])
    assert telluric_mask.strong_telluric_windows(wave, np.array([1.0, 0.01, 0.02, 1.0])) == [
        (2.0, 3.0)
    ]
    with pytest.raises(ValueError, match="same shape"):
        telluric_mask.strong_telluric_windows(wave, np.ones(3))


def _fake_pysme(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    """Stand-in ``pysme`` modules that record what the synthesis sets."""
    import sys
    import types

    class Abund:
        def __init__(self, monh: float = 0.0, pattern: str = "solar") -> None:
            self.monh, self.pattern = monh, pattern

    class Atmo:
        source = method = None

    class Nlte:
        def set_nlte(self, elem: str, grid: str) -> None:
            seen.setdefault("nlte", {})[elem] = grid

    class SME:
        def __init__(self) -> None:
            self.atmo, self.nlte, self.wave = Atmo(), Nlte(), None

    def synthesize(sme: Any) -> Any:
        seen["sme"] = sme
        grid = sme.wave[0] if sme.wave is not None else np.linspace(*sme.wran, 5)
        sme.wave, sme.synth = [np.asarray(grid)], [np.ones(len(grid))]
        return sme

    class Vald(dict):  # type: ignore[type-arg]
        def __init__(self, path: str) -> None:
            super().__init__(gflog=np.zeros(1), excit=np.zeros(1))

    modules = {
        "pysme": types.ModuleType("pysme"),
        "pysme.abund": types.SimpleNamespace(Abund=Abund),
        "pysme.sme": types.SimpleNamespace(SME_Structure=SME),
        "pysme.synthesize": types.SimpleNamespace(synthesize_spectrum=synthesize),
        "pysme.linelist": types.ModuleType("pysme.linelist"),
        "pysme.linelist.vald": types.SimpleNamespace(ValdFile=Vald),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_pysme_settings_reach_the_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gtellcorr.preprocessing.config import PysmeSettings
    from gtellcorr.preprocessing.synthesis import _pysme_synth

    seen: dict[str, Any] = {}
    _fake_pysme(monkeypatch, seen)
    settings = Settings(
        resource_root=tmp_path, cache_dir=tmp_path, stellar_linelist=tmp_path / "ll"
    )
    default = Synthesis.from_config(STAR, StellarModelConfig(), settings)
    _pysme_synth(default, 600.0, 601.0, None)
    sme = seen["sme"]
    assert (sme.abund.pattern, sme.abund.monh) == ("grevesse2007", STAR.feh)  # PySME's default
    assert sme.atmo.source == "marcs2012.sav" and len(sme.wave[0]) == 5  # adaptive grid
    custom = StellarModelConfig(
        pysme=PysmeSettings(abundances="asplund2009", wave_step_nm=0.01, nlte={"Fe": "g.grd"})
    )
    w, _flux = _pysme_synth(Synthesis.from_config(STAR, custom, settings), 600.0, 601.0, None)
    sme = seen["sme"]
    assert sme.abund.pattern == "asplund2009" and seen["nlte"] == {"Fe": "g.grd"}
    np.testing.assert_allclose(np.diff(w), 0.01)  # the fixed output grid, in nm
