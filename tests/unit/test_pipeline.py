from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gtellcorr import pipeline as runmod
from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.target import Star
from gtellcorr.data.products.fits_schema import write_exposure_fits
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.fitting.results import OrderFitResult
from gtellcorr.settings import Settings


@dataclass
class _Exposure:
    path: Path
    time_utc: datetime = datetime(2022, 2, 13, 7, 35, tzinfo=UTC)


STAR = Star(name="T")


class _Obs:
    def __init__(self, exposures: list[Any], target: Star | None = STAR) -> None:
        self.instrument = instruments.get("NEID")
        self.exposures = exposures
        self.target = target
        self.transit = None


class _Model:
    class resources:
        local_fingerprint = "fp"


def _pipeline(tmp_path: Path, reuse: str = "matching", **kw: Any) -> tuple[runmod.Pipeline, Any]:
    raw = tmp_path / "neidL2_x.fits"
    raw.write_bytes(b"spectrum")
    exposure = _Exposure(raw)
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)
    pipe = runmod.Pipeline(
        _Obs([exposure]), FitConfig(relcol_mode="joint_orders"), output_dir=tmp_path / "out",
        settings=settings, reuse=reuse, telluric_model=_Model(), **kw,  # type: ignore[arg-type]
    )  # fmt: skip
    return pipe, exposure


def _result(order: int = 1) -> OrderFitResult:
    w = np.linspace(600.0, 601.0, 10)
    ones = np.ones(10)
    return OrderFitResult(
        order=order, wave_input=w, wave_fit=w, flux_obs=ones, err=ones, tell_model=ones,
        flux_corrected=ones, status="joint_ok",
    )  # fmt: skip


def test_layout_and_reuse(tmp_path: Path) -> None:
    pipe, exposure = _pipeline(tmp_path)
    assert pipe.layout.root == tmp_path / "out" / "T" / "KPNO_NEID" / "20220213"
    path = pipe.product_path(exposure)  # type: ignore[arg-type]
    assert path.name == "neidL2_x_joint_corrected_all_orders.fits"
    assert not pipe.reusable(exposure)  # type: ignore[arg-type]
    write_exposure_fits(
        path, [_result()], exposure="x", instrument="NEID", correction_mode="joint_orders"
    )
    assert not pipe.reusable(exposure)  # no provenance cards: an older result file
    assert _pipeline(tmp_path, reuse="any")[0].reusable(exposure)  # type: ignore[arg-type]
    write_exposure_fits(
        path, [_result()], exposure="x", instrument="NEID", correction_mode="joint_orders",
        provenance=pipe._provenance(exposure),  # type: ignore[arg-type]
    )  # fmt: skip
    assert pipe.reusable(exposure)  # type: ignore[arg-type]
    assert not _pipeline(tmp_path, reuse="never")[0].reusable(exposure)  # type: ignore[arg-type]
    other = runmod.Pipeline(
        pipe.observation, FitConfig(relcol_mode="joint_orders", stellar="template"),
        output_dir=tmp_path / "out", settings=pipe.settings, telluric_model=_Model(),  # type: ignore[arg-type]
    )  # fmt: skip
    assert not other.reusable(exposure)  # type: ignore[arg-type]  # different configuration
    assert runmod.config_hash(FitConfig()) == runmod.config_hash(FitConfig())
    assert runmod.config_hash(FitConfig()) != runmod.config_hash(FitConfig(stellar="template"))


def test_config_hash_ignores_documentation(monkeypatch: pytest.MonkeyPatch) -> None:
    from gtellcorr.config import toml_format
    from gtellcorr.config.toml_io import dumps_config

    before, commented = runmod.config_hash(FitConfig()), dumps_config(FitConfig())
    real = toml_format.field_docs
    monkeypatch.setattr(
        toml_format, "field_docs", lambda cls: {k: v + " (edited)" for k, v in real(cls).items()}
    )
    assert dumps_config(FitConfig()) != commented  # the documentation did change
    assert runmod.config_hash(FitConfig()) == before


def test_requirements(tmp_path: Path) -> None:
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)
    with pytest.raises(DataError, match="no exposures"):
        runmod.Pipeline(_Obs([]), FitConfig(), settings=settings)  # type: ignore[arg-type]
    no_target = _Obs([_Exposure(tmp_path)], target=None)
    with pytest.raises(DataError, match="needs a target"):
        runmod.Pipeline(no_target, FitConfig(), settings=settings)  # type: ignore[arg-type]
    events: list[runmod.Progress] = []
    pipe, _ = _pipeline(tmp_path, progress=events.append)
    pipe._emit("stage", "message", 1, 2)
    assert events == [runmod.Progress("stage", "message", 1, 2)]


def test_helpers() -> None:
    results = [_result(1), _result(2)]
    results[0].resolution, results[1].resolution = 1.0e5, 1.2e5
    assert runmod._median(results, "resolution") == 1.1e5
    assert runmod._median(results, "lsf_shape") is None
    assert runmod._median_shift(results) is None
    results[0].wave_shift, results[1].wave_shift = (0.1, 0.2), (0.3, 0.0)
    assert runmod._median_shift(results) == pytest.approx((0.2, 0.1))

    @dataclass
    class Raw:
        wave_input: Any
        wave_fit: Any
        flux: Any
        err: Any
        status: str

    raw = Raw(np.arange(3.0), np.full(3, np.nan), np.ones(3), np.ones(3), "no_valid_wavelength")
    placeholder = runmod._placeholder(5, raw, "mask")
    assert placeholder.status == "no_valid_wavelength" and placeholder.exclusion.all()
    assert np.isnan(placeholder.tell_model).all()
    s = Raw(np.arange(3.0), np.arange(3.0) + 600, np.full(3, 2.0), np.ones(3), "")
    identity = runmod._identity(3, s, None, None, "template")
    assert identity.status == "no_fit_molecules" and not identity.exclusion.any()
    np.testing.assert_array_equal(identity.flux_corrected, s.flux)


def test_a_supplied_model_must_use_the_observations_formula(tmp_path: Path) -> None:
    from gtellcorr.data.observation.config import Frames

    class Ciddor(_Model):
        vacuum_air_formula = "ciddor1996"

    exposure = _Exposure(tmp_path / "neidL2_x.fits")
    settings = Settings(resource_root=tmp_path, cache_dir=tmp_path)
    with pytest.raises(ConfigError, match="vacuum_air_formula 'ciddor1996'"):
        runmod.Pipeline(
            _Obs([exposure]), FitConfig(), output_dir=tmp_path / "out", settings=settings,
            telluric_model=Ciddor(),  # type: ignore[arg-type]
        )  # fmt: skip
    config = FitConfig(frames=Frames(vacuum_air_formula="ciddor1996"))
    runmod.Pipeline(
        _Obs([exposure]), config, output_dir=tmp_path / "out", settings=settings,
        telluric_model=Ciddor(),  # type: ignore[arg-type]
    )  # fmt: skip


def test_result_files_name_no_local_directory(tmp_path: Path) -> None:
    """RUN_METADATA and run_manifest.json give paths relative to the output directory or
    the run directory, and the resources outside them by name only."""
    model = SimpleNamespace(
        resources=SimpleNamespace(
            bundle=tmp_path / "resources" / "lblrtm" / "gnu_double_v4",
            local_fingerprint="fp",
            setup_id="setup",
        ),
        timeout_s=60.0,
        config=SimpleNamespace(wave_margin_nm=20.0),
        tau_cache=tmp_path / "caches" / "tau_cache",
    )
    pipe, exposure = _pipeline(tmp_path)
    pipe._model = model  # type: ignore[assignment]
    pipe.stellar = SimpleNamespace(rv_table=pipe.layout.rv_table)  # type: ignore[assignment]
    exposure = SimpleNamespace(
        path=exposure.path, time_utc=exposure.time_utc, airmass=1.2, airmass_source="fits_header"
    )
    meta = pipe._run_metadata(exposure, [], {})  # type: ignore[arg-type]
    assert meta["run_root"] == "T/KPNO_NEID/20220213"
    assert meta["rv_table"] == pipe.layout.rv_table.relative_to(pipe.layout.root).as_posix()
    assert meta["lblrtm_bundle"] == "gnu_double_v4"
    assert meta["tau_cache"] == "tau_cache"
    assert not any(str(tmp_path) in str(value) for value in meta.values())

    summary = pipe.layout.root / "telluric_corr" / "data" / "summary.csv"
    pipe.layout.root.mkdir(parents=True, exist_ok=True)
    pipe._write_manifest(summary, [])
    text = (pipe.layout.root / "run_manifest.json").read_text()
    assert json.loads(text)["summary_path"] == "telluric_corr/data/summary.csv"
    assert str(tmp_path) not in text
