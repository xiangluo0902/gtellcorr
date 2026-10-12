from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

import gtellcorr.config.toml_format as toml_format
import gtellcorr.config.toml_io as toml_io
from gtellcorr.config.fit_config import FitConfig
from gtellcorr.config.run_config import RunConfig
from gtellcorr.config.toml_format import CURRENT_SCHEMA_VERSION, dumps, loads
from gtellcorr.config.toml_io import dumps_config
from gtellcorr.errors import ConfigError


def _run() -> RunConfig:
    return RunConfig.from_dict(
        {
            "data": {
                "directory": "/data/NEID",
                "instrument": "WIYN/NEID",
                "include": ["T0730"],
                "target": {"name": "TOI-2076", "teff": 5180.0},
                "site": {"lat_deg": 31.96, "lon_deg": -111.6, "alt_m": 2096.0},
            },
            "fit": {
                "stellar": "template",
                "relcol_mode": "joint_orders",
                "molecules": {"h2o": {"anchor_orders": [78, 79]}, "co2": {"anchor_orders": []}},
                "stellar_model": {"pysme": {"nlte": {"Fe": "marcs2012_Fe2016.grd"}}},
            },
        }
    )


@pytest.mark.parametrize("comments", [True, False])
def test_run_config_round_trip(tmp_path: Path, comments: bool) -> None:
    run = _run()
    path = tmp_path / "run.toml"
    toml_io.save_run_config(run, path, comments=comments)
    assert RunConfig.load(path) == run
    text = path.read_text()
    assert text.count("# ") > 50 if comments else "# " not in text.split("schema_version")[1]


def test_fit_config_round_trip_and_partial_file(tmp_path: Path) -> None:
    cfg = FitConfig(stellar="template")
    path = tmp_path / "fit.toml"
    cfg.save(path)
    assert FitConfig.load(path) == cfg
    partial = tmp_path / "partial.toml"
    partial.write_text("schema_version = 1\n[fit.continuum]\norder = 3\n")
    assert FitConfig.load(partial) == FitConfig().replace(continuum={"order": 3})
    # a full run file also works as a fit file
    run_path = tmp_path / "run.toml"
    _run().save(run_path)
    assert FitConfig.load(run_path) == _run().fit


def test_fit_config_file_carries_comments() -> None:
    text = dumps_config(FitConfig())
    assert "# Polynomial degree of the continuum." in text
    assert "# How stellar lines are handled" in text


def test_long_scalar_arrays_are_packed() -> None:
    text = dumps({"t": {"orders": list(range(122))}})
    body = text.split("orders = [")[1].split("]")[0]
    assert 5 < body.count("\n") < 20  # several numbers per line
    assert max(len(line) for line in text.splitlines()) <= 88
    assert tomllib.loads(text)["t"]["orders"] == list(range(122))


def test_output_is_compact_and_readable() -> None:
    text = dumps_config(FitConfig(), comments=False)
    assert "sigma_clip = [2.0, 3.0]" in text
    assert '[fit.molecules."no+"]' in text  # non-bare key is quoted
    assert len(text.splitlines()) < 500
    assert max(len(line) for line in text.splitlines()) <= 88


def test_writer_handles_special_values() -> None:
    tables: dict[str, Any] = {
        "t": {
            "quote": 'a "b" \\ c',
            "unicode": "λ-Å",
            "inf": float("inf"),
            "empty": [],
            "inline": [{"a": 1}],
            "long": list(range(40)),
        },
        "empty_table": {},
    }
    text = dumps(tables)
    assert tomllib.loads(text) == {"schema_version": CURRENT_SCHEMA_VERSION, **tables}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[fit]\n", "missing 'schema_version'"),
        ("schema_version = 'one'\n", "must be a positive integer"),
        ("schema_version = 0\n", "must be a positive integer"),
        ("schema_version = 99\n", "newer than this gtellcorr supports"),
        ("schema_version = \n", "invalid TOML"),
    ],
)
def test_schema_version_errors(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        loads(text)


def test_migration_chain_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    def v1_to_v2(doc: dict[str, Any]) -> dict[str, Any]:
        doc["fit"]["relcol_mode"] = doc["fit"].pop("mode")
        return doc

    monkeypatch.setattr(toml_format, "CURRENT_SCHEMA_VERSION", 2)
    monkeypatch.setitem(toml_format.MIGRATIONS, 1, v1_to_v2)
    doc = loads('schema_version = 1\n[fit]\nmode = "joint_orders"\n')
    assert doc == {"fit": {"relcol_mode": "joint_orders"}}


def test_load_errors_name_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("schema_version = 1\n[data]\ndirectory = '/d'\n[fit.continuum]\norder = -1\n")
    with pytest.raises(ConfigError, match=r"bad\.toml: fit\.continuum\.order: must be >= 0"):
        RunConfig.load(bad)
    with pytest.raises(ConfigError, match=r"bad\.toml: fit\.continuum\.order"):
        FitConfig.load(bad)
    stray = tmp_path / "stray.toml"
    stray.write_text("schema_version = 1\n[fitt]\n")
    with pytest.raises(ConfigError, match="unknown table 'fitt'"):
        FitConfig.load(stray)
    with pytest.raises(ConfigError, match="cannot read configuration file"):
        RunConfig.load(tmp_path / "missing.toml")


def test_schema_1_to_2_drops_unused_pysme_settings(caplog: pytest.LogCaptureFixture) -> None:
    from gtellcorr.config.run_config import RunConfig

    v1 = """schema_version = 1
[data]
directory = "/d"
[fit.stellar_model.pysme]
atmosphere = "marcs2012"
abundances = "asplund2009"
wave_step_nm = 0.001
cscale_flag = "linear"
vrad_flag = "whole"
mgb_wing_windows_nm = [[516.35, 516.66]]
"""
    with caplog.at_level("INFO", logger="gtellcorr.config.toml_format"):
        doc = loads(v1)
    pysme = doc["fit"]["stellar_model"]["pysme"]
    assert pysme == {"atmosphere": "marcs2012"}
    assert "dropped unused fit.stellar_model.pysme settings" in caplog.text
    # version 1 never used them: the effective synthesis is unchanged
    loaded = RunConfig.from_dict(doc).fit.stellar_model.pysme
    assert (loaded.abundances, loaded.wave_step_nm) == ("grevesse2007", None)
    # a version-2 file keeps what it sets
    v2 = v1.replace("schema_version = 1", "schema_version = 2")
    v2 = v2.replace('cscale_flag = "linear"\nvrad_flag = "whole"\n', "")
    v2 = v2.replace("mgb_wing_windows_nm = [[516.35, 516.66]]\n", "")
    kept = RunConfig.from_dict(loads(v2)).fit.stellar_model.pysme
    assert (kept.abundances, kept.wave_step_nm) == ("asplund2009", 0.001)


def test_schema_1_to_2_maps_fallback_mask_to_error(caplog: pytest.LogCaptureFixture) -> None:
    from gtellcorr.config.run_config import RunConfig

    v1 = (
        'schema_version = 1\n[data]\ndirectory = "/d"\n'
        '[fit]\non_missing_stellar_data = "fallback_mask"\n'
    )
    with caplog.at_level("INFO", logger="gtellcorr.config.toml_format"):
        config = RunConfig.from_dict(loads(v1))
    assert config.fit.on_missing_stellar_data == "error"
    assert '"fallback_mask" is now "error"' in caplog.text
    v2 = v1.replace("schema_version = 1", "schema_version = 2")
    with pytest.raises(ConfigError, match="must be one of 'generate', 'error'"):
        RunConfig.from_dict(loads(v2))


def test_schema_2_to_3_renames_strong_mask_relcol(caplog: pytest.LogCaptureFixture) -> None:
    from gtellcorr.config.run_config import RunConfig

    v2 = 'schema_version = 2\n[data]\ndirectory = "/d"\n[fit.masks.strong_telluric]\nrelcol = 1.4\n'
    with caplog.at_level("INFO", logger="gtellcorr.config.toml_format"):
        config = RunConfig.from_dict(loads(v2))
    assert config.fit.masks.strong_telluric.h2o_relcol == 1.4
    assert "relcol is now h2o_relcol" in caplog.text
    v3 = v2.replace("schema_version = 2", "schema_version = 3")
    with pytest.raises(ConfigError, match="relcol"):
        RunConfig.from_dict(loads(v3))


def test_schema_3_to_4_drops_the_original_lsf_options(caplog: pytest.LogCaptureFixture) -> None:
    from gtellcorr.config.run_config import RunConfig

    head = 'schema_version = 3\n[data]\ndirectory = "/d"\n'
    current = head + '[fit.lsf]\nconvolution = "fixed_grid"\n'
    current += "[fit.solver]\nscale_parameters = true\n"
    with caplog.at_level("INFO", logger="gtellcorr.config.toml_format"):
        config = RunConfig.from_dict(loads(current))
    assert config.fit == RunConfig.from_dict(loads(head)).fit
    assert "dropped fit.lsf.convolution" in caplog.text
    assert "dropped fit.solver.scale_parameters" in caplog.text
    with pytest.raises(ConfigError, match="'reference' is no longer supported"):
        loads(head + '[fit.lsf]\nconvolution = "reference"\n')
    with pytest.raises(ConfigError, match="scale_parameters = false is no longer supported"):
        loads(head + "[fit.solver]\nscale_parameters = false\n")
    # the keys are gone from schema 4
    v4 = current.replace("schema_version = 3", "schema_version = 4")
    with pytest.raises(ConfigError, match=r"convolution|solver"):
        RunConfig.from_dict(loads(v4))


def test_schema_3_to_4_drops_the_strong_mask_sampling(caplog: pytest.LogCaptureFixture) -> None:
    from gtellcorr.config.run_config import RunConfig

    head = 'schema_version = 3\n[data]\ndirectory = "/d"\n'
    v3 = head + "[fit.masks.strong_telluric]\nthreshold = 0.1\nn_grid_per_nm = 50\n"
    with caplog.at_level("INFO", logger="gtellcorr.config.toml_format"):
        config = RunConfig.from_dict(loads(v3))
    assert config.fit.masks.strong_telluric.threshold == 0.1
    assert "dropped fit.masks.strong_telluric.n_grid_per_nm" in caplog.text
    v4 = v3.replace("schema_version = 3", "schema_version = 4")
    with pytest.raises(ConfigError, match="n_grid_per_nm: unknown parameter"):
        RunConfig.from_dict(loads(v4))
