from __future__ import annotations

from pathlib import Path

import pytest

from gtellcorr.errors import ResourceError
from gtellcorr.settings import Settings
from gtellcorr.telluric.resources import build, freeze, manager, status, toolchain_packages, verify


def _settings(tmp_path: Path) -> Settings:
    return Settings(resource_root=tmp_path / "res", cache_dir=tmp_path / "cache")


def test_default_bundle_location(tmp_path: Path) -> None:
    assert _settings(tmp_path).lblrtm_bundle == tmp_path / "res" / "lblrtm" / "gnu_double"


def test_verify_reports_missing_bundle_and_manifest(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    with pytest.raises(ResourceError, match=r"LBLRTM bundle not found.*resources build"):
        verify(s)
    assert s.lblrtm_bundle is not None
    s.lblrtm_bundle.mkdir(parents=True)
    with pytest.raises(ResourceError, match=r"has no gtellcorr_manifest\.json.*resources freeze"):
        verify(s)
    (s.lblrtm_bundle / "gtellcorr_manifest.json").write_text("{}")
    with pytest.raises(ResourceError, match="failed validation"):
        verify(s)


def test_freeze_errors(tmp_path: Path) -> None:
    with pytest.raises(ResourceError, match="cannot freeze"):
        freeze(tmp_path)
    (tmp_path / "gtellcorr_manifest.json").write_text("{}")
    with pytest.raises(ResourceError, match="exists; pass force=True"):
        freeze(tmp_path)


def test_status_never_raises(tmp_path: Path) -> None:
    report = status(_settings(tmp_path))
    assert report["lblrtm_bundle"]["ok"] is False
    assert "not found" in report["lblrtm_bundle"]["problem"]
    assert report["mipas"]["ok"] is False and report["celestrak"]["ok"] is False
    assert "no CelesTrak table recorded" in report["celestrak"]["problem"]


def test_status_checks_celestrak_as_a_run_does(tmp_path: Path) -> None:
    from gtellcorr.data.atmosphere.reference import record_celestrak

    s = _settings(tmp_path)
    folder = s.resource_root / "atmosphere"
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "celestrak_space_weather_sw_all_20260923.csv"
    snapshot.write_text("DATE,BSRN\n")
    assert status(s)["celestrak"]["ok"] is False  # a file alone is not used by a run
    record_celestrak(snapshot)
    assert status(s)["celestrak"] == {"path": str(snapshot), "ok": True}


def test_toolchain_packages() -> None:
    assert "clang" in toolchain_packages("Darwin")
    linux = toolchain_packages("Linux")
    assert "clang" not in linux and {"netcdf-fortran", "gfortran", "binutils"} <= set(linux)


def test_build_refuses_non_empty_output(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(ResourceError, match="not empty"):
        build(
            _settings(tmp_path),
            lnfl_archive="a",
            lblrtm_archive="b",
            line_file_archive="c",
            output=out,
        )


def test_build_needs_conda_or_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda name: None)
    with pytest.raises(ResourceError, match="no conda/mamba found"):
        build(_settings(tmp_path), lnfl_archive="a", lblrtm_archive="b", line_file_archive="c")


def test_fingerprint_covers_only_what_determines_a_calculation() -> None:
    import copy

    from gtellcorr.telluric.lblrtm import load_lblrtm_resource_manifest
    from gtellcorr.telluric.lblrtm.lblrtm_resources import compute_lblrtm_resource_fingerprint

    data = copy.deepcopy(dict(load_lblrtm_resource_manifest().data))
    base = compute_lblrtm_resource_fingerprint(data)
    described = copy.deepcopy(data)
    described["compatibility"] = {**described["compatibility"], "note": "any text"}
    described["citations"] = [*described["citations"], "another citation"]
    described["created_utc"] = "2030-01-01T00:00:00Z"
    described["status"] = "accepted"
    described.pop("golden_compatibility", None)
    described.pop("resolved_discrepancies", None)
    assert compute_lblrtm_resource_fingerprint(described) == base
    rebuilt = copy.deepcopy(data)
    name = next(iter(rebuilt["sources"]))
    rebuilt["sources"][name] = {**rebuilt["sources"][name], "sha256": "0" * 64}
    assert compute_lblrtm_resource_fingerprint(rebuilt) != base
