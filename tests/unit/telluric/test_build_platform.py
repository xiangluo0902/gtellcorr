"""Linux code paths of the ported builder / freezer / validator (D8), exercised on any host."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from gtellcorr.telluric.lblrtm import lblrtm_resources
from gtellcorr.telluric.lblrtm.lblrtm_errors import LblrtmManifestError
from gtellcorr.telluric.resources import _build, _freeze

READELF = """
Dynamic section at offset 0x2d8 contains 30 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libnetcdff.so.7]
 0x0000000000000001 (NEEDED)             Shared library: [libgfortran.so.5]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000001d (RUNPATH)            Library runpath: [/opt/tc/lib:$ORIGIN/../lib]
"""


@pytest.fixture
def linux_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, ModuleType]]:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    build, freeze = importlib.reload(_build), importlib.reload(_freeze)
    yield build, freeze
    monkeypatch.undo()
    importlib.reload(_build)
    importlib.reload(_freeze)


def test_macos_values_equal_the_reference() -> None:
    if platform.system() != "Darwin":
        pytest.skip("macOS host only")
    assert _build.LNFL_MAKE_TARGET == "osxGNUsgl"
    assert _build.LNFL_EXECUTABLE_NAME == "lnfl_v3.2_OS_X_gnu_sgl"
    assert _build._lblrtm_build_spec("double").executable_name == "lblrtm_v12.17_OS_X_gnu_dbl"


def test_linux_targets_and_product_names(linux_modules: tuple[ModuleType, ModuleType]) -> None:
    build, freeze = linux_modules
    assert build.LNFL_MAKE_TARGET == "linuxGNUsgl"
    assert build.LNFL_EXECUTABLE_NAME == "lnfl_v3.2_linux_gnu_sgl"
    spec = build._lblrtm_build_spec("double")
    assert (spec.make_target, spec.executable_name) == (
        "linuxGNUdbl",
        "lblrtm_v12.17_linux_gnu_dbl",
    )
    assert freeze._EXPECTED_LNFL_BUILD_CONTRACT["make_target"] == "linuxGNUsgl"
    assert freeze._EXPECTED_LBLRTM_BUILD_CONTRACTS["double"]["product_basename"].endswith(
        "_linux_gnu_dbl"
    )
    records = build._product_records(lblrtm_build_spec=spec)
    assert records["lnfl_executable"]["relative_path"] == "bin/lnfl_v3.2_linux_gnu_sgl"
    assert records["lblrtm_executable"]["relative_path"] == "bin/lblrtm_v12.17_linux_gnu_dbl"


def test_elf_linkage_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_build, "_capture_checked_text", lambda *a, **k: READELF)
    record = _build._inspect_elf_linkage(Path("/x/lblrtm"), readelf_command="readelf")
    assert record["linked_libraries"] == ["libnetcdff.so.7", "libgfortran.so.5", "libc.so.6"]
    assert record["lc_rpaths"] == ["/opt/tc/lib", "$ORIGIN/../lib"]
    assert record["absolute_rpaths"] == ["/opt/tc/lib"]
    assert record["relocatable"] is False


def _contract(lnfl: str, lblrtm: str) -> dict[str, object]:
    return {
        "lnfl_precision": "single",
        "lblrtm_precision": "double",
        "lnfl_make_target": lnfl,
        "lblrtm_make_target": lblrtm,
        "lnfl_product_basename": f"lnfl_v3.2_{lnfl[:-6]}_gnu_sgl",
        "lblrtm_product_basename": f"lblrtm_v12.17_{lblrtm[:-6]}_gnu_dbl",
        "tape12_abi": {
            "byte_order": "little",
            "record_marker_bytes": 4,
            "real_bytes": 8,
            "integer_bytes": 8,
        },
    }


def _products(contract: dict[str, object]) -> dict[str, object]:
    return {
        "lnfl_executable": {"relative_path": f"bin/{contract['lnfl_product_basename']}"},
        "lblrtm_executable": {"relative_path": f"bin/{contract['lblrtm_product_basename']}"},
    }


@pytest.mark.parametrize(
    ("lnfl", "lblrtm"), [("osxGNUsgl", "osxGNUdbl"), ("linuxGNUsgl", "linuxGNUdbl")]
)
def test_validator_accepts_both_platforms(lnfl: str, lblrtm: str) -> None:
    contract = _contract(lnfl, lblrtm)
    lblrtm_resources._validate_manifest_tape12_contract(
        schema_version=2, build_contract=contract, products=_products(contract)
    )


def test_validator_rejects_mixed_platforms() -> None:
    contract = _contract("linuxGNUsgl", "osxGNUdbl")
    with pytest.raises(LblrtmManifestError, match="lblrtm_make_target"):
        lblrtm_resources._validate_manifest_tape12_contract(
            schema_version=2, build_contract=contract, products=_products(contract)
        )
