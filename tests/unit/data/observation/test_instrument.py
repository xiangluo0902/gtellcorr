from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from astropy.io import fits

from gtellcorr.data.observation.instrument import (
    identify,
    instruments,
    load_instrument_file,
    read_resolution,
)
from gtellcorr.errors import ConfigError, DataError

REPO = Path(__file__).parents[4]
BUILTIN_DIR = REPO / "src" / "gtellcorr" / "data" / "observation" / "instruments"


def test_builtin_registry() -> None:
    assert instruments.names() == ("LBT/PEPSI", "VLT/ESPRESSO", "WIYN/NEID")
    neid = instruments.get("neid")
    assert neid.name == "WIYN/NEID" and neid.site_token == "KPNO"
    assert len(neid.orders) == 122 and neid.spectral.reader == "echelle_hdu"
    assert neid.anchor_orders["o2"] == (83, 84, 85, 92, 93)
    pepsi = instruments.get("PEPSI")
    assert pepsi.default_channel == "red" and pepsi.spectral.frame.removed_velocity_mps == (
        "SSTVEL",
    )
    esp = instruments.get("VLT/ESPRESSO")
    assert esp.spectral.frame.berv_kms == "ESO QC BERV"


def test_identify() -> None:
    assert identify(fits.Header({"INSTRUME": " neid "})).name == "WIYN/NEID"
    with pytest.raises(DataError, match="no registered instrument matches"):
        identify(fits.Header({"INSTRUME": "HARPS"}))


def test_identify_ambiguous() -> None:
    clone = instruments.get("NEID").replace(name="TEST/NEID2", aliases=())
    instruments.register(clone.name, clone)
    try:
        with pytest.raises(DataError, match="several instruments match"):
            identify(fits.Header({"INSTRUME": "NEID"}))
    finally:
        instruments.unregister(clone.name)


@pytest.mark.parametrize(
    ("instrument", "header", "expected"),
    [
        ("NEID", {}, 110000.0),
        ("PEPSI", {"FIBER": "200"}, 130000.0),
        ("PEPSI", {"FIBER": 300}, 50000.0),
        ("PEPSI", {"FIBER": 100.0}, 250000.0),
        ("ESPRESSO", {"HIERARCH ESO INS MODE": "SINGLEUHR"}, 190000.0),
    ],
)
def test_read_resolution(instrument: str, header: dict[str, Any], expected: float) -> None:
    assert read_resolution(fits.Header(header), instruments.get(instrument)) == expected


def test_read_resolution_is_strict() -> None:
    pepsi = instruments.get("PEPSI")
    with pytest.raises(DataError, match="none of the resolution cards"):
        read_resolution(fits.Header({}), pepsi)
    with pytest.raises(DataError, match="not in the resolution map"):
        read_resolution(fits.Header({"FIBER": "150"}), pepsi)
    direct = pepsi.replace(resolution={"default": 1.0, "keys": ["R"]})
    assert read_resolution(fits.Header({"R": "75000"}), direct) == 75000.0


def test_channel_glob() -> None:
    pepsi = instruments.get("PEPSI")
    assert pepsi.channel_glob(None) == ("red", "pepsir.*.dxt.nor")
    assert pepsi.channel_glob("blue")[0] == "blue"
    with pytest.raises(DataError, match="unknown channel 'green'"):
        pepsi.channel_glob("green")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "NEID"}, r"^name: must have the form"),
        ({"channels": {}}, r"^channels: at least one"),
        ({"default_channel": "x"}, r"^default_channel: 'x' is not"),
        ({"channels": {"a": "a*", "b": "b*"}}, r"^default_channel: required"),
        ({"orders": []}, r"^orders: at least one"),
        ({"anchor_orders": {"h2o": [999]}}, r"^anchor_orders\.h2o: orders \[999\]"),
        ({"resolution": {"default": 0}}, r"^resolution\.default: must be > 0"),
        ({"resolution": {"default": 1, "map": {"a": -1}}}, r"^resolution\.map\.a"),
    ],
)
def test_validation(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        instruments.get("NEID").replace(**changes)


def test_spectral_validation() -> None:
    spectral = instruments.get("NEID").spectral
    with pytest.raises(ConfigError, match="wavelength_range_nm"):
        spectral.replace(wavelength_range_nm=(5.0, 1.0))
    with pytest.raises(ConfigError, match=r"order_windows_nm\[0\]"):
        spectral.replace(order_windows_nm=[[2.0, 1.0]])


def test_load_instrument_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read instrument file"):
        load_instrument_file(tmp_path / "missing.toml")
    extra = tmp_path / "x.toml"
    extra.write_text("schema_version = 1\n[instrument]\n[other]\n")
    with pytest.raises(ConfigError, match=r"exactly one \[instrument\] table"):
        load_instrument_file(extra)
    bad = tmp_path / "bad.toml"
    text = (
        (BUILTIN_DIR / "wiyn_neid.toml").read_text().replace('name = "WIYN/NEID"', 'name = "NEID"')
    )
    bad.write_text(text)
    with pytest.raises(ConfigError, match=r"bad\.toml: instrument\.name: must have the form"):
        load_instrument_file(bad)
