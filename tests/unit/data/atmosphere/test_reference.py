from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gtellcorr.data.atmosphere import reference
from gtellcorr.errors import ResourceError


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes | Exception) -> list[str]:
    urls: list[str] = []

    def fake(url: str, timeout: float) -> Any:
        urls.append(url)
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(payload)

    monkeypatch.setattr(reference.urllib.request, "urlopen", fake)
    return urls


def test_fetch_mipas_verifies_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"! MIPAS profile\n"
    digest = hashlib.sha256(data).hexdigest()
    urls = _serve(monkeypatch, data)
    got = reference.fetch_mipas(tmp_path, file_name="m.atm", sha256=digest, url="http://x")
    assert got.path.read_bytes() == data and urls == ["http://x"]
    reference.fetch_mipas(tmp_path, file_name="m.atm", sha256=digest, url="http://x")
    assert urls == ["http://x"]  # verified copy reused, no second download
    with pytest.raises(ResourceError, match="not the expected one"):
        reference.fetch_mipas(tmp_path / "b", file_name="m.atm", sha256="0" * 64, url="http://x")


def test_fetch_and_read_celestrak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, b"DATE,BSRN\n2022-02-13,1\n")
    got = reference.fetch_celestrak(tmp_path, today=datetime(2026, 9, 23, tzinfo=UTC))
    assert got.path.name == "celestrak_space_weather_sw_all_20260923.csv"
    assert reference.latest_celestrak(tmp_path) == got
    got.path.write_text("tampered")
    with pytest.raises(ResourceError, match="does not match"):
        reference.latest_celestrak(tmp_path)


def test_celestrak_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ResourceError, match="no CelesTrak snapshot recorded"):
        reference.latest_celestrak(tmp_path)
    _serve(monkeypatch, b"<html>")
    with pytest.raises(ResourceError, match="unexpected content"):
        reference.fetch_celestrak(tmp_path)
    _serve(monkeypatch, OSError("offline"))
    with pytest.raises(ResourceError, match="cannot download"):
        reference.fetch_celestrak(tmp_path)


_SW_ALL = (
    "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,"
    "AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,"
    "F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n"
    "2022-02-10,2571,10,27,27,20,20,27,47,50,43,260,12,12,7,7,12,39,48,32,21,1.1,5,72,"
    "118.0,114.9,OBS,107.7,104.5,105.1,101.3\n"
    "2022-02-11,2571,11,40,30,20,40,13,27,43,40,253,27,15,7,27,5,12,32,27,19,1.0,5,75,"
    "113.1,110.2,OBS,107.8,104.9,105.3,101.7\n"
    "2022-02-12,2571,12,33,30,23,17,27,27,33,30,220,18,15,9,6,12,12,18,15,13,0.8,4,54,"
    "110.5,107.7,OBS,108.1,105.2,105.6,102.0\n"
    "2022-02-13,2571,13,37,30,23,27,30,33,23,30,233,22,15,9,12,15,18,9,15,14,0.8,4,46,"
    "105.4,102.8,{kind},108.4,105.5,105.9,102.3\n"
)


def _space_weather(kind: str = "OBS") -> Any:
    from gtellcorr.data.atmosphere import space_weather

    data = _SW_ALL.format(kind=kind).encode()
    return space_weather.load_celestrak_space_weather(
        data, expected_sha256=hashlib.sha256(data).hexdigest(), source_url=reference.CELESTRAK_URL
    )


def test_space_weather_outside_the_table_says_what_to_do() -> None:
    from gtellcorr.data.atmosphere.space_weather import get_msis_space_weather_inputs

    table = _space_weather()
    get_msis_space_weather_inputs(table, datetime(2022, 2, 13, 7, tzinfo=UTC))
    with pytest.raises(ResourceError, match=r"no data for 2022-02-14.*fetch celestrak"):
        get_msis_space_weather_inputs(table, datetime(2022, 2, 14, 7, tzinfo=UTC))
    with pytest.raises(ResourceError, match=r"no data for 2022-02-09 .*57 hours"):
        get_msis_space_weather_inputs(table, datetime(2022, 2, 11, 7, tzinfo=UTC))


def test_space_weather_warns_about_predicted_flux() -> None:
    from gtellcorr.data.atmosphere.space_weather import get_msis_space_weather_inputs

    with pytest.warns(UserWarning, match=r"81-day mean F10.7a: predicted"):
        get_msis_space_weather_inputs(_space_weather("PRD"), datetime(2022, 2, 13, tzinfo=UTC))
