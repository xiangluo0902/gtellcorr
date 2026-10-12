from __future__ import annotations

from typing import Any

import pytest

from gtellcorr.telluric import LblrtmBackend, rt_backends
from gtellcorr.telluric.lblrtm import optical_depth_calculator, transmission_calculator


def test_registry() -> None:
    assert rt_backends.get("LBLRTM") is LblrtmBackend


def test_lblrtm_backend_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake(profile: Any, window: Any, **kwargs: Any) -> str:
        calls.update(kwargs, profile=profile, window=window)
        return "result"

    monkeypatch.setattr(transmission_calculator, "calculate_lblrtm_transmission", fake)
    monkeypatch.setattr(optical_depth_calculator, "calculate_lblrtm_optical_depth", fake)
    backend = LblrtmBackend("RES", timeout_s=5.0)
    out = backend.transmission(
        {}, (1, 2), wavelength_medium="air", formula="espresso", line_of_sight="LOS"
    )
    assert out == "result"
    assert calls["resources"] == "RES" and calls["quality_mode"] == "evaluate"
    assert calls["wavelength_conversion_formula"] == "espresso" and calls["timeout_s"] == 5.0
    calls.clear()
    backend.optical_depth(
        {}, (1, 2), wavelength_medium="vacuum", formula="ciddor1996", line_of_sight="L"
    )
    assert calls["wavelength_medium"] == "vacuum" and "quality_mode" not in calls
