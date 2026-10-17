from __future__ import annotations

import pytest

from gtellcorr.errors import ConfigError
from gtellcorr.telluric.config import TelluricConfig
from gtellcorr.telluric.molecules import ENGINE_MOLECULES, MOLECULE_NAMES


def test_molecule_lists_match_reference() -> None:
    assert len(ENGINE_MOLECULES) == 12 and len(MOLECULE_NAMES) == 39
    assert MOLECULE_NAMES[:12] == ENGINE_MOLECULES
    assert len(set(MOLECULE_NAMES)) == 39


def test_wave_margin() -> None:
    assert TelluricConfig().wave_margin_nm == 20.0
    with pytest.raises(ConfigError, match=r"^wave_margin_nm: must be >= 0"):
        TelluricConfig(wave_margin_nm=-1)


def test_continuum_mode() -> None:
    assert TelluricConfig().continuum_mode == "full"
    assert TelluricConfig(continuum_mode="none").continuum_mode == "none"
    assert TelluricConfig(continuum_mode="scaled").continuum_mode == "scaled"
    with pytest.raises(ConfigError, match=r"^continuum_mode: must be one of"):
        TelluricConfig(continuum_mode="partial")  # type: ignore[arg-type]
