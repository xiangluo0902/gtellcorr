from __future__ import annotations

import importlib
from importlib.metadata import version
from importlib.resources import files

import pytest

import gtellcorr as gtc

SUBPACKAGES = [
    "gtellcorr.config",
    "gtellcorr.data",
    "gtellcorr.data.observation",
    "gtellcorr.data.observation.readers",
    "gtellcorr.data.atmosphere",
    "gtellcorr.data.products",
    "gtellcorr.telluric",
    "gtellcorr.telluric.lblrtm",
    "gtellcorr.telluric.resources",
    "gtellcorr.preprocessing",
    "gtellcorr.preprocessing.treatments",
    "gtellcorr.fitting",
    "gtellcorr.fitting.blocks",
    "gtellcorr.plotting",
]
#: Subpackages that must import but have no ``__all__``.
PORTED_SUBPACKAGES = ["gtellcorr.data.atmosphere.backends"]


def test_version_matches_installed_metadata() -> None:
    assert gtc.__version__ == version("gtellcorr")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_subpackage_imports_and_declares_all(name: str) -> None:
    module = importlib.import_module(name)
    assert isinstance(module.__all__, list)
    assert module.__doc__


@pytest.mark.parametrize("name", PORTED_SUBPACKAGES)
def test_ported_subpackages_import(name: str) -> None:
    assert importlib.import_module(name).__doc__


def test_top_level_subpackages_are_lazy_attributes() -> None:
    assert gtc.telluric.__name__ == "gtellcorr.telluric"
    assert "telluric" in dir(gtc)


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = gtc.nope


def test_py_typed_marker_is_shipped() -> None:
    assert files("gtellcorr").joinpath("py.typed").is_file()
