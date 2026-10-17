"""Declared dependencies agree with the versions the code requires at run time."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from gtellcorr.data.atmosphere.full_profile import _PYMSIS_VERSION_CONTRACT

REPO = Path(__file__).resolve().parents[2]


def test_pymsis_is_pinned_to_the_version_the_profiles_require() -> None:
    """NRLMSISE-00 profiles accept exactly one pymsis version; pip and conda must install it
    (a newer pymsis made every default run fail after a plain pip install)."""
    dependencies = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    pymsis = [d for d in dependencies if re.match(r"pymsis\b", d)]
    assert pymsis == [f"pymsis=={_PYMSIS_VERSION_CONTRACT}"]
    environment = (REPO / "environment.yml").read_text()
    assert re.findall(r"pymsis==([\w.]+)", environment) == [_PYMSIS_VERSION_CONTRACT]
