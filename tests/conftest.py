"""Shared pytest configuration.

Markers (declared in pyproject.toml) and when their tests run:

* ``lblrtm``     — only if the configured LBLRTM bundle (``Settings().lblrtm_bundle``)
  has a local manifest;
* ``network``    — only with ``--run-network``;
* ``regression`` — only if ``GTELLCORR_REGRESSION_DATA`` points to the baseline data.

Tests must never read the developer's real settings file or environment by
accident: the autouse fixture below isolates both.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REGRESSION_DATA_ENV = "GTELLCORR_REGRESSION_DATA"

# Figures are written, never shown: the package leaves the backend to the user.
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network", action="store_true", default=False, help="run tests marked 'network'"
    )


def _lblrtm_available() -> bool:
    from gtellcorr.settings import Settings

    try:
        bundle = Settings.load().lblrtm_bundle
    except Exception:
        return False
    return bundle is not None and (bundle / "gtellcorr_manifest.json").is_file()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_network = pytest.mark.skip(reason="needs --run-network")
    skip_lblrtm = pytest.mark.skip(reason="no LBLRTM resource bundle found")
    skip_regression = pytest.mark.skip(reason=f"{REGRESSION_DATA_ENV} is not set")
    have_lblrtm: bool | None = None
    for item in items:
        # get_closest_marker, not item.keywords: keywords also contain test names
        # and parameter ids (a parameter called "lblrtm" must not skip a test).
        if item.get_closest_marker("network") and not config.getoption("--run-network"):
            item.add_marker(skip_network)
        if item.get_closest_marker("regression") and not os.environ.get(REGRESSION_DATA_ENV):
            item.add_marker(skip_regression)
        if item.get_closest_marker("lblrtm"):
            if have_lblrtm is None:
                have_lblrtm = _lblrtm_available()
            if not have_lblrtm:
                item.add_marker(skip_lblrtm)


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """An empty environment plus a settings-file path that does not exist.

    Use with ``Settings.load(environ=isolated_env)`` so tests never see the
    developer's own ``~/.config/gtellcorr/settings.toml`` or ``GTELLCORR_*`` vars.
    """
    for var in [v for v in os.environ if v.startswith("GTELLCORR_")]:
        monkeypatch.delenv(var)
    return {"GTELLCORR_SETTINGS": str(tmp_path / "no-such-settings.toml")}
