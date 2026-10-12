from __future__ import annotations

from pathlib import Path

import pytest

from gtellcorr.errors import ConfigError
from gtellcorr.settings import Settings


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_use_platform_dirs(isolated_env: dict[str, str]) -> None:
    s = Settings.load(environ=isolated_env)
    assert s.resource_root.name == "resources"
    assert "gtellcorr" in str(s.resource_root)
    assert "gtellcorr" in str(s.cache_dir)
    assert s.credentials_file is None
    assert s.lblrtm_timeout_s == 600.0


def test_priority_argument_over_env_over_file(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    settings_file = write(
        tmp_path / "settings.toml",
        f'resource_root = "{tmp_path / "from_file"}"\n'
        f'cache_dir = "{tmp_path / "cache_from_file"}"\n'
        "lblrtm_timeout_s = 900\n",
    )
    env = {**isolated_env, "GTELLCORR_SETTINGS": str(settings_file)}
    env["GTELLCORR_RESOURCES"] = str(tmp_path / "from_env")

    s = Settings.load(environ=env)
    assert s.resource_root == tmp_path / "from_env"  # env beats file
    assert s.cache_dir == tmp_path / "cache_from_file"  # file beats default
    assert s.lblrtm_timeout_s == 900.0

    s = Settings.load(environ=env, resource_root=tmp_path / "from_arg", cache_dir=None)
    assert s.resource_root == tmp_path / "from_arg"  # argument beats env
    assert s.cache_dir == tmp_path / "cache_from_file"  # None means "not given"


def test_explicit_settings_file_beats_env_variable(
    tmp_path: Path, isolated_env: dict[str, str]
) -> None:
    a = write(tmp_path / "a.toml", "lblrtm_timeout_s = 1\n")
    b = write(tmp_path / "b.toml", "lblrtm_timeout_s = 2\n")
    s = Settings.load(settings_file=b, environ={**isolated_env, "GTELLCORR_SETTINGS": str(a)})
    assert s.lblrtm_timeout_s == 2.0


def test_paths_are_expanded_and_absolute(isolated_env: dict[str, str]) -> None:
    s = Settings.load(environ=isolated_env, resource_root="~/res", credentials_file="creds.env")
    assert s.resource_root == Path.home() / "res"
    assert s.credentials_file is not None
    assert s.credentials_file.is_absolute()


def test_unknown_key_in_file_is_rejected_with_hint(
    tmp_path: Path, isolated_env: dict[str, str]
) -> None:
    bad = write(tmp_path / "s.toml", 'resource_rot = "/x"\n')
    with pytest.raises(ConfigError, match=r"unknown setting 'resource_rot'.*'resource_root'"):
        Settings.load(settings_file=bad, environ=isolated_env)


def test_unknown_override_is_rejected(isolated_env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="unknown setting 'cache'"):
        Settings.load(environ=isolated_env, cache="/x")


def test_missing_explicit_file_is_an_error(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Settings.load(settings_file=tmp_path / "missing.toml", environ=isolated_env)


def test_invalid_toml_is_reported(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    bad = write(tmp_path / "s.toml", "resource_root = \n")
    with pytest.raises(ConfigError, match="cannot parse"):
        Settings.load(settings_file=bad, environ=isolated_env)


@pytest.mark.parametrize("timeout", [0, -5])
def test_timeout_must_be_positive(isolated_env: dict[str, str], timeout: float) -> None:
    with pytest.raises(ConfigError, match="positive"):
        Settings.load(environ=isolated_env, lblrtm_timeout_s=timeout)


def test_settings_are_immutable(isolated_env: dict[str, str]) -> None:
    s = Settings.load(environ=isolated_env)
    with pytest.raises(AttributeError):
        s.cache_dir = Path("/tmp")  # type: ignore[misc]
