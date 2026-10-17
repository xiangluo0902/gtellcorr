"""Reading and writing run and fit configuration files.

The file format itself (schema version, migrations, writer) lives in
:mod:`gtellcorr.config.toml_format`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.config.run_config import RunConfig
from gtellcorr.config.toml_format import dumps, dumps_group, loads
from gtellcorr.errors import ConfigError

__all__ = [
    "dumps_config",
    "load_fit_config",
    "load_run_config",
    "save_fit_config",
    "save_run_config",
]


def dumps_config(config: RunConfig | FitConfig, *, comments: bool = True) -> str:
    """TOML text of a run or fit configuration, optionally with field comments."""
    if comments:
        header = (
            "gtellcorr configuration. Each parameter is documented above its line;\n"
            "parameters left out keep their default values."
        )
    else:
        header = "gtellcorr configuration. Parameters left out keep their default values."
    if isinstance(config, RunConfig):
        return dumps(config.to_dict(), source=config if comments else None, header=header)
    return dumps_group("fit", config, comments=comments, header=header)


def _read(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    """Raw tables of a configuration file (migrated to the current schema) and its name."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {p}: {exc}") from exc
    return loads(text, source=str(p)), str(p)


def _prefixed(source: str, exc: ConfigError) -> ConfigError:
    """The same error, prefixed with the file it came from."""
    return ConfigError(f"{source}: {exc}")


def load_run_config(path: str | os.PathLike[str]) -> RunConfig:
    """Read a run file (``[data]`` and ``[fit]``)."""
    doc, source = _read(path)
    try:
        return RunConfig.from_dict(doc)
    except ConfigError as exc:
        raise _prefixed(source, exc) from None


def load_fit_config(path: str | os.PathLike[str]) -> FitConfig:
    """Read the ``[fit]`` table of a fit file or a run file."""
    doc, source = _read(path)
    unknown = sorted(set(doc) - {"fit", "data"})
    if unknown:
        raise ConfigError(f"{source}: unknown table {unknown[0]!r}; expected [fit] (and [data])")
    try:
        return FitConfig.from_dict(doc.get("fit", {}))
    except ConfigError as exc:
        raise ConfigError(f"{source}: fit.{exc}") from None


def save_run_config(
    config: RunConfig, path: str | os.PathLike[str], *, comments: bool = True
) -> None:
    """Write a run file; *comments* adds each parameter's documentation."""
    Path(path).write_text(dumps_config(config, comments=comments), encoding="utf-8")


def save_fit_config(
    config: FitConfig, path: str | os.PathLike[str], *, comments: bool = True
) -> None:
    """Write a fit file (only the ``[fit]`` table); *comments* as for run files."""
    Path(path).write_text(dumps_config(config, comments=comments), encoding="utf-8")
