"""Configuration framework: parameter-group base class, ``FitConfig``, ``RunConfig``
and TOML reading/writing.

Names are resolved lazily because the parameter groups live in the subpackages
that use them, and those import :mod:`gtellcorr.config.base`.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ConfigGroup",
    "FitConfig",
    "FrozenDict",
    "RunConfig",
    "dumps_config",
    "load_fit_config",
    "load_run_config",
    "save_fit_config",
    "save_run_config",
]

_WHERE = {
    "ConfigGroup": "base",
    "FrozenDict": "base",
    "FitConfig": "fit_config",
    "RunConfig": "run_config",
    "dumps_config": "toml_io",
    "load_fit_config": "toml_io",
    "load_run_config": "toml_io",
    "save_fit_config": "toml_io",
    "save_run_config": "toml_io",
}


def __getattr__(name: str) -> Any:
    """Import a configuration name from its module the first time it is used."""
    if name in _WHERE:
        value = getattr(importlib.import_module(f"{__name__}.{_WHERE[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The public names (``__all__``), whether or not they are imported yet."""
    return sorted(__all__)
