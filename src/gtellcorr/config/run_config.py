"""``RunConfig``: a complete run description — which data, and how to fit it."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from gtellcorr.config.base import ConfigGroup
from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation.config import DataConfig

__all__ = ["RunConfig"]


@dataclass(frozen=True, kw_only=True)
class RunConfig(ConfigGroup):
    """The ``[data]`` and ``[fit]`` tables of a run file (``gtellcorr run <file>``)."""

    data: DataConfig
    """Which observations to process: directory, instrument, target, exposure filters."""
    fit: FitConfig = field(default_factory=FitConfig)
    """How to fit them: every scientific parameter of the telluric correction."""

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the run file (TOML, every parameter documented)."""
        from gtellcorr.config.toml_io import save_run_config

        save_run_config(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RunConfig:
        """Read a run file; files of earlier schema versions are migrated."""
        from gtellcorr.config.toml_io import load_run_config

        return load_run_config(path)
