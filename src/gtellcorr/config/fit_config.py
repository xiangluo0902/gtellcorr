"""``FitConfig``: every scientific parameter of a telluric fit, in one object.

Each parameter group is defined next to the code that uses it (for example
:class:`~gtellcorr.fitting.config.LSF` in :mod:`gtellcorr.fitting`); this module
only assembles them into the ``[fit]`` table of a run file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from gtellcorr.config.base import ConfigGroup
from gtellcorr.data.atmosphere.config import AtmosphereConfig
from gtellcorr.data.observation.config import Frames
from gtellcorr.data.products.config import Postprocessing
from gtellcorr.fitting.config import (
    LSF,
    Continuum,
    Molecules,
    MoleculeSelection,
    OrderSelection,
    Wavelength,
)
from gtellcorr.preprocessing.config import (
    FitMasks,
    MasterSpectrum,
    RVConfig,
    StellarModelConfig,
)
from gtellcorr.telluric.config import TelluricConfig

__all__ = ["FitConfig"]


@dataclass(frozen=True, kw_only=True)
class FitConfig(ConfigGroup):
    """All scientific parameters of a telluric fit (the ``[fit]`` table of a run file).

    Machine-local facts (resource paths, caches, credentials) are *not* here;
    they belong to :class:`gtellcorr.settings.Settings`.
    """

    stellar: Literal["mask", "template"] = "mask"
    """How stellar lines are handled: ``mask`` excludes them from the telluric fit;
    ``template`` models them with a stellar template (synthesised, or
    ``stellar_model.template_file``) placed at each exposure's RV."""
    relcol_mode: Literal["single_order", "joint_orders"] = "joint_orders"
    """``joint_orders``: one set of relcols per exposure, shared by the fit orders and
    applied to all orders; ``single_order``: each order fits its own relcols (useful for
    looking at one order on its own)."""
    on_missing_stellar_data: Literal["generate", "error"] = "generate"
    """What to do when a stellar product (template, RV table, stellar-line masks) is
    not on disk yet: ``generate`` builds it, ``error`` stops with a message that says
    how to build it."""

    # defined in gtellcorr.fitting.config
    continuum: Continuum = field(default_factory=Continuum)
    """Continuum polynomial fitted together with the telluric model."""
    lsf: LSF = field(default_factory=LSF)
    """Instrumental line-spread function: shape and resolving power."""
    wavelength: Wavelength = field(default_factory=Wavelength)
    """Optional correction of the wavelength solution, fitted on the telluric lines."""
    molecules: Molecules = field(default_factory=Molecules.default)
    """Per molecule: whether its column is fitted, its start value and bounds."""
    molecule_selection: MoleculeSelection = field(default_factory=MoleculeSelection)
    """Per-run additions to or removals from the fitted molecules."""
    orders: OrderSelection = field(default_factory=OrderSelection)
    """Which orders are fitted and which are written out."""
    # defined in gtellcorr.preprocessing.config
    masks: FitMasks = field(default_factory=FitMasks)
    """Pixels excluded from the fit: stellar lines, activity lines, bad regions,
    saturated telluric cores."""
    master_spectrum: MasterSpectrum = field(default_factory=MasterSpectrum)
    """Co-added stellar spectrum used to find stellar lines (``mask`` treatment)."""
    stellar_model: StellarModelConfig = field(default_factory=StellarModelConfig)
    """Stellar template: synthesis backend and settings, or a user-supplied file."""
    rv: RVConfig = field(default_factory=RVConfig)
    """Per-exposure stellar RV: measured by cross-correlation or read from a file."""
    # defined in gtellcorr.telluric.config
    telluric: TelluricConfig = field(default_factory=TelluricConfig)
    """Telluric forward model (LBLRTM) settings."""
    # defined in gtellcorr.data
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    """Sources of the per-exposure atmospheric profile."""
    frames: Frames = field(default_factory=Frames)
    """Wavelength-frame conventions (vacuum/air conversion)."""
    postprocessing: Postprocessing = field(default_factory=Postprocessing)
    """Products computed after all exposures are corrected."""

    def validate(self) -> None:
        """Molecules added through ``molecule_selection`` must exist."""
        for molecule in self.molecule_selection.add:
            if molecule not in self.molecules:
                self.fail("molecule_selection.add", f"unknown molecule {molecule!r}")

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a TOML file holding ``schema_version`` and the ``[fit]`` table."""
        from gtellcorr.config.toml_io import save_fit_config

        save_fit_config(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> FitConfig:
        """Read the ``[fit]`` table of a TOML file (a fit file or a full run file)."""
        from gtellcorr.config.toml_io import load_fit_config

        return load_fit_config(path)
