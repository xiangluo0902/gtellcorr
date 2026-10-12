"""gtellcorr — forward-model telluric correction for high-resolution spectrographs.

Usage::

    import gtellcorr as gtc

    star = gtc.Star(name="TOI-2076", teff=5180, logg=4.61, feh=0.0, vsini_kms=5.0)
    obs = gtc.Observation.from_directory("spectra/", instrument="WIYN/NEID", target=star)
    result = gtc.run(obs, gtc.FitConfig(), output_dir="out/")

The public API has two levels:

* the workflow API, for running a correction: ``gtc.Observation``, ``gtc.Star``,
  ``gtc.FitConfig``, ``gtc.RunConfig``, ``gtc.Settings``, ``gtc.run``,
  ``gtc.Pipeline``, ``gtc.Result``, ``gtc.load_result``, ``gtc.sites`` and
  ``gtc.instruments``;
* the component API, for using one step on its own: the subpackages
  ``gtc.data``, ``gtc.telluric``, ``gtc.preprocessing``, ``gtc.fitting`` and
  ``gtc.plotting``, each listing its public names in ``__all__``.

Subpackages are imported lazily on first attribute access, so ``import gtellcorr``
stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

from gtellcorr._version import __version__

#: Workflow-API names -> defining module (imported on first access).
_WORKFLOW = {
    "Observation": "gtellcorr.data.observation",
    "Exposure": "gtellcorr.data.observation",
    "Site": "gtellcorr.data.observation",
    "sites": "gtellcorr.data.observation",
    "instruments": "gtellcorr.data.observation",
    "Star": "gtellcorr.data.observation.target",
    "Transit": "gtellcorr.data.observation.target",
    "FitConfig": "gtellcorr.config.fit_config",
    "RunConfig": "gtellcorr.config.run_config",
    "Settings": "gtellcorr.settings",
    "run": "gtellcorr.pipeline",
    "Pipeline": "gtellcorr.pipeline",
    "Result": "gtellcorr.data.products.result",
    "load_result": "gtellcorr.data.products.result",
}

__all__ = [
    "Exposure",
    "FitConfig",
    "Observation",
    "Pipeline",
    "Result",
    "RunConfig",
    "Settings",
    "Site",
    "Star",
    "Transit",
    "__version__",
    "config",
    "data",
    "errors",
    "fitting",
    "instruments",
    "load_result",
    "plotting",
    "preprocessing",
    "registry",
    "run",
    "settings",
    "sites",
    "telluric",
]

_LAZY_MODULES = frozenset(__all__) - {"__version__"} - set(_WORKFLOW)


def __getattr__(name: str) -> Any:
    """Import a workflow-API name or a subpackage the first time it is used."""
    if name in _WORKFLOW:
        value = getattr(importlib.import_module(_WORKFLOW[name]), name)
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The public names (``__all__``), whether or not they are imported yet."""
    return sorted(__all__)
