"""The LBLRTM setup a bundle must have, checked in two steps.

A bundle is described by two documents:

* the **local manifest** (``<bundle>/gtellcorr_manifest.json``), written by
  :func:`gtellcorr.telluric.resources.freeze` on the machine that built the
  bundle: sizes and checksums of every file, including host-specific executables;
* the **LBLRTM setup description** shipped with gtellcorr (``setups/*.json``):
  only what decides the physics — source archives, the compiled line database
  (TAPE3 checksums without the LNFL date and title), the MT_CKD continuum data,
  the LNFL control input, line counts and wavenumber limits, and the arithmetic
  precision / TAPE12 layout.

A bundle is accepted when its files match its local manifest **and** the physics
part of the manifest equals the shipped setup description. Rebuilding on another
machine or with another compiler therefore needs no change to gtellcorr.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from gtellcorr.errors import ResourceError

__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_SETUP",
    "check_contract",
    "check_setup",
    "load_contract",
    "load_setup",
    "physics_fields",
    "scientific_identity",
]

DEFAULT_SETUP = "lblrtm_v12_17_aer_v3_9_double.json"


def physics_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a (schema v2) resource manifest that decides the physics."""
    sources = manifest["sources"]
    products = manifest["products"]
    build = manifest["build_contract"]
    return {
        "sources": {
            name: {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
            for name, record in sorted(sources.items())
        },
        "tape3": {
            "scientific_sha256": products["tape3"]["scientific_sha256"],
            "semantic_sha256": products["tape3"]["semantic_sha256"],
        },
        "mt_ckd_h2o_continuum_coefficients": {
            "sha256": products["mt_ckd_h2o_continuum_coefficients"]["sha256"],
            "size_bytes": products["mt_ckd_h2o_continuum_coefficients"]["size_bytes"],
        },
        "lnfl_tape5_sha256": products["lnfl_tape5"]["sha256"],
        "scientific_contract": manifest["scientific_contract"],
        "precision": {
            "lnfl": build["lnfl_precision"],
            "lblrtm": build["lblrtm_precision"],
            "tape12_abi": build["tape12_abi"],
        },
    }


def load_setup(name: str = DEFAULT_SETUP) -> dict[str, Any]:
    """An LBLRTM setup description shipped with gtellcorr (``setups/<name>``)."""
    try:
        text = files("gtellcorr.telluric.resources").joinpath("setups", name).read_text("utf-8")
    except FileNotFoundError as exc:
        raise ResourceError(f"unknown LBLRTM setup description {name!r}") from exc
    data: dict[str, Any] = json.loads(text)
    return data


def _differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Dotted paths of the leaves where two nested mappings differ."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in sorted(set(a) | set(b)):
            out += _differences(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
        return out
    return [] if a == b else [path or "<root>"]


def check_setup(manifest: Mapping[str, Any], setup: Mapping[str, Any]) -> None:
    """Raise :class:`ResourceError` if *manifest* does not have the LBLRTM *setup*."""
    diffs = _differences(physics_fields(manifest), setup["physics"])
    if diffs:
        raise ResourceError(
            "the LBLRTM bundle does not match the supported LBLRTM setup; it differs in: "
            f"{', '.join(diffs[:8])}{' …' if len(diffs) > 8 else ''}"
        )


# names used by gtellcorr 0.9.0, removed in 1.0
DEFAULT_CONTRACT = DEFAULT_SETUP
scientific_identity = physics_fields
load_contract = load_setup
check_contract = check_setup
