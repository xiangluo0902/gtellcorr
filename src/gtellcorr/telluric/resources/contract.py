"""Scientific identity of an LBLRTM resource bundle (two-level manifests).

A bundle is described by two documents:

* the **local manifest** (``<bundle>/gtellcorr_manifest.json``), written by
  :func:`gtellcorr.telluric.resources.freeze` on the machine that built the
  bundle: exact bytes of every file, including host-specific executables;
* the **scientific contract** shipped with gtellcorr (``contracts/*.json``):
  only what defines the physics — source archives, the compiled line database
  (TAPE3 scientific and semantic hashes), the MT_CKD continuum data, the LNFL
  control input, the scientific contract (line counts, wavenumber bounds) and
  the arithmetic precision / TAPE12 layout.

A bundle is accepted when its files match its local manifest **and** the local
manifest's scientific identity equals the shipped contract. Rebuilding on another
machine or with another compiler therefore needs no change to gtellcorr.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from gtellcorr.errors import ResourceError

__all__ = ["DEFAULT_CONTRACT", "check_contract", "load_contract", "scientific_identity"]

DEFAULT_CONTRACT = "lblrtm_v12_17_aer_v3_9_double.json"


def scientific_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The physics-defining subset of a (schema v2) resource manifest."""
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


def load_contract(name: str = DEFAULT_CONTRACT) -> dict[str, Any]:
    """A contract shipped with gtellcorr (``contracts/<name>``)."""
    try:
        text = files("gtellcorr.telluric.resources").joinpath("contracts", name).read_text("utf-8")
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


def check_contract(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Raise :class:`ResourceError` if *manifest* does not fulfil *contract*."""
    diffs = _differences(scientific_identity(manifest), contract["identity"])
    if diffs:
        raise ResourceError(
            "the LBLRTM bundle does not match the supported LBLRTM setup; it differs in: "
            f"{', '.join(diffs[:8])}{' …' if len(diffs) > 8 else ''}"
        )
