"""LBLRTM resources: two-level validation, freezing, building (macOS + Linux).

Public API::

    from gtellcorr.telluric import resources
    resources.status()            # what is installed and usable
    resources.verify()            # -> VerifiedResources (raises ResourceError)
    resources.freeze(bundle)      # write the bundle's local manifest
    resources.build(lnfl_archive=..., lblrtm_archive=..., line_file_archive=...)

``_build`` compiles LNFL / LBLRTM and assembles a bundle from the AER source
archives; ``_freeze`` measures an assembled bundle into its local manifest.
"""

from __future__ import annotations

from gtellcorr.telluric.resources.manager import (
    LOCAL_MANIFEST,
    VerifiedResources,
    build,
    describe,
    freeze,
    status,
    toolchain_packages,
    verify,
)
from gtellcorr.telluric.resources.setup import (
    check_contract,
    check_setup,
    load_contract,
    load_setup,
    physics_fields,
    scientific_identity,
)

__all__ = [
    "LOCAL_MANIFEST",
    "VerifiedResources",
    "build",
    "check_contract",
    "check_setup",
    "describe",
    "freeze",
    "load_contract",
    "load_setup",
    "physics_fields",
    "scientific_identity",
    "status",
    "toolchain_packages",
    "verify",
]
