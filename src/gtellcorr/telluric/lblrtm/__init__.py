"""gtellcorr-owned LBLRTM resource helpers for the active project backend.

Importing this package remains intentionally lightweight: the facade exposes
resource/provenance helpers but does not validate a bundle, start LBLRTM, or
bind the run-scoped TauCache context.
A run (:class:`~gtellcorr.telluric.model.TelluricModel`) validates the configured
double-precision bundle once and reuses the validated resource object for all its
LBLRTM work: optical-depth caches, fits and the strong-telluric mask.
"""

from .lblrtm_errors import LblrtmResourceError
from .lblrtm_resources import (
    DEFAULT_MANIFEST_FILENAME,
    LblrtmManifestError,
    LblrtmResourceIntegrityError,
    LblrtmResourceManifest,
    LnflTape5Summary,
    LnflTape6Summary,
    ResolvedLblrtmResourceFiles,
    compute_file_sha256,
    compute_lblrtm_resource_fingerprint,
    load_lblrtm_resource_manifest,
    parse_lnfl_tape5,
    parse_lnfl_tape6,
    resolve_lblrtm_resource_files,
    validate_lblrtm_build_recipe,
    validate_lblrtm_resource_bundle,
    validate_lblrtm_resource_manifest,
)
from .lnfl_molecule_selection import (
    LNFL_MOLECULE_SLOT_COUNT,
    LnflMolecule,
    build_lnfl_molecule_selection_mask,
    load_lnfl_molecule_registry,
    validate_lnfl_molecule_selection_mask,
)
from .tape3_metadata import (
    Tape3FormatError,
    Tape3Header,
    Tape3ValidationError,
    compute_tape3_scientific_sha256,
    compute_tape3_semantic_sha256,
    read_tape3_header,
    validate_tape3_metadata,
)

__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "LNFL_MOLECULE_SLOT_COUNT",
    "LblrtmManifestError",
    "LblrtmResourceError",
    "LblrtmResourceIntegrityError",
    "LblrtmResourceManifest",
    "LnflMolecule",
    "LnflTape5Summary",
    "LnflTape6Summary",
    "ResolvedLblrtmResourceFiles",
    "Tape3FormatError",
    "Tape3Header",
    "Tape3ValidationError",
    "build_lnfl_molecule_selection_mask",
    "compute_file_sha256",
    "compute_lblrtm_resource_fingerprint",
    "compute_tape3_scientific_sha256",
    "compute_tape3_semantic_sha256",
    "load_lblrtm_resource_manifest",
    "load_lnfl_molecule_registry",
    "parse_lnfl_tape5",
    "parse_lnfl_tape6",
    "read_tape3_header",
    "resolve_lblrtm_resource_files",
    "validate_lblrtm_build_recipe",
    "validate_lblrtm_resource_bundle",
    "validate_lblrtm_resource_manifest",
    "validate_lnfl_molecule_selection_mask",
    "validate_tape3_metadata",
]
