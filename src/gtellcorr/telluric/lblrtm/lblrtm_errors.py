"""Exception hierarchy for the direct LBLRTM backend.

All direct-LBL modules import their exceptions from this module.  Modules
which historically defined a more specific exception re-export the same
class object so existing import paths remain valid.
"""

from __future__ import annotations


class LblrtmError(RuntimeError):
    """Base class for all direct LBLRTM backend failures."""


class LblrtmResourceError(LblrtmError):
    """Base class for missing, invalid, or inconsistent LBLRTM resources."""


class LblrtmManifestError(LblrtmResourceError, ValueError):
    """Raised when a resource manifest does not have the expected format."""


class LblrtmResourceIntegrityError(LblrtmResourceError):
    """Raised when a resolved resource bundle differs from its manifest."""


class LblrtmInputError(LblrtmError, ValueError):
    """Raised when a direct radiative-transfer request is invalid."""


class LblrtmExecutionError(LblrtmError):
    """Raised when the LBLRTM executable cannot complete successfully."""


class LblrtmOutputError(LblrtmError):
    """Raised when an LBLRTM output is missing, malformed, or inconsistent."""


class OpacityBasisCacheError(LblrtmError):
    """Raised when an opacity-basis cache artifact is invalid or unusable."""


class Tape3FormatError(LblrtmResourceError, ValueError):
    """Raised when a TAPE3 does not have a valid first Fortran record."""


class Tape3ValidationError(LblrtmResourceError, ValueError):
    """Raised when parsed TAPE3 metadata disagrees with an expectation."""


__all__ = [
    "LblrtmError",
    "LblrtmExecutionError",
    "LblrtmInputError",
    "LblrtmManifestError",
    "LblrtmOutputError",
    "LblrtmResourceError",
    "LblrtmResourceIntegrityError",
    "OpacityBasisCacheError",
    "Tape3FormatError",
    "Tape3ValidationError",
]
