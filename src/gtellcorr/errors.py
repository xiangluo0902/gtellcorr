"""Exception hierarchy.

Every error raised deliberately by gtellcorr derives from :class:`GtellcorrError`,
so callers can catch the whole family with one ``except``. Each subclass also
derives from the closest built-in exception, so generic handlers keep working
(for example ``except ValueError`` still catches a :class:`ConfigError`).
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = [
    "ConfigError",
    "DataError",
    "FitError",
    "GtellcorrError",
    "RegistryError",
    "ResourceError",
    "did_you_mean",
]


class GtellcorrError(Exception):
    """Base class of every error raised deliberately by gtellcorr."""


class ConfigError(GtellcorrError, ValueError):
    """Invalid, unknown or inconsistent configuration value."""


class RegistryError(GtellcorrError, LookupError):
    """Unknown or duplicate name in a registry (instrument, site, plugin …)."""


class DataError(GtellcorrError, ValueError):
    """Input data cannot be read or does not match its declared format."""


class ResourceError(GtellcorrError, RuntimeError):
    """An external resource (LBLRTM bundle, reference file, cache) is missing or invalid."""


class FitError(GtellcorrError, RuntimeError):
    """A fit could not be set up or did not produce a usable result."""


def did_you_mean(name: str, candidates: Iterable[str], *, n: int = 3) -> str:
    """Return a ``" Did you mean …?"`` hint for a misspelt *name*, or ``""``.

    Matching is case-insensitive; the hint shows the candidates' original spelling.
    """
    by_folded: dict[str, str] = {}
    for candidate in candidates:
        by_folded.setdefault(candidate.casefold(), candidate)
    matches = difflib.get_close_matches(name.casefold(), list(by_folded), n=n, cutoff=0.6)
    if not matches:
        return ""
    return " Did you mean " + " or ".join(repr(by_folded[m]) for m in matches) + "?"
