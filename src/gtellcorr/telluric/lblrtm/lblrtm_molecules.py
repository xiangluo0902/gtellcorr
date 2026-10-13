"""LBLRTM runtime-molecule names and full-profile field mappings.

This module defines the 12 molecules of the user atmosphere passed to
LBLRTM.  It is deliberately separate from
``lnfl_molecule_selection``: LNFL selects spectral-line data using 47 fixed
slots, whereas an LBLRTM run supplies profiles for molecule IDs 1--12.

The registry is project-owned metadata derived independently from the
``ORDER OF MOLECULES``/``HMOLC`` records in official AER LBLRTM v12.17
``src/lblatm.f90`` (tag commit
``a85ac73447c1e62401a57a34bbcb040683345dca``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

from .lblrtm_data_models import LblrtmMoleculeDefinition

_LBLRTM_PROFILE_MOLECULES: Final[tuple[LblrtmMoleculeDefinition, ...]] = (
    LblrtmMoleculeDefinition(
        1,
        "H2O",
        "H2O_ppmv",
        ("water", "water vapor", "water vapour"),
    ),
    LblrtmMoleculeDefinition(
        2,
        "CO2",
        "CO2_ppmv",
        ("carbon dioxide",),
    ),
    LblrtmMoleculeDefinition(
        3,
        "O3",
        "O3_ppmv",
        ("ozone",),
    ),
    LblrtmMoleculeDefinition(
        4,
        "N2O",
        "N2O_ppmv",
        ("nitrous oxide",),
    ),
    LblrtmMoleculeDefinition(
        5,
        "CO",
        "CO_ppmv",
        ("carbon monoxide",),
    ),
    LblrtmMoleculeDefinition(
        6,
        "CH4",
        "CH4_ppmv",
        ("methane",),
    ),
    LblrtmMoleculeDefinition(
        7,
        "O2",
        "O2_ppmv",
        ("molecular oxygen", "dioxygen"),
    ),
    LblrtmMoleculeDefinition(
        8,
        "NO",
        "NO_ppmv",
        ("nitric oxide", "nitrogen monoxide"),
    ),
    LblrtmMoleculeDefinition(
        9,
        "SO2",
        "SO2_ppmv",
        ("sulfur dioxide", "sulphur dioxide"),
    ),
    LblrtmMoleculeDefinition(
        10,
        "NO2",
        "NO2_ppmv",
        ("nitrogen dioxide",),
    ),
    LblrtmMoleculeDefinition(
        11,
        "NH3",
        "NH3_ppmv",
        ("ammonia",),
    ),
    LblrtmMoleculeDefinition(
        12,
        "HNO3",
        "HNO3_ppmv",
        ("nitric acid",),
    ),
)


LBLRTM_MOLECULES_BY_ID: Final[Mapping[int, LblrtmMoleculeDefinition]] = MappingProxyType(
    {molecule.molecule_id: molecule for molecule in _LBLRTM_PROFILE_MOLECULES}
)
"""Read-only molecule definitions keyed by LBLRTM molecule ID."""


LBLRTM_MOLECULES_BY_NAME: Final[Mapping[str, LblrtmMoleculeDefinition]] = MappingProxyType(
    {molecule.canonical_name: molecule for molecule in _LBLRTM_PROFILE_MOLECULES}
)
"""Read-only molecule definitions keyed by canonical formula."""


def _normalize_molecule_alias_token(name: str) -> str:
    """Return a case-insensitive lookup token for one molecule name.

    ASCII letters, digits, spaces, underscores, and hyphens are accepted.
    Runs of separators are made equivalent, so ``"water_vapor"`` and
    ``"water vapor"`` address the same explicitly registered alias.  Other
    punctuation is rejected instead of being silently discarded.
    """

    if not isinstance(name, str):
        raise TypeError(f"molecule name must be a string, not {type(name).__name__}")

    stripped_name = name.strip()
    if not stripped_name:
        raise ValueError("molecule name must not be empty")
    if re.fullmatch(r"[A-Za-z0-9]+(?:[ _-]+[A-Za-z0-9]+)*", stripped_name) is None:
        raise ValueError(
            "molecule name may contain only ASCII letters, digits, spaces, underscores, and hyphens"
        )

    return " ".join(re.split(r"[ _-]+", stripped_name)).casefold()


_LBLRTM_MOLECULE_ALIAS_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    (
        _normalize_molecule_alias_token(alias),
        molecule.canonical_name,
    )
    for molecule in _LBLRTM_PROFILE_MOLECULES
    for alias in (
        molecule.canonical_name,
        molecule.profile_field,
        *molecule.aliases,
    )
)

if len(dict(_LBLRTM_MOLECULE_ALIAS_PAIRS)) != len(_LBLRTM_MOLECULE_ALIAS_PAIRS):
    raise RuntimeError("LBLRTM molecule aliases must be unique after normalization")

_LBLRTM_MOLECULE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    dict(_LBLRTM_MOLECULE_ALIAS_PAIRS)
)


def normalize_lblrtm_molecule_name(name: str) -> str:
    """Resolve a canonical name, profile field, or alias to its formula.

    Parameters
    ----------
    name
        Canonical formula such as ``"H2O"``, a full-profile field such as
        ``"H2O_ppmv"``, or an explicitly registered unambiguous alias such as
        ``"water vapor"``.

    Returns
    -------
    str
        The canonical uppercase LBLRTM formula.
    """

    alias_token = _normalize_molecule_alias_token(name)
    try:
        return _LBLRTM_MOLECULE_ALIASES[alias_token]
    except KeyError as exc:
        expected_names = ", ".join(LBLRTM_MOLECULES_BY_NAME)
        raise ValueError(
            f"unknown LBLRTM profile molecule {name!r}; expected one of: {expected_names}"
        ) from exc


def get_lblrtm_molecule_definition(
    name: str,
) -> LblrtmMoleculeDefinition:
    """Return the definition for a canonical name or alias."""

    canonical_name = normalize_lblrtm_molecule_name(name)
    return LBLRTM_MOLECULES_BY_NAME[canonical_name]


def validate_lblrtm_molecule_names(
    names: Iterable[str],
) -> tuple[str, ...]:
    """Validate an ordered molecule collection and return canonical names.

    Input order is retained.  The collection must be non-empty, and aliases
    that resolve to the same molecule count as duplicates.
    """

    if isinstance(names, (str, bytes)):
        raise TypeError("names must be an iterable of molecule-name strings, not a single string")
    try:
        iterator = iter(names)
    except TypeError as exc:
        raise TypeError("names must be an iterable of molecule-name strings") from exc

    canonical_names: list[str] = []
    seen_names: set[str] = set()
    for position, name in enumerate(iterator):
        try:
            canonical_name = normalize_lblrtm_molecule_name(name)
        except (TypeError, ValueError) as exc:
            exception_type = type(exc)
            raise exception_type(f"invalid molecule name at position {position}: {exc}") from exc

        if canonical_name in seen_names:
            raise ValueError(
                f"duplicate LBLRTM profile molecule after normalization: {canonical_name}"
            )
        seen_names.add(canonical_name)
        canonical_names.append(canonical_name)

    if not canonical_names:
        raise ValueError("names must contain at least one molecule")

    return tuple(canonical_names)


def order_lblrtm_molecule_names(
    names: Iterable[str],
) -> tuple[str, ...]:
    """Return validated canonical names in ascending LBLRTM molecule-ID order."""

    canonical_names = validate_lblrtm_molecule_names(names)
    return tuple(
        sorted(
            canonical_names,
            key=lambda name: LBLRTM_MOLECULES_BY_NAME[name].molecule_id,
        )
    )


__all__ = [
    "LBLRTM_MOLECULES_BY_ID",
    "LBLRTM_MOLECULES_BY_NAME",
    "get_lblrtm_molecule_definition",
    "normalize_lblrtm_molecule_name",
    "order_lblrtm_molecule_names",
    "validate_lblrtm_molecule_names",
]
