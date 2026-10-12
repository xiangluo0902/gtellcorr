"""LNFL v3.2 molecule registry and fixed-width selection-mask helpers.

LNFL v3.2 reads one molecule-selection flag per slot from a 47-column field.
Keeping the registry and mask construction here makes that file-format
contract explicit and prevents ambiguous short masks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

LNFL_MOLECULE_SLOT_COUNT: Final = 47


@dataclass(frozen=True, slots=True)
class LnflMolecule:
    """One molecule slot in the LNFL v3.2 selection field."""

    molecule_id: int
    formula: str


_LNFL_V3_2_MOLECULE_REGISTRY: Final[tuple[LnflMolecule, ...]] = (
    LnflMolecule(1, "H2O"),
    LnflMolecule(2, "CO2"),
    LnflMolecule(3, "O3"),
    LnflMolecule(4, "N2O"),
    LnflMolecule(5, "CO"),
    LnflMolecule(6, "CH4"),
    LnflMolecule(7, "O2"),
    LnflMolecule(8, "NO"),
    LnflMolecule(9, "SO2"),
    LnflMolecule(10, "NO2"),
    LnflMolecule(11, "NH3"),
    LnflMolecule(12, "HNO3"),
    LnflMolecule(13, "OH"),
    LnflMolecule(14, "HF"),
    LnflMolecule(15, "HCl"),
    LnflMolecule(16, "HBr"),
    LnflMolecule(17, "HI"),
    LnflMolecule(18, "ClO"),
    LnflMolecule(19, "OCS"),
    LnflMolecule(20, "H2CO"),
    LnflMolecule(21, "HOCl"),
    LnflMolecule(22, "N2"),
    LnflMolecule(23, "HCN"),
    LnflMolecule(24, "CH3Cl"),
    LnflMolecule(25, "H2O2"),
    LnflMolecule(26, "C2H2"),
    LnflMolecule(27, "C2H6"),
    LnflMolecule(28, "PH3"),
    LnflMolecule(29, "COF2"),
    LnflMolecule(30, "SF6"),
    LnflMolecule(31, "H2S"),
    LnflMolecule(32, "HCOOH"),
    LnflMolecule(33, "HO2"),
    LnflMolecule(34, "O"),
    LnflMolecule(35, "ClONO2"),
    LnflMolecule(36, "NO+"),
    LnflMolecule(37, "HOBr"),
    LnflMolecule(38, "C2H4"),
    LnflMolecule(39, "CH3OH"),
    LnflMolecule(40, "CH3Br"),
    LnflMolecule(41, "CH3CN"),
    LnflMolecule(42, "CF4"),
    LnflMolecule(43, "C4H2"),
    LnflMolecule(44, "HC3N"),
    LnflMolecule(45, "H2"),
    LnflMolecule(46, "CS"),
    LnflMolecule(47, "SO3"),
)


def load_lnfl_molecule_registry() -> tuple[LnflMolecule, ...]:
    """Return the immutable LNFL v3.2 molecule registry in slot order."""

    return _LNFL_V3_2_MOLECULE_REGISTRY


def _validate_slot_count(slot_count: object, *, argument_name: str) -> int:
    """Require the complete 47-slot LNFL v3.2 field."""

    if isinstance(slot_count, bool) or not isinstance(slot_count, int):
        raise TypeError(f"{argument_name} must be an integer, not {type(slot_count).__name__}")
    if slot_count != LNFL_MOLECULE_SLOT_COUNT:
        raise ValueError(
            f"{argument_name} must be exactly "
            f"{LNFL_MOLECULE_SLOT_COUNT} for LNFL v3.2, got {slot_count}"
        )
    return slot_count


def build_lnfl_molecule_selection_mask(
    selected_molecule_ids: Iterable[int] | None = None,
    *,
    slot_count: int = LNFL_MOLECULE_SLOT_COUNT,
) -> str:
    """Build an LNFL selection mask with one flag per molecule slot.

    Parameters
    ----------
    selected_molecule_ids
        Molecule IDs to enable. ``None`` enables every slot. An empty iterable
        produces an all-zero mask.
    slot_count
        Width of the LNFL v3.2 field. It must remain 47 even when only a
        scientific subset is selected; disabled trailing slots are explicit
        zeroes rather than an ambiguous short string.

    Returns
    -------
    str
        A fixed-width string containing only ``"0"`` and ``"1"``.
    """

    validated_slot_count = _validate_slot_count(
        slot_count,
        argument_name="slot_count",
    )
    if selected_molecule_ids is None:
        selected_molecule_ids = range(1, validated_slot_count + 1)

    try:
        iterator = iter(selected_molecule_ids)
    except TypeError as exc:
        raise TypeError("selected_molecule_ids must be an iterable of integers or None") from exc

    selected_ids: set[int] = set()
    for position, molecule_id in enumerate(iterator):
        if isinstance(molecule_id, bool) or not isinstance(molecule_id, int):
            raise TypeError(
                "selected_molecule_ids must contain only integers; "
                f"item {position} is {type(molecule_id).__name__}"
            )
        if not 1 <= molecule_id <= validated_slot_count:
            raise ValueError(
                f"molecule ID {molecule_id} is outside the valid range 1..{validated_slot_count}"
            )
        if molecule_id in selected_ids:
            raise ValueError(f"duplicate molecule ID: {molecule_id}")
        selected_ids.add(molecule_id)

    mask = "".join(
        "1" if molecule_id in selected_ids else "0"
        for molecule_id in range(1, validated_slot_count + 1)
    )
    return validate_lnfl_molecule_selection_mask(
        mask,
        expected_slot_count=validated_slot_count,
    )


def validate_lnfl_molecule_selection_mask(
    mask: str,
    *,
    expected_slot_count: int = LNFL_MOLECULE_SLOT_COUNT,
) -> str:
    """Validate and return an LNFL fixed-width molecule-selection mask."""

    validated_slot_count = _validate_slot_count(
        expected_slot_count,
        argument_name="expected_slot_count",
    )
    if not isinstance(mask, str):
        raise TypeError(f"mask must be a string, not {type(mask).__name__}")
    if "\n" in mask or "\r" in mask:
        raise ValueError("mask must not contain newline characters")
    if len(mask) != validated_slot_count:
        raise ValueError(f"mask must contain exactly {validated_slot_count} flags, got {len(mask)}")

    invalid_characters = sorted(set(mask) - {"0", "1"})
    if invalid_characters:
        formatted = ", ".join(repr(character) for character in invalid_characters)
        raise ValueError(f"mask contains characters other than '0' and '1': {formatted}")

    return mask
