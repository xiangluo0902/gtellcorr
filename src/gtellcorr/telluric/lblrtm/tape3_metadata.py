"""Read and validate the fixed metadata record at the start of a TAPE3.

TAPE3 is a Fortran sequential-unformatted file.  The first record is the
``/LINHDR/`` common block written by LNFL and read by LBLRTM v12.17::

    HLINID(10), BMOLID(64), MOLCNT(64), MCNTLC(64), MCNTNL(64),
    SUMSTR(64), LINMOL, FLINLO, FLINHI, LINCNT, ILINLC, ILINNL,
    IREC, IRECTL, HID1(2)

The character fields use eight bytes per element; integers and reals use four
bytes.  This module deliberately depends only on the Python standard library
so resource validation can run before the scientific Python stack is loaded.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .lblrtm_errors import Tape3FormatError, Tape3ValidationError

_RECORD_MARKER_BYTES = 4
_HEADER_IDENTIFIER_OFFSET = 0
_HEADER_IDENTIFIER_BYTES = 10 * 8
_HEADER_IDENTIFIER_ITEM_BYTES = 8
_FREEFORM_HEADER_IDENTIFIER_RANGES = (
    # The current historical/rebuilt label difference is confined to
    # HLINID(1:6). Preserve HLINID(7:10): elements 7, 8, and 10 carry format
    # or version controls, and element 9 has not been proven semantically inert.
    (0, 6 * _HEADER_IDENTIFIER_ITEM_BYTES),
)
_MOLECULE_NAMES_OFFSET = _HEADER_IDENTIFIER_BYTES
_MOLECULE_NAME_BYTES = 8
_MOLECULE_CAPACITY = 64
_MOLECULE_LINE_COUNTS_OFFSET = _MOLECULE_NAMES_OFFSET + (_MOLECULE_CAPACITY * _MOLECULE_NAME_BYTES)
_COUPLED_LINE_COUNTS_OFFSET = _MOLECULE_LINE_COUNTS_OFFSET + (_MOLECULE_CAPACITY * 4)
_NLTE_LINE_COUNTS_OFFSET = _COUPLED_LINE_COUNTS_OFFSET + (_MOLECULE_CAPACITY * 4)
_SUMMED_LINE_STRENGTHS_OFFSET = _NLTE_LINE_COUNTS_OFFSET + (_MOLECULE_CAPACITY * 4)
_LINMOL_OFFSET = _SUMMED_LINE_STRENGTHS_OFFSET + (_MOLECULE_CAPACITY * 4)
_NUMERIC_TRAILER_BYTES = struct.calcsize("iffiiiii")
_HID1_BYTES = 2 * 8
_MINIMUM_HEADER_RECORD_LENGTH = _LINMOL_OFFSET + _NUMERIC_TRAILER_BYTES + _HID1_BYTES
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Tape3Header:
    """Scientific metadata from the first TAPE3 record.

    Entries in each molecule tuple correspond to one-based HITRAN/LNFL
    molecule IDs ``1..linmol``.
    """

    record_marker_bytes: int
    byte_order: str
    header_record_length: int
    molecule_names: tuple[str, ...]
    molecule_line_counts: tuple[int, ...]
    coupled_line_counts: tuple[int, ...]
    nlte_line_counts: tuple[int, ...]
    summed_line_strengths: tuple[float, ...]
    linmol: int
    actual_line_bounds_cm_minus_1: tuple[float, float]
    total_line_count: int
    header_identifier: str
    generated_date: str
    generated_time: str


@dataclass(frozen=True)
class _FirstRecord:
    """Layout and contents of the first TAPE3 record."""

    byte_order: str
    struct_prefix: str
    record_length: int
    fixed_payload: bytes
    hid1: bytes


def _path_label(path: os.PathLike[str] | str) -> str:
    """*path* as a string for messages."""
    return os.fspath(path)


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    path: os.PathLike[str] | str,
    field: str,
) -> bytes:
    """Exactly *size* bytes from *stream* (``Tape3FormatError`` if truncated)."""
    value = stream.read(size)
    if len(value) != size:
        raise Tape3FormatError(
            f"{_path_label(path)}: truncated {field}; expected {size} bytes, read {len(value)}"
        )
    return value


def _inspect_first_record(
    stream: BinaryIO,
    path: os.PathLike[str] | str,
) -> _FirstRecord:
    """Detect byte order and return the validated pieces of record one."""

    marker = _read_exact(
        stream,
        _RECORD_MARKER_BYTES,
        path=path,
        field="opening record marker",
    )
    stream.seek(0, os.SEEK_END)
    file_size = stream.tell()

    candidates: list[_FirstRecord] = []
    diagnostics: list[str] = []
    for byte_order, prefix in (("little", "<"), ("big", ">")):
        record_length = struct.unpack(f"{prefix}I", marker)[0]
        if record_length != _MINIMUM_HEADER_RECORD_LENGTH:
            diagnostics.append(
                f"{byte_order}: record length {record_length} does not equal "
                f"the {_MINIMUM_HEADER_RECORD_LENGTH}-byte LNFL v3.2 "
                "/LINHDR/ layout"
            )
            continue
        if record_length + (2 * _RECORD_MARKER_BYTES) > file_size:
            diagnostics.append(
                f"{byte_order}: record length {record_length} exceeds the {file_size}-byte file"
            )
            continue

        stream.seek(_RECORD_MARKER_BYTES + record_length)
        closing_marker = _read_exact(
            stream,
            _RECORD_MARKER_BYTES,
            path=path,
            field=f"{byte_order}-endian closing record marker",
        )
        closing_length = struct.unpack(f"{prefix}I", closing_marker)[0]
        if closing_length != record_length:
            diagnostics.append(
                f"{byte_order}: closing record marker {closing_length} does "
                f"not match opening marker {record_length}"
            )
            continue

        stream.seek(_RECORD_MARKER_BYTES)
        fixed_payload = _read_exact(
            stream,
            _LINMOL_OFFSET + _NUMERIC_TRAILER_BYTES,
            path=path,
            field="fixed /LINHDR/ payload",
        )
        linmol = struct.unpack_from(f"{prefix}i", fixed_payload, _LINMOL_OFFSET)[0]
        if not 1 <= linmol <= _MOLECULE_CAPACITY:
            diagnostics.append(
                f"{byte_order}: LINMOL={linmol} is outside the supported "
                f"range 1..{_MOLECULE_CAPACITY}"
            )
            continue

        stream.seek(
            _RECORD_MARKER_BYTES + record_length - _HID1_BYTES,
            os.SEEK_SET,
        )
        hid1 = _read_exact(
            stream,
            _HID1_BYTES,
            path=path,
            field="HID1 date/time fields",
        )
        candidates.append(
            _FirstRecord(
                byte_order=byte_order,
                struct_prefix=prefix,
                record_length=record_length,
                fixed_payload=fixed_payload,
                hid1=hid1,
            )
        )

    if not candidates:
        detail = "; ".join(diagnostics)
        raise Tape3FormatError(
            f"{_path_label(path)}: cannot identify a valid little- or "
            f"big-endian 4-byte Fortran header record ({detail})"
        )
    if len(candidates) != 1:
        orders = ", ".join(candidate.byte_order for candidate in candidates)
        raise Tape3FormatError(
            f"{_path_label(path)}: ambiguous first-record byte order; "
            f"valid interpretations: {orders}"
        )
    return candidates[0]


def _decode_ascii(
    value: bytes,
    *,
    path: os.PathLike[str] | str,
    field: str,
) -> str:
    """An ASCII field, stripped of blanks and NULs."""
    try:
        return value.decode("ascii").strip(" \x00")
    except UnicodeDecodeError as exc:
        raise Tape3FormatError(f"{_path_label(path)}: {field} is not ASCII text") from exc


def _unpack_array(
    payload: bytes,
    *,
    prefix: str,
    format_code: str,
    count: int,
    offset: int,
) -> tuple[int, ...] | tuple[float, ...]:
    """*count* values of *format_code* at *offset* in the record payload."""
    return struct.unpack_from(f"{prefix}{count}{format_code}", payload, offset)


def read_tape3_header(path: os.PathLike[str] | str) -> Tape3Header:
    """Parse and strictly validate the first record of *path*.

    Both little- and big-endian four-byte Fortran record markers are
    recognised.  The opening/closing markers, minimum payload size and
    ``LINMOL`` range are validated before any metadata is returned.
    """

    tape3_path = Path(path)
    with tape3_path.open("rb") as stream:
        first_record = _inspect_first_record(stream, tape3_path)

    payload = first_record.fixed_payload
    prefix = first_record.struct_prefix
    (
        linmol,
        lower_bound,
        upper_bound,
        total_line_count,
        _coupled_line_count,
        _nlte_line_count,
        _record_count,
        _record_length,
    ) = struct.unpack_from(f"{prefix}iffiiiii", payload, _LINMOL_OFFSET)

    molecule_names = tuple(
        _decode_ascii(
            payload[
                _MOLECULE_NAMES_OFFSET + (index * _MOLECULE_NAME_BYTES) : _MOLECULE_NAMES_OFFSET
                + ((index + 1) * _MOLECULE_NAME_BYTES)
            ],
            path=tape3_path,
            field=f"molecule name {index + 1}",
        )
        for index in range(linmol)
    )
    molecule_line_counts = _unpack_array(
        payload,
        prefix=prefix,
        format_code="i",
        count=linmol,
        offset=_MOLECULE_LINE_COUNTS_OFFSET,
    )
    coupled_line_counts = _unpack_array(
        payload,
        prefix=prefix,
        format_code="i",
        count=linmol,
        offset=_COUPLED_LINE_COUNTS_OFFSET,
    )
    nlte_line_counts = _unpack_array(
        payload,
        prefix=prefix,
        format_code="i",
        count=linmol,
        offset=_NLTE_LINE_COUNTS_OFFSET,
    )
    summed_line_strengths = _unpack_array(
        payload,
        prefix=prefix,
        format_code="f",
        count=linmol,
        offset=_SUMMED_LINE_STRENGTHS_OFFSET,
    )

    count_fields = (
        ("molecule line count", molecule_line_counts),
        ("coupled line count", coupled_line_counts),
        ("NLTE line count", nlte_line_counts),
    )
    for field_name, values in count_fields:
        for molecule_id, value in enumerate(values, start=1):
            if value < 0:
                raise Tape3FormatError(
                    f"{tape3_path}: {field_name} for molecule {molecule_id} is negative ({value})"
                )
    if total_line_count < 0:
        raise Tape3FormatError(f"{tape3_path}: total line count is negative ({total_line_count})")
    if not (
        math.isfinite(lower_bound) and math.isfinite(upper_bound) and lower_bound <= upper_bound
    ):
        raise Tape3FormatError(
            f"{tape3_path}: invalid actual line bounds ({lower_bound}, {upper_bound}) cm^-1"
        )
    for molecule_id, value in enumerate(summed_line_strengths, start=1):
        if not math.isfinite(value):
            raise Tape3FormatError(
                f"{tape3_path}: summed line strength for molecule "
                f"{molecule_id} is not finite ({value})"
            )

    header_identifier = _decode_ascii(
        payload[_HEADER_IDENTIFIER_OFFSET : _HEADER_IDENTIFIER_OFFSET + _HEADER_IDENTIFIER_BYTES],
        path=tape3_path,
        field="header identifier",
    )
    generated_date = _decode_ascii(
        first_record.hid1[:8],
        path=tape3_path,
        field="generated date",
    )
    generated_time = _decode_ascii(
        first_record.hid1[8:],
        path=tape3_path,
        field="generated time",
    )

    return Tape3Header(
        record_marker_bytes=_RECORD_MARKER_BYTES,
        byte_order=first_record.byte_order,
        header_record_length=first_record.record_length,
        molecule_names=molecule_names,
        molecule_line_counts=tuple(molecule_line_counts),
        coupled_line_counts=tuple(coupled_line_counts),
        nlte_line_counts=tuple(nlte_line_counts),
        summed_line_strengths=tuple(summed_line_strengths),
        linmol=linmol,
        actual_line_bounds_cm_minus_1=(lower_bound, upper_bound),
        total_line_count=total_line_count,
        header_identifier=header_identifier,
        generated_date=generated_date,
        generated_time=generated_time,
    )


def _hash_exact_bytes(
    stream: BinaryIO,
    digest: object,
    size: int,
    *,
    path: os.PathLike[str] | str,
) -> None:
    """Feed exactly *size* bytes of *stream* into *digest*."""
    remaining = size
    while remaining:
        chunk = stream.read(min(_HASH_CHUNK_BYTES, remaining))
        if not chunk:
            raise Tape3FormatError(
                f"{_path_label(path)}: the file was truncated while its checksum was computed"
            )
        digest.update(chunk)
        remaining -= len(chunk)


def _compute_normalized_tape3_sha256(
    path: os.PathLike[str] | str,
    *,
    normalize_freeform_header_identifier: bool,
) -> str:
    """SHA-256 of a TAPE3 with the HID1 date/time bytes (and, if
    *normalize_freeform_header_identifier*, the free-form HLINID title) replaced
    by zeros, so rebuilt files that differ only there hash alike."""
    tape3_path = Path(path)
    digest = hashlib.sha256()
    with tape3_path.open("rb") as stream:
        first_record = _inspect_first_record(stream, tape3_path)
        stream.seek(0)

        normalized_ranges = [
            (
                _RECORD_MARKER_BYTES + first_record.record_length - _HID1_BYTES,
                _HID1_BYTES,
                "HID1 date/time fields",
            )
        ]
        if normalize_freeform_header_identifier:
            normalized_ranges.extend(
                (
                    _RECORD_MARKER_BYTES + _HEADER_IDENTIFIER_OFFSET + relative_start,
                    size,
                    "free-form HLINID provenance title",
                )
                for relative_start, size in _FREEFORM_HEADER_IDENTIFIER_RANGES
            )
        cursor = 0
        for start, size, field in sorted(normalized_ranges):
            _hash_exact_bytes(
                stream,
                digest,
                start - cursor,
                path=tape3_path,
            )
            _read_exact(stream, size, path=tape3_path, field=field)
            digest.update(b"\x00" * size)
            cursor = start + size

        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_tape3_semantic_sha256(path: os.PathLike[str] | str) -> str:
    """Return SHA256 after zeroing only the first record's HID1 fields.

    LNFL writes its eight-byte generation date and eight-byte generation time
    at the end of the first record.  Those 16 bytes are replaced by zeroes in
    the hash stream; every other byte, including the human-readable HLINID
    title and both record markers, remains unchanged.  The TAPE3 itself is
    never modified.
    """

    return _compute_normalized_tape3_sha256(
        path,
        normalize_freeform_header_identifier=False,
    )


def compute_tape3_scientific_sha256(path: os.PathLike[str] | str) -> str:
    """Return a build-label-independent SHA256 for TAPE3 scientific content.

    In addition to HID1's generation date/time, this digest normalizes only
    the free-form HLINID title elements 1--6 copied from LNFL TAPE5. HLINID
    elements 7--10 remain hashed; elements 7, 8, and 10 carry the negative-EPP
    layout flag, LNFL version field, and isotope-layout flag, while element 9
    has not been proven semantically inert. All molecule metadata, line
    records, record markers, and numeric bytes remain hashed exactly.

    The result can therefore prove that a renamed gtellcorr rebuild has
    the same line database and format-control metadata as the historical
    reference, while :func:`compute_tape3_semantic_sha256` still distinguishes
    every title element.
    """

    return _compute_normalized_tape3_sha256(
        path,
        normalize_freeform_header_identifier=True,
    )


def validate_tape3_metadata(
    header: Tape3Header,
    *,
    expected_linmol: int | None = None,
    expected_actual_line_bounds_cm_minus_1: tuple[float, float] | None = None,
    expected_bounds_cm_minus_1: tuple[float, float] | None = None,
    bounds_tolerance_cm_minus_1: float = 0.0,
    expected_total_line_count: int | None = None,
    expected_molecule_line_counts: Mapping[int, int] | None = None,
) -> None:
    """Validate parsed metadata against explicitly supplied expectations.

    ``expected_molecule_line_counts`` is keyed by one-based molecule ID.  The
    shorter ``expected_bounds_cm_minus_1`` keyword is retained as an exact
    alias for ``expected_actual_line_bounds_cm_minus_1``; supplying both is an
    error.
    """

    if (
        expected_actual_line_bounds_cm_minus_1 is not None
        and expected_bounds_cm_minus_1 is not None
    ):
        raise Tape3ValidationError(
            "supply only one of expected_actual_line_bounds_cm_minus_1 "
            "and expected_bounds_cm_minus_1"
        )
    expected_bounds = (
        expected_actual_line_bounds_cm_minus_1
        if expected_actual_line_bounds_cm_minus_1 is not None
        else expected_bounds_cm_minus_1
    )
    if not math.isfinite(bounds_tolerance_cm_minus_1):
        raise Tape3ValidationError("bounds_tolerance_cm_minus_1 must be finite")
    if bounds_tolerance_cm_minus_1 < 0:
        raise Tape3ValidationError("bounds_tolerance_cm_minus_1 must be non-negative")

    mismatches: list[str] = []
    if expected_linmol is not None and header.linmol != expected_linmol:
        mismatches.append(f"LINMOL: expected {expected_linmol}, found {header.linmol}")
    if (
        expected_total_line_count is not None
        and header.total_line_count != expected_total_line_count
    ):
        mismatches.append(
            "total line count: expected "
            f"{expected_total_line_count}, found {header.total_line_count}"
        )

    if expected_bounds is not None:
        if len(expected_bounds) != 2:
            raise Tape3ValidationError("expected line bounds must contain exactly two values")
        for label, expected, actual in zip(
            ("lower", "upper"),
            expected_bounds,
            header.actual_line_bounds_cm_minus_1,
            strict=False,
        ):
            if not math.isfinite(expected):
                raise Tape3ValidationError(f"expected {label} line bound must be finite")
            if abs(actual - expected) > bounds_tolerance_cm_minus_1:
                mismatches.append(
                    f"{label} line bound: expected {expected} cm^-1, "
                    f"found {actual} cm^-1 "
                    f"(tolerance {bounds_tolerance_cm_minus_1} cm^-1)"
                )

    if expected_molecule_line_counts is not None:
        for molecule_id, expected_count in expected_molecule_line_counts.items():
            if isinstance(molecule_id, bool) or not isinstance(molecule_id, int):
                raise Tape3ValidationError(
                    "expected molecule line counts must use integer, one-based molecule IDs"
                )
            if not 1 <= molecule_id <= header.linmol:
                raise Tape3ValidationError(
                    f"expected molecule ID {molecule_id} is outside the "
                    f"parsed range 1..{header.linmol}"
                )
            if isinstance(expected_count, bool) or not isinstance(expected_count, int):
                raise Tape3ValidationError(
                    f"expected line count for molecule {molecule_id} must be an integer"
                )
            actual_count = header.molecule_line_counts[molecule_id - 1]
            if actual_count != expected_count:
                molecule_name = header.molecule_names[molecule_id - 1]
                mismatches.append(
                    f"molecule {molecule_id} ({molecule_name}) line count: "
                    f"expected {expected_count}, found {actual_count}"
                )

    if mismatches:
        raise Tape3ValidationError("TAPE3 metadata validation failed: " + "; ".join(mismatches))


__all__ = [
    "Tape3FormatError",
    "Tape3Header",
    "Tape3ValidationError",
    "compute_tape3_scientific_sha256",
    "compute_tape3_semantic_sha256",
    "read_tape3_header",
    "validate_tape3_metadata",
]
