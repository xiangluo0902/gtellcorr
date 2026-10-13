"""Strict reader for the official RFM MIPAS mid-latitude-night atmosphere.

The numerical reference file is an external, caller-owned resource.  This
module contains only the parser and gtellcorr's profile format; it does not
bundle or reconstruct any third-party atmospheric values.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from hashlib import sha256
from typing import BinaryIO, Final

import numpy as np

CANONICAL_ALTITUDE_GRID_KM: Final[np.ndarray] = np.arange(121, dtype=np.float64)
CANONICAL_ALTITUDE_GRID_KM.setflags(write=False)

_MAXIMUM_REFERENCE_BYTES: Final[int] = 4 * 1024 * 1024
_REFERENCE_SCHEMA: Final[str] = "gtellcorr_rfm_mipas_reference_v1"
_REQUIRED_HEADER_FRAGMENTS: Final[tuple[str, ...]] = (
    "MID-LATITUDE NIGHT PROFILES FOR MIPAS",
    "03-JAN-2001",
    "ORIGINATOR: J.J. REMEDIOS",
)
_PROFILE_FIELDS_BY_RFM_NAME: Final[Mapping[str, str]] = {
    "HGT": "z_km",
    "PRE": "P_hPa",
    "TEM": "T_K",
    "H2O": "H2O_ppmv",
    "CO2": "CO2_ppmv",
    "O3": "O3_ppmv",
    "N2O": "N2O_ppmv",
    "CO": "CO_ppmv",
    "CH4": "CH4_ppmv",
    "O2": "O2_ppmv",
    "NO": "NO_ppmv",
    "SO2": "SO2_ppmv",
    "NO2": "NO2_ppmv",
    "NH3": "NH3_ppmv",
    "HNO3": "HNO3_ppmv",
}
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_RFM_FIELD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9+]*")


def load_rfm_mipas_reference(
    reference_source: str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
    *,
    expected_sha256: str,
    source_url: str,
) -> dict:
    """Load the official RFM ``ngt.atm`` reference, checked against its SHA-256.

    Parameters
    ----------
    reference_source
        Path, exact bytes, or an open binary stream containing the RFM
        mid-latitude-night MIPAS v3 atmosphere.
    expected_sha256
        SHA-256 of the exact official file bytes.  It is mandatory so a caller
        cannot silently substitute a different climatology.
    source_url
        Persistent source URL recorded with the returned profile.

    Returns
    -------
    dict
        Float64 arrays named with gtellcorr units plus
        ``reference_provenance`` (where the file came from).  The altitude array
        is checked against gtellcorr's fixed grid but not used to build it.
    """

    if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must contain 64 lowercase hexadecimal digits")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("source_url must be a non-empty string")

    if isinstance(reference_source, (bytes, bytearray, memoryview)):
        payload = bytes(reference_source)
    elif isinstance(reference_source, (str, os.PathLike)):
        with open(reference_source, "rb") as reference_file:
            payload = reference_file.read(_MAXIMUM_REFERENCE_BYTES + 1)
    else:
        try:
            payload = reference_source.read(_MAXIMUM_REFERENCE_BYTES + 1)
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            raise TypeError(
                "reference_source must be a path, exact bytes, or a readable binary stream"
            ) from exc
        if not isinstance(payload, bytes):
            raise TypeError("reference_source stream must return bytes")

    if not payload:
        raise ValueError("RFM reference file is empty")
    if len(payload) > _MAXIMUM_REFERENCE_BYTES:
        raise ValueError(f"RFM reference exceeds {_MAXIMUM_REFERENCE_BYTES} bytes")
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"RFM reference SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("RFM reference must be ASCII text") from exc

    lines = text.splitlines()
    line_index = 0
    header_comments: list[str] = []
    while line_index < len(lines):
        stripped = lines[line_index].strip()
        if not stripped:
            line_index += 1
            continue
        if stripped.startswith("!"):
            header_comments.append(stripped[1:].strip())
            line_index += 1
            continue
        break
    if line_index >= len(lines):
        raise ValueError("RFM reference is missing its profile-level record")

    header_identity = " ".join(header_comments).upper()
    missing_identity = [
        fragment for fragment in _REQUIRED_HEADER_FRAGMENTS if fragment not in header_identity
    ]
    if missing_identity:
        raise ValueError(
            "the reference file is not the MIPAS mid-latitude night profile (RFM); "
            f"its header lacks: {missing_identity}"
        )

    level_record = lines[line_index].partition("!")[0].strip().split()
    if len(level_record) != 1:
        raise ValueError("RFM profile-level record must contain one integer")
    try:
        level_count = int(level_record[0])
    except ValueError as exc:
        raise ValueError("RFM profile-level record is not an integer") from exc
    if level_count != CANONICAL_ALTITUDE_GRID_KM.size:
        raise ValueError(
            "RFM reference must contain exactly "
            f"{CANONICAL_ALTITUDE_GRID_KM.size} levels; got {level_count}"
        )
    line_index += 1

    parsed_sections: dict[str, np.ndarray] = {}
    saw_end_marker = False
    while line_index < len(lines):
        stripped = lines[line_index].strip()
        if not stripped or stripped.startswith("!"):
            line_index += 1
            continue
        if not stripped.startswith("*"):
            raise ValueError(f"unexpected RFM content at line {line_index + 1}: {stripped!r}")

        field_token = stripped[1:].split()[0].upper()
        if field_token == "END":
            saw_end_marker = True
            line_index += 1
            break
        if _RFM_FIELD_PATTERN.fullmatch(field_token) is None:
            raise ValueError(
                f"invalid RFM field identifier at line {line_index + 1}: {field_token!r}"
            )
        if field_token in parsed_sections:
            raise ValueError(f"duplicate RFM profile field {field_token}")

        values: list[float] = []
        line_index += 1
        while line_index < len(lines):
            value_line = lines[line_index].strip()
            if not value_line or value_line.startswith("!"):
                line_index += 1
                continue
            if value_line.startswith("*"):
                break
            data_text = value_line.partition("!")[0].strip()
            for token in data_text.split():
                try:
                    values.append(float(token.replace("D", "E").replace("d", "e")))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid numeric token for {field_token} at line "
                        f"{line_index + 1}: {token!r}"
                    ) from exc
            if len(values) > level_count:
                raise ValueError(f"RFM field {field_token} contains more than {level_count} values")
            line_index += 1
        if len(values) != level_count:
            raise ValueError(
                f"RFM field {field_token} contains {len(values)} values; expected {level_count}"
            )
        parsed_sections[field_token] = np.asarray(values, dtype=np.float64)

    if not saw_end_marker:
        raise ValueError("RFM reference is missing the *END marker")
    for trailing_index in range(line_index, len(lines)):
        trailing = lines[trailing_index].strip()
        if trailing and not trailing.startswith("!"):
            raise ValueError(
                "RFM reference contains non-comment content after *END at "
                f"line {trailing_index + 1}"
            )

    missing_fields = sorted(set(_PROFILE_FIELDS_BY_RFM_NAME) - set(parsed_sections))
    if missing_fields:
        raise ValueError(f"RFM reference is missing required fields: {missing_fields}")

    profile = {
        output_name: np.array(
            parsed_sections[rfm_name],
            dtype=np.float64,
            copy=True,
        )
        for rfm_name, output_name in _PROFILE_FIELDS_BY_RFM_NAME.items()
    }
    for field_name, values in profile.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"RFM reference field {field_name} contains non-finite values")
    if not np.array_equal(profile["z_km"], CANONICAL_ALTITUDE_GRID_KM):
        raise ValueError("RFM reference HGT must be the exact 0--120 km, 1 km grid")
    if np.any(profile["P_hPa"] <= 0.0) or not np.all(np.diff(profile["P_hPa"]) < 0.0):
        raise ValueError("RFM reference pressure must be positive and strictly decreasing")
    if np.any(profile["T_K"] <= 0.0):
        raise ValueError("RFM reference temperature must be positive")
    for field_name in _PROFILE_FIELDS_BY_RFM_NAME.values():
        if field_name in {"z_km", "P_hPa", "T_K"}:
            continue
        if np.any(profile[field_name] < 0.0):
            raise ValueError(f"RFM reference molecular field {field_name} must be non-negative")
    for values in profile.values():
        values.setflags(write=False)

    profile["reference_provenance"] = {
        "schema": _REFERENCE_SCHEMA,
        "reference_id": "rfm-mipas-v3-2001-midlatitude-night",
        "provider": "University of Leicester via the Oxford RFM profile archive",
        "source_url": source_url.strip(),
        "source_sha256": actual_sha256,
        "source_format": "RFM .atm ASCII",
        "profile_condition": "mid-latitude night",
        "level_count": int(level_count),
        "altitude_grid": {
            "owner": "gtellcorr",
            "dtype": "float64",
            "start_km": 0.0,
            "stop_km": 120.0,
            "step_km": 1.0,
        },
        "scientific_origin": ("J.J. Remedios, MIPAS model atmospheres, version 3, 03-Jan-2001"),
        "redistribution": (
            "external caller-supplied resource; gtellcorr does not claim "
            "ownership or bundle the numerical profile"
        ),
    }
    return profile
