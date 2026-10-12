"""Offline, hash-pinned CelesTrak space-weather input for NRLMSISE-00.

The numerical CSV is an external, caller-owned resource.  Production code
never downloads it and never reads ``pymsis`` private module state.  This
module independently derives the three inputs that are passed explicitly to
``pymsis.calculate``:

* previous-day observed F10.7;
* the current day's centered 81-day observed F10.7 average;
* the seven-element storm-time Ap vector defined by NRLMSISE-00.
"""

from __future__ import annotations

import csv
import io
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from hashlib import sha256
from types import MappingProxyType
from typing import BinaryIO, Final

import numpy as np

from gtellcorr.errors import ResourceError

_MAXIMUM_SPACE_WEATHER_BYTES: Final[int] = 16 * 1024 * 1024
_SPACE_WEATHER_SCHEMA: Final[str] = "gtellcorr_celestrak_space_weather_reference_v1"
_MSIS_INPUT_SCHEMA: Final[str] = "gtellcorr_msis_space_weather_input_v2"
_QUALIFIER_WARNING_POLICY: Final[str] = (
    "warn_if_selected_previous_f107_or_current_center81_is_int_or_prd_v1"
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "DATE",
    "AP1",
    "AP2",
    "AP3",
    "AP4",
    "AP5",
    "AP6",
    "AP7",
    "AP8",
    "AP_AVG",
    "F10.7_OBS",
    "F10.7_DATA_TYPE",
    "F10.7_OBS_CENTER81",
)
_ACCEPTED_F107_DATA_TYPES: Final[frozenset[str]] = frozenset({"OBS", "INT", "PRD", "PRM"})


@dataclass(frozen=True)
class _DailySpaceWeather:
    """One complete daily record needed by the MSIS input convention."""

    day: date
    ap_three_hour: tuple[float, ...]
    ap_daily: float
    f107_observed: float
    f107_data_type: str
    f107a_observed_center81: float


@dataclass(frozen=True)
class CelestrakSpaceWeather:
    """Validated daily records plus immutable resource provenance."""

    records: Mapping[date, _DailySpaceWeather]
    provenance: Mapping[str, object]


def _read_exact_bytes(
    source: (str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO),
) -> bytes:
    """The exact bytes of *source* (bytes, a path, or a binary file object)."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as space_weather_file:
            payload = space_weather_file.read(_MAXIMUM_SPACE_WEATHER_BYTES + 1)
    else:
        try:
            payload = source.read(_MAXIMUM_SPACE_WEATHER_BYTES + 1)
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            raise TypeError(
                "space-weather source must be a path, exact bytes, or a readable binary stream"
            ) from exc
        if not isinstance(payload, bytes):
            raise TypeError("space-weather stream must return bytes")

    if not payload:
        raise ValueError("CelesTrak space-weather file is empty")
    if len(payload) > _MAXIMUM_SPACE_WEATHER_BYTES:
        raise ValueError(
            f"CelesTrak space-weather file exceeds {_MAXIMUM_SPACE_WEATHER_BYTES} bytes"
        )
    return payload


def _parse_finite_non_negative(
    value: str,
    *,
    column: str,
    line_number: int,
) -> float:
    """*value* as a finite, non-negative float (``ValueError`` naming the line)."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column} value at CSV line {line_number}: {value!r}") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{column} must be finite and non-negative at CSV line {line_number}")
    return number


def load_celestrak_space_weather(
    source: (str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO),
    *,
    expected_sha256: str,
    source_url: str,
) -> CelestrakSpaceWeather:
    """Load a complete daily CelesTrak ``SW-All.csv`` resource.

    Monthly ``PRM`` prediction rows are excluded because their Ap fields are
    not daily values.  Daily observed, interpolated and predicted records are
    retained and their data type is recorded in every query's provenance.
    """

    if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must contain 64 lowercase hexadecimal digits")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("source_url must be a non-empty string")

    payload = _read_exact_bytes(source)
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "CelesTrak space-weather SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("CelesTrak space-weather resource must be ASCII CSV") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError("CelesTrak space-weather CSV is missing its header")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CelesTrak space-weather CSV contains duplicate header columns")
    missing_columns = sorted(set(_REQUIRED_COLUMNS) - set(reader.fieldnames))
    if missing_columns:
        raise ValueError(
            f"CelesTrak space-weather CSV is missing required columns: {missing_columns}"
        )

    records: dict[date, _DailySpaceWeather] = {}
    excluded_monthly_rows = 0
    previous_day: date | None = None
    for line_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(
                f"CelesTrak space-weather CSV contains extra fields at line {line_number}"
            )
        data_type = (row["F10.7_DATA_TYPE"] or "").strip().upper()
        if data_type not in _ACCEPTED_F107_DATA_TYPES:
            raise ValueError(
                f"unsupported F10.7_DATA_TYPE at CSV line {line_number}: {data_type!r}"
            )
        if data_type == "PRM":
            excluded_monthly_rows += 1
            continue
        try:
            current_day = date.fromisoformat((row["DATE"] or "").strip())
        except ValueError as exc:
            raise ValueError(f"invalid DATE at CSV line {line_number}: {row['DATE']!r}") from exc
        if current_day in records:
            raise ValueError(f"duplicate daily space-weather record for {current_day}")
        if previous_day is not None and current_day != previous_day + timedelta(days=1):
            raise ValueError(
                "daily space-weather records must be contiguous; "
                f"{previous_day} is followed by {current_day}"
            )

        ap_three_hour = tuple(
            _parse_finite_non_negative(
                row[f"AP{index}"],
                column=f"AP{index}",
                line_number=line_number,
            )
            for index in range(1, 9)
        )
        ap_daily = _parse_finite_non_negative(
            row["AP_AVG"],
            column="AP_AVG",
            line_number=line_number,
        )
        f107_observed = _parse_finite_non_negative(
            row["F10.7_OBS"],
            column="F10.7_OBS",
            line_number=line_number,
        )
        f107a_observed_center81 = _parse_finite_non_negative(
            row["F10.7_OBS_CENTER81"],
            column="F10.7_OBS_CENTER81",
            line_number=line_number,
        )
        if not 0.0 < f107a_observed_center81 <= 400.0:
            raise ValueError(f"F10.7_OBS_CENTER81 must be in (0, 400] at CSV line {line_number}")

        records[current_day] = _DailySpaceWeather(
            day=current_day,
            ap_three_hour=ap_three_hour,
            ap_daily=ap_daily,
            f107_observed=f107_observed,
            f107_data_type=data_type,
            f107a_observed_center81=f107a_observed_center81,
        )
        previous_day = current_day

    if not records:
        raise ValueError("CelesTrak space-weather CSV contains no complete daily records")
    ordered_days = tuple(records)
    provenance = MappingProxyType(
        {
            "schema": _SPACE_WEATHER_SCHEMA,
            "provider": "CelesTrak",
            "source_url": source_url.strip(),
            "source_sha256": actual_sha256,
            "source_format": "CelesTrak SW-All.csv",
            "byte_count": len(payload),
            "first_daily_date": ordered_days[0].isoformat(),
            "last_daily_date": ordered_days[-1].isoformat(),
            "daily_record_count": len(records),
            "excluded_monthly_prediction_count": excluded_monthly_rows,
            "parser_policy": ("complete_daily_obs_int_prd_rows_prm_excluded_v1"),
            "redistribution": (
                "external caller-supplied resource; gtellcorr does not bundle the numerical table"
            ),
        }
    )
    return CelestrakSpaceWeather(
        records=MappingProxyType(records),
        provenance=provenance,
    )


def _normalize_utc(when_utc: datetime) -> datetime:
    """*when_utc* as an aware UTC datetime (naive values are taken as UTC)."""
    if not isinstance(when_utc, datetime):
        raise TypeError("when_utc must be a datetime")
    if when_utc.tzinfo is None:
        return when_utc.replace(tzinfo=UTC)
    return when_utc.astimezone(UTC)


def _three_hour_ap(
    reference: CelestrakSpaceWeather,
    when_utc: datetime,
) -> float:
    """The 3-hour Ap index of the interval containing *when_utc*."""
    record = reference.records.get(when_utc.date())
    if record is None:
        raise ResourceError(
            "the CelesTrak space-weather table has no data for "
            f"{when_utc.date().isoformat()} (the upper atmosphere uses the geomagnetic index "
            "of the 57 hours before each exposure); run 'gtellcorr resources fetch "
            "celestrak' for a complete table"
        )
    return record.ap_three_hour[when_utc.hour // 3]


def get_msis_space_weather_inputs(
    reference: CelestrakSpaceWeather,
    when_utc: datetime,
) -> Mapping[str, object]:
    """Derive explicit NRLMSISE-00 F10.7/F10.7a/storm-time Ap inputs."""

    when = _normalize_utc(when_utc)
    current_record = reference.records.get(when.date())
    previous_f107_day = when.date() - timedelta(days=1)
    previous_record = reference.records.get(previous_f107_day)
    if current_record is None or previous_record is None:
        raise ResourceError(
            f"the CelesTrak space-weather table has no data for {when.date().isoformat()} "
            "or the day before; run 'gtellcorr resources fetch celestrak' for a newer table"
        )

    f107_previous_day = previous_record.f107_observed
    f107_substituted = f107_previous_day <= 0.0 or f107_previous_day > 400.0
    if f107_substituted:
        f107_previous_day = previous_record.f107a_observed_center81
    previous_f107_effective_data_type = (
        "INT" if f107_substituted else previous_record.f107_data_type
    )
    center81_f107a_data_type = current_record.f107_data_type
    qualifier_warning_emitted = previous_f107_effective_data_type in {
        "INT",
        "PRD",
    } or center81_f107a_data_type in {"INT", "PRD"}
    if qualifier_warning_emitted:
        kinds = {"INT": "interpolated", "PRD": "predicted", "OBS": "observed"}
        previous = kinds.get(previous_f107_effective_data_type, previous_f107_effective_data_type)
        mean = kinds.get(center81_f107a_data_type, center81_f107a_data_type)
        warnings.warn(
            f"CelesTrak solar flux for {when.date().isoformat()} is not all observed "
            f"(F10.7 of the day before: {previous}; 81-day mean F10.7a: {mean}). These "
            "values are used for the upper atmosphere and recorded with its profile.",
            UserWarning,
            stacklevel=2,
        )

    ap_values = [
        current_record.ap_daily,
        _three_hour_ap(reference, when),
        _three_hour_ap(reference, when - timedelta(hours=3)),
        _three_hour_ap(reference, when - timedelta(hours=6)),
        _three_hour_ap(reference, when - timedelta(hours=9)),
        float(
            np.mean(
                [
                    _three_hour_ap(reference, when - timedelta(hours=offset))
                    for offset in range(12, 34, 3)
                ],
                dtype=np.float64,
            )
        ),
        float(
            np.mean(
                [
                    _three_hour_ap(reference, when - timedelta(hours=offset))
                    for offset in range(36, 58, 3)
                ],
                dtype=np.float64,
            )
        ),
    ]
    ap_storm_vector = np.asarray(ap_values, dtype=np.float64)
    if (
        ap_storm_vector.shape != (7,)
        or not np.all(np.isfinite(ap_storm_vector))
        or np.any(ap_storm_vector < 0.0)
    ):
        raise ValueError("derived NRLMSISE-00 storm-time Ap vector is invalid")
    ap_storm_vector.setflags(write=False)
    canonical_ap = np.ascontiguousarray(ap_storm_vector, dtype="<f8")
    ap_bin_start = when.replace(
        hour=(when.hour // 3) * 3,
        minute=0,
        second=0,
        microsecond=0,
    )

    resource_provenance = MappingProxyType(dict(reference.provenance))
    provenance = MappingProxyType(
        {
            "schema": _MSIS_INPUT_SCHEMA,
            "selection_policy": ("previous_day_observed_f107_current_center81_utc_3h_ap_v1"),
            "bad_f107_policy": ("outside_0_400_substitute_same_day_center81_v1"),
            "requested_datetime_utc": when.isoformat(),
            "previous_day_f107_date": previous_f107_day.isoformat(),
            "center81_date": when.date().isoformat(),
            "ap_bin_start_utc": ap_bin_start.isoformat(),
            "previous_day_f107_data_type": previous_record.f107_data_type,
            "previous_day_f107_effective_data_type": (previous_f107_effective_data_type),
            "center81_f107a_data_type": center81_f107a_data_type,
            "previous_day_f107_substituted_by_center81": f107_substituted,
            "qualifier_warning_policy": _QUALIFIER_WARNING_POLICY,
            "qualifier_warning_emitted": qualifier_warning_emitted,
            "ap_storm_vector_sha256": sha256(canonical_ap.tobytes(order="C")).hexdigest(),
            "resource": resource_provenance,
        }
    )
    return MappingProxyType(
        {
            "f107_previous_day": float(f107_previous_day),
            "f107a_center81": float(current_record.f107a_observed_center81),
            "ap_storm_vector": ap_storm_vector,
            "provenance": provenance,
        }
    )


def canonical_msis_space_weather_identity(inputs: Mapping) -> dict:
    """Return the JSON-safe, validated identity of one MSIS input record."""

    try:
        f107 = float(inputs["f107_previous_day"])
        f107a = float(inputs["f107a_center81"])
        ap = np.asarray(inputs["ap_storm_vector"], dtype=np.float64)
        provenance = inputs["provenance"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid MSIS space-weather input mapping") from exc
    if (
        not np.isfinite(f107)
        or not 0.0 < f107 <= 400.0
        or not np.isfinite(f107a)
        or not 0.0 < f107a <= 400.0
    ):
        raise ValueError("MSIS F10.7 and F10.7a inputs must be in (0, 400]")
    if ap.shape != (7,) or not np.all(np.isfinite(ap)) or np.any(ap < 0.0):
        raise ValueError("MSIS storm-time Ap input must be seven finite non-negative values")
    if not isinstance(provenance, Mapping):
        raise ValueError("the MSIS space-weather record must be a table")
    if provenance.get("schema") != _MSIS_INPUT_SCHEMA:
        raise ValueError("the MSIS space-weather record has an unsupported format")
    for qualifier_field in (
        "previous_day_f107_data_type",
        "previous_day_f107_effective_data_type",
        "center81_f107a_data_type",
    ):
        if provenance.get(qualifier_field) not in {"OBS", "INT", "PRD"}:
            raise ValueError(f"the MSIS space-weather record has no valid {qualifier_field}")
    if provenance.get("qualifier_warning_policy") != _QUALIFIER_WARNING_POLICY:
        raise ValueError("the MSIS space-weather record has an unsupported warning rule")
    expected_qualifier_warning = provenance["previous_day_f107_effective_data_type"] in {
        "INT",
        "PRD",
    } or provenance["center81_f107a_data_type"] in {"INT", "PRD"}
    qualifier_warning_emitted = provenance.get("qualifier_warning_emitted")
    if (
        not isinstance(qualifier_warning_emitted, bool)
        or qualifier_warning_emitted != expected_qualifier_warning
    ):
        raise ValueError(
            "the MSIS space-weather record says a warning was given that its flux "
            "types do not call for, or the reverse"
        )
    resource = provenance.get("resource")
    if not isinstance(resource, Mapping):
        raise ValueError("the MSIS space-weather record does not name its CelesTrak table")
    source_sha256 = resource.get("source_sha256")
    if not isinstance(source_sha256, str) or _SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise ValueError("MSIS space-weather resource lacks a valid source_sha256")
    source_url = resource.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("MSIS space-weather resource lacks a valid source_url")

    canonical_provenance = dict(provenance)
    canonical_provenance["resource"] = dict(resource)
    return {
        "schema": _MSIS_INPUT_SCHEMA,
        "f107_previous_day": f107,
        "f107a_center81": f107a,
        "ap_storm_vector": [float(value) for value in ap],
        "provenance": canonical_provenance,
    }


@lru_cache(maxsize=4)
def _load_configured_reference(
    path: str,
    expected_sha256: str,
    source_url: str,
) -> CelestrakSpaceWeather:
    """The configured snapshot, loaded and hash-checked once per (path, hash, URL)."""
    return load_celestrak_space_weather(
        path,
        expected_sha256=expected_sha256,
        source_url=source_url,
    )


def configured_msis_space_weather_inputs(
    when_utc: datetime,
) -> Mapping[str, object]:
    """Return MSIS inputs from the configured local, hash-pinned resource."""

    from gtellcorr.data.atmosphere._context import atm

    reference = _load_configured_reference(
        str(atm.CELESTRAK_SPACE_WEATHER_PATH),
        str(atm.CELESTRAK_SPACE_WEATHER_SHA256),
        str(atm.CELESTRAK_SPACE_WEATHER_SOURCE_URL),
    )
    return get_msis_space_weather_inputs(reference, when_utc)
