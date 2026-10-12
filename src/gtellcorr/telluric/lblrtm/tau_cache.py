"""Per-molecule optical-depth cache backed by direct LBLRTM IEMIT=0 output.

The public surface is the cache contract consumed by the fitting engine.  A
complete 121-level atmosphere is mandatory; the caller, rather than this
module, owns construction of that atmosphere.

Cache construction runs LBLRTM once for each fitted molecule and once for the
non-fitted molecules. Public H2O values are direct profile multipliers. For
every non-water molecule, the public ppmv column is converted to a profile
multiplier by dividing it by the supplied profile value interpolated at the
observer altitude. All other
molecules are zero in a fitted-molecule run; fitted molecules are zero in the
fixed run.

Each component is read directly from the IEMIT=0 binary TAPE12 sight-line
optical-depth product.  Finite values, including finite negative values, are
retained unchanged; NaN or infinity rejects the complete build.  No
transmission floor, logarithm, or value clipping participates in construction.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from gtellcorr.data.observation.frames import vac_to_air

from .lblrtm_data_models import (
    LblrtmLineOfSight,
    LblrtmOpticalDepthSpectrum,
)
from .lblrtm_errors import LblrtmInputError
from .lblrtm_molecules import (
    LBLRTM_MOLECULES_BY_ID,
    normalize_lblrtm_molecule_name,
)
from .lblrtm_resources import ResolvedLblrtmResourceFiles
from .optical_depth_calculator import calculate_lblrtm_optical_depth
from .tape5_writer import validate_lblrtm_layer_input

_REFERENCE_COLUMNS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "h2o": 1.0,
        "co2": 422.8,
        "o3": 3.9e-2,
        "n2o": 0.3378,
        "co": 0.14,
        "ch4": 1.937,
        "o2": 2.1e5,
        "no": 1.1e-19,
        "so2": 1.0e-4,
        "no2": 1.0e-4,
        "nh3": 1.0e-4,
        "hno3": 5.6e-4,
    }
)

# A mutable dict and ordered tuple are retained for compatibility with callers
# of the established cache API.  Internal calculations use the immutable table
# above, so external mutation cannot change the scientific interpretation.
REF_COLUMN = dict(_REFERENCE_COLUMNS)
MOLECULES = tuple(_REFERENCE_COLUMNS)

_PROFILE_LEVEL_COUNT: Final[int] = 121
_CACHE_SCHEMA: Final[str] = "gtellcorr_tau_cache_v3"
_CACHE_KEY_DOMAIN: Final[bytes] = b"gtellcorr_tau_cache_key_v3\x00"
_REGRID_POLICY: Final[str] = "exact_iemit0_iod1_dvout_grid_required_v1"
_DIRECT_OPTICAL_DEPTH_POLICY: Final[str] = "binary_tape12_iemit0_raw_finite_no_clipping_v1"
_INPUT_WAVELENGTH_MEDIUM: Final[str] = "vacuum"
_MAXIMUM_CACHE_GRID_POINTS: Final[int] = 5_000_000
_MAXIMUM_METADATA_BYTES: Final[int] = 16 * 1024 * 1024
_SUPPORTED_WAVELENGTH_FORMULAS: Final[frozenset[str]] = frozenset(
    {"espresso", "edlen1966", "ciddor1996"}
)
_CIDDOR_WAVELENGTH_RANGE_NM: Final[tuple[float, float]] = (230.0, 1700.0)
_CALCULATOR_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "metadata_schema_version",
    "source_product",
    "spectral_output",
    "iemit",
    "iod",
    "optical_depth_grid_spacing_cm1",
    "optical_depth_policy",
    "continuum_mode",
    "expanded_input_wavelength_range_nm",
    "expanded_vacuum_wavelength_range_nm",
    "input_wavelength_medium",
    "line_of_sight",
    "molecule_relative_columns",
    "output_point_count",
    "output_grid_kind",
    "output_wavelength_medium",
    "output_wavelength_range_nm",
    "raw_values_modified",
    "requested_wavelength_range_nm",
    "resampling",
    "resource_fingerprint",
    "resource_line_margin_policy",
    "resource_safe_wavenumber_range_cm1",
    "segment_count",
    "segment_plan_sha256",
    "ownership_policy",
    "unconvolved",
    "wave_margin_nm",
    "wavelength_conversion_formula",
    "negative_optical_depth_count",
    "minimum_optical_depth",
    "maximum_optical_depth",
)
_CALCULATOR_SEGMENT_FIELDS: Final[tuple[str, ...]] = (
    "segment_index",
    "rendered_wavenumber_range_cm1",
    "returned_wavenumber_range_cm1",
    "returned_point_count",
    "owned_point_count",
    "tape5_sha256",
    "tape6_sha256",
    "tape12_sha256",
    "tape12_abi",
)
_CALCULATOR_SEAM_FIELDS: Final[tuple[str, ...]] = (
    "lower_segment_index",
    "upper_segment_index",
    "source_switch_wavenumber_cm1",
    "selected_seam_gap_cm1",
    "diagnostic_point_count",
    "rms_optical_depth_difference",
    "p99_absolute_optical_depth_difference",
    "maximum_absolute_optical_depth_difference",
)
_PROFILE_FIELDS: Final[tuple[str, ...]] = (
    "z_km",
    "P_hPa",
    "T_K",
    *tuple(molecule.profile_field for _, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())),
)
_PROFILE_MOLECULES: Final[tuple[str, ...]] = tuple(
    molecule.canonical_name for _, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())
)
_PROFILE_FIELDS_BY_MOLECULE: Final[Mapping[str, str]] = MappingProxyType(
    {
        molecule.canonical_name.lower(): molecule.profile_field
        for _, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())
    }
)

__all__ = [
    "MOLECULES",
    "REF_COLUMN",
    "TauCache",
    "compute_tau_cache_key",
]


def _coerce_finite_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    """*value* as a finite float (not a bool), optionally bounded below by
    *minimum* (strictly if *strict_minimum*)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a real number, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None:
        invalid = normalized <= minimum if strict_minimum else normalized < minimum
        if invalid:
            operator = ">" if strict_minimum else ">="
            raise ValueError(f"{field_name} must be {operator} {minimum}")
    return normalized


def _snapshot_profile(
    atmosphere_profile: Mapping[str, object] | None,
) -> Mapping[str, np.ndarray]:
    """A read-only copy of the complete 121-level profile, checked field by field."""
    if atmosphere_profile is None:
        raise ValueError(
            "atmosphere_profile is required; this cache never constructs or "
            "fills an atmospheric profile"
        )
    if not isinstance(atmosphere_profile, Mapping):
        raise TypeError("atmosphere_profile must be a mapping of level arrays")

    captured: dict[str, np.ndarray] = {}
    try:
        for field_name in _PROFILE_FIELDS:
            captured[field_name] = np.array(
                atmosphere_profile[field_name],
                dtype=object,
                copy=True,
            )
    except KeyError as exc:
        raise ValueError(f"atmosphere_profile is missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise ValueError("cannot snapshot atmosphere_profile") from exc

    captured_mapping: Mapping[str, object] = MappingProxyType(captured)
    try:
        validate_lblrtm_layer_input(captured_mapping, _PROFILE_MOLECULES)
    except LblrtmInputError as exc:
        raise ValueError(f"invalid atmosphere_profile: {exc}") from exc
    level_count = int(captured["z_km"].size)
    if level_count != _PROFILE_LEVEL_COUNT:
        raise ValueError(
            "atmosphere_profile must contain exactly "
            f"{_PROFILE_LEVEL_COUNT} levels; got {level_count}"
        )

    normalized: dict[str, np.ndarray] = {}
    try:
        for field_name in _PROFILE_FIELDS:
            array = np.array(captured[field_name], dtype=np.float64, copy=True)
            array.setflags(write=False)
            normalized[field_name] = array
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise ValueError("cannot normalize atmosphere_profile") from exc

    normalized_mapping: Mapping[str, np.ndarray] = MappingProxyType(normalized)
    try:
        validate_lblrtm_layer_input(normalized_mapping, _PROFILE_MOLECULES)
    except LblrtmInputError as exc:
        raise ValueError(f"invalid normalized atmosphere_profile: {exc}") from exc
    return normalized_mapping


def _profile_sha256(profile: Mapping[str, np.ndarray]) -> str:
    """SHA-256 of the profile arrays (field names, sizes and values)."""
    digest = sha256()
    for field_name in _PROFILE_FIELDS:
        array = np.asarray(profile[field_name], dtype=np.float64)
        encoded_name = field_name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(array.size.to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _column_scale_denominators(
    profile: Mapping[str, np.ndarray],
    observer_altitude_km: float,
) -> Mapping[str, float]:
    """Return the public-column scale denominators: 1 for H2O, otherwise each
    molecule's profile value at the observer altitude."""

    altitude = np.asarray(profile["z_km"], dtype=np.float64)
    denominators: dict[str, float] = {"h2o": 1.0}
    for molecule in MOLECULES:
        if molecule == "h2o":
            continue
        profile_field = _PROFILE_FIELDS_BY_MOLECULE[molecule]
        denominator = float(
            np.interp(
                observer_altitude_km,
                altitude,
                np.asarray(profile[profile_field], dtype=np.float64),
            )
        )
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"{profile_field} must be positive at observer altitude for public-column scaling"
            )
        denominators[molecule] = denominator
    return MappingProxyType(denominators)


def _array_sha256(array: np.ndarray) -> str:
    """SHA-256 of an array's shape and float64 values."""
    normalized = np.asarray(array, dtype=np.float64)
    digest = sha256()
    digest.update(normalized.ndim.to_bytes(4, "big"))
    for extent in normalized.shape:
        digest.update(int(extent).to_bytes(8, "big"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_molecule_collection(
    fit_molecules: Iterable[str],
) -> tuple[str, ...]:
    """The fitted molecules as a tuple of known, unique engine names."""
    if isinstance(fit_molecules, (str, bytes)):
        raise TypeError("fit_molecules must be an iterable, not a string")
    try:
        supplied = tuple(fit_molecules)
    except TypeError as exc:
        raise TypeError("fit_molecules must be an iterable of molecule names") from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for position, molecule in enumerate(supplied):
        try:
            public_name = normalize_lblrtm_molecule_name(molecule).lower()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid fit molecule at position {position}: {exc}") from exc
        if public_name not in _REFERENCE_COLUMNS:
            raise ValueError(f"unknown molecule {public_name!r}; valid keys are {MOLECULES}")
        if public_name in seen:
            raise ValueError(f"duplicate fit molecule after normalization: {public_name}")
        seen.add(public_name)
        normalized.append(public_name)
    return tuple(normalized)


def _normalize_fixed_columns(
    fixed: Mapping[str, object] | None,
) -> dict[str, float]:
    """The user-fixed columns of non-fitted molecules, checked."""
    if fixed is None:
        return {}
    if not isinstance(fixed, Mapping):
        raise TypeError("fixed must be a mapping or None")
    try:
        supplied = tuple(fixed.items())
    except RuntimeError as exc:
        raise ValueError("fixed changed while being read") from exc

    normalized: dict[str, float] = {}
    for supplied_name, supplied_value in supplied:
        try:
            public_name = normalize_lblrtm_molecule_name(supplied_name).lower()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid fixed molecule {supplied_name!r}: {exc}") from exc
        if public_name in normalized:
            raise ValueError(f"duplicate fixed molecule after normalization: {public_name}")
        normalized[public_name] = _coerce_finite_float(
            supplied_value,
            field_name=f"fixed column for {public_name}",
            minimum=0.0,
        )
    return normalized


def _cache_component_contract(
    fit_molecules: Iterable[str],
    fixed: Mapping[str, object] | None,
) -> tuple[tuple[str, ...], dict[str, float], dict[str, float]]:
    """``(fitted molecules, fixed columns of the others, reference columns of
    the fitted)``; unset fixed columns take their reference value."""
    molecules = _normalize_molecule_collection(fit_molecules)
    supplied_fixed = _normalize_fixed_columns(fixed)
    fixed_columns = {
        molecule: float(supplied_fixed.get(molecule, _REFERENCE_COLUMNS[molecule]))
        for molecule in MOLECULES
        if molecule not in molecules
    }
    references = {molecule: float(_REFERENCE_COLUMNS[molecule]) for molecule in molecules}
    return molecules, fixed_columns, references


def _normalize_sha256(value: object, *, field_name: str) -> str:
    """*value*, which must be 64 lowercase hexadecimal digits."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must contain 64 lowercase hexadecimal digits")
    return value


def _normalize_tape12_abi(value: object) -> tuple[dict[str, object], str]:
    """``(TAPE12 ABI record, precision)`` with the precision (``single`` /
    ``double``) inferred from the real and integer widths."""
    if not isinstance(value, Mapping):
        raise ValueError("optical-depth cache: its TAPE12 binary layout record is not a table")
    record = dict(value)
    expected_fields = {
        "byte_order",
        "record_marker_bytes",
        "real_bytes",
        "integer_bytes",
    }
    if set(record) != expected_fields:
        raise ValueError("optical-depth cache: its TAPE12 binary layout record has wrong fields")
    byte_order = record["byte_order"]
    if byte_order not in {"little", "big"}:
        raise ValueError("optical-depth cache: TAPE12 byte_order must be 'little' or 'big'")
    widths: dict[str, int] = {}
    for field_name in (
        "record_marker_bytes",
        "real_bytes",
        "integer_bytes",
    ):
        field_value = record[field_name]
        if isinstance(field_value, (bool, np.bool_)) or not isinstance(
            field_value,
            (int, np.integer),
        ):
            raise ValueError(f"optical-depth cache: TAPE12 {field_name} must be an integer")
        normalized = int(field_value)
        if normalized not in {4, 8}:
            raise ValueError(f"optical-depth cache: TAPE12 {field_name} must be 4 or 8")
        widths[field_name] = normalized
    precision_widths = (
        widths["real_bytes"],
        widths["integer_bytes"],
    )
    if precision_widths == (4, 4):
        precision = "single"
    elif precision_widths == (8, 8):
        precision = "double"
    else:
        raise ValueError("TAPE12 real/integer widths do not identify a supported LBLRTM precision")
    return (
        {
            "byte_order": byte_order,
            **widths,
        },
        precision,
    )


def _resource_provenance(
    resources: ResolvedLblrtmResourceFiles,
) -> Mapping[str, Any]:
    """The resource identity recorded in the cache (manifest fingerprint,
    precision, TAPE12 ABI, validation evidence) of fully validated resources."""
    if not isinstance(resources, ResolvedLblrtmResourceFiles):
        raise TypeError(
            "resources must be a ResolvedLblrtmResourceFiles instance returned "
            "by the complete bundle validator"
        )
    resources.require_fully_validated()
    fingerprint = _normalize_sha256(
        resources.manifest_fingerprint,
        field_name="resource manifest_fingerprint",
    )
    evidence = resources.validation_evidence
    if evidence is None:
        raise ValueError("internal error: the LBLRTM bundle was not checked before use")
    abi_contract = resources.require_validated_tape12_abi()
    abi, precision = _normalize_tape12_abi(
        {
            "byte_order": abi_contract.byte_order,
            "record_marker_bytes": abi_contract.record_marker_bytes,
            "real_bytes": abi_contract.real_bytes,
            "integer_bytes": abi_contract.integer_bytes,
        }
    )
    return MappingProxyType(
        {
            "manifest_fingerprint": fingerprint,
            "lblrtm_precision": precision,
            "tape12_abi": MappingProxyType(abi),
            "validation_evidence": MappingProxyType(dict(evidence)),
        }
    )


def _normalize_loaded_resource_provenance(value: object) -> Mapping[str, Any]:
    """The resource provenance of a loaded cache, checked field by field."""
    if not isinstance(value, Mapping):
        raise ValueError("optical-depth cache: its LBLRTM bundle record is not a table")
    record = dict(value)
    expected_fields = {
        "manifest_fingerprint",
        "lblrtm_precision",
        "tape12_abi",
        "validation_evidence",
    }
    if set(record) != expected_fields:
        raise ValueError("optical-depth cache: its LBLRTM bundle record has wrong fields")
    fingerprint = _normalize_sha256(
        record["manifest_fingerprint"],
        field_name="cache resource manifest_fingerprint",
    )
    abi, inferred_precision = _normalize_tape12_abi(record["tape12_abi"])
    if record["lblrtm_precision"] != inferred_precision:
        raise ValueError(
            "optical-depth cache: its LBLRTM precision contradicts its TAPE12 binary layout"
        )
    evidence = record["validation_evidence"]
    if not isinstance(evidence, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in evidence.items()
    ):
        raise ValueError("optical-depth cache: its LBLRTM bundle check record is invalid")
    normalized_evidence = dict(evidence)
    expected_evidence = {
        "manifest_fingerprint": fingerprint,
        "build_contract.tape12_abi.byte_order": str(abi["byte_order"]),
        "build_contract.tape12_abi.record_marker_bytes": str(abi["record_marker_bytes"]),
        "build_contract.tape12_abi.real_bytes": str(abi["real_bytes"]),
        "build_contract.tape12_abi.integer_bytes": str(abi["integer_bytes"]),
    }
    for field_name, expected_value in expected_evidence.items():
        if normalized_evidence.get(field_name) != expected_value:
            raise ValueError(
                f"optical-depth cache: its LBLRTM bundle check record contradicts {field_name}"
            )
    return MappingProxyType(
        {
            "manifest_fingerprint": fingerprint,
            "lblrtm_precision": inferred_precision,
            "tape12_abi": MappingProxyType(abi),
            "validation_evidence": MappingProxyType(normalized_evidence),
        }
    )


def _json_ready(value: object) -> object:
    """*value* as plain JSON types (finite floats only; NumPy scalars unwrapped)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("internal error: the cache build record holds a NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    raise TypeError(
        f"internal error: the cache build record holds a {type(value).__name__}, "
        "which JSON cannot store"
    )


def _freeze_json(value: object) -> object:
    """A read-only (MappingProxyType / tuple) copy of JSON-ready *value*."""
    ready = _json_ready(value)
    if isinstance(ready, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in ready.items()})
    if isinstance(ready, list):
        return tuple(_freeze_json(item) for item in ready)
    return ready


def _json_bytes(value: Mapping[str, object]) -> bytes:
    """Canonical, size-bounded JSON bytes of *value* (sorted keys, ASCII)."""
    try:
        encoded = json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cache metadata is not deterministic JSON") from exc
    if len(encoded) > _MAXIMUM_METADATA_BYTES:
        raise ValueError(f"cache metadata exceeds the {_MAXIMUM_METADATA_BYTES}-byte safety limit")
    return encoded


def compute_tau_cache_key(cache_metadata: Mapping[str, object]) -> str:
    """Return the schema-v3 key for one complete cache identity record."""

    if not isinstance(cache_metadata, Mapping):
        raise TypeError("cache_metadata must be a mapping")
    ready = _json_ready(cache_metadata)
    if not isinstance(ready, dict) or ready.get("schema") != _CACHE_SCHEMA:
        raise ValueError("optical-depth cache: unsupported cache key format")
    payload = _json_bytes(ready)
    digest = sha256()
    digest.update(_CACHE_KEY_DOMAIN)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _cache_key_metadata(
    *,
    resource_provenance: Mapping[str, object],
    profile_digest: str,
    window: Mapping[str, float],
    effective_airmass: float,
    observer_altitude_km: float,
    path_end_altitude_km: float,
    reference_latitude_deg: float,
    vac2air: bool,
    wavelength_conversion_formula: str,
    wave_margin_nm: float,
    continuum_mode: str,
    optical_depth_grid_spacing_cm1: float,
    fit_molecules: tuple[str, ...],
    reference_columns: Mapping[str, float],
    fixed_columns: Mapping[str, float],
    column_scale_denominators: Mapping[str, float],
) -> Mapping[str, object]:
    """Everything that identifies a cache: resources, profile digest, window,
    geometry, wavelength conversion, molecules and columns, construction policy."""
    resource = {
        "manifest_fingerprint": resource_provenance["manifest_fingerprint"],
        "lblrtm_precision": resource_provenance["lblrtm_precision"],
        "tape12_abi": resource_provenance["tape12_abi"],
    }
    record = {
        "schema": _CACHE_SCHEMA,
        "resource": resource,
        "profile": {
            "sha256": _normalize_sha256(
                profile_digest,
                field_name="profile_sha256",
            ),
            "level_count": _PROFILE_LEVEL_COUNT,
        },
        "spectral_window_cm1": {
            "lowfreq": float(window["lowfreq"]),
            "highfreq": float(window["highfreq"]),
        },
        "line_of_sight": {
            "effective_airmass": float(effective_airmass),
            "observer_altitude_km": float(observer_altitude_km),
            "path_end_altitude_km": float(path_end_altitude_km),
            "reference_latitude_deg": float(reference_latitude_deg),
        },
        "wavelength": {
            "input_medium": _INPUT_WAVELENGTH_MEDIUM,
            "output_medium": "air" if vac2air else "vacuum",
            "conversion_formula": wavelength_conversion_formula,
            "wave_margin_nm": float(wave_margin_nm),
        },
        "continuum_mode": continuum_mode,
        "optical_depth_output": {
            "spectral_output": "optical_depth",
            "iemit": 0,
            "iod": 1,
            "grid_spacing_cm1": float(optical_depth_grid_spacing_cm1),
            "policy": _DIRECT_OPTICAL_DEPTH_POLICY,
            "raw_values_modified": False,
        },
        "fit": {
            "molecules": list(fit_molecules),
            "reference_columns": {
                molecule: float(reference_columns[molecule]) for molecule in fit_molecules
            },
        },
        "fixed_columns": {molecule: float(column) for molecule, column in fixed_columns.items()},
        "fixed_reference_columns": {
            molecule: float(_REFERENCE_COLUMNS[molecule]) for molecule in fixed_columns
        },
        "public_column_scaling": {
            "policy": (
                "h2o_direct_profile_scale; nonwater_target_ppmv_divided_by_"
                "supplied_profile_ppmv_at_observer_altitude"
            ),
            "denominators": {
                molecule: float(column_scale_denominators[molecule]) for molecule in MOLECULES
            },
        },
        "optical_depth_value_policy": {
            "nonfinite": "reject_complete_build",
            "finite_negative": "retain_unchanged",
            "clipping": "none",
            "transmission_floor": "none",
            "logarithm": "none",
        },
        "regrid_policy": _REGRID_POLICY,
        "first_run_defines_grid": True,
        "regrid_extrapolation": "forbidden",
    }
    return _freeze_json(record)


def _require_expected_cache_identity(
    *,
    recorded_cache_key: object,
    recorded_metadata: object,
    expected_cache_key: object,
    expected_metadata: Mapping[str, object] | None,
) -> tuple[str, Mapping[str, object]]:
    """The recorded ``(cache_key, metadata)`` of a loaded cache, which must
    equal the expected ones when given."""
    cache_key = _normalize_sha256(
        recorded_cache_key,
        field_name="recorded cache_key",
    )
    if not isinstance(recorded_metadata, Mapping):
        raise ValueError("recorded cache key metadata must be a mapping")
    normalized_metadata = _freeze_json(recorded_metadata)
    recomputed_key = compute_tau_cache_key(normalized_metadata)
    if recomputed_key != cache_key:
        raise ValueError("recorded cache key does not match its metadata")

    if expected_cache_key is None and expected_metadata is None:
        raise ValueError("loading a cache requires expected_cache_key or expected_metadata")
    if expected_cache_key is not None:
        normalized_expected_key = _normalize_sha256(
            expected_cache_key,
            field_name="expected_cache_key",
        )
        if normalized_expected_key != cache_key:
            raise ValueError("cache key does not match expected_cache_key")
    if expected_metadata is not None:
        if not isinstance(expected_metadata, Mapping):
            raise TypeError("expected_metadata must be a mapping or None")
        normalized_expected_metadata = _freeze_json(expected_metadata)
        if _json_ready(normalized_expected_metadata) != _json_ready(normalized_metadata):
            raise ValueError("cache key metadata does not match expected_metadata")
        if compute_tau_cache_key(normalized_expected_metadata) != cache_key:
            raise ValueError("expected_metadata does not produce the recorded cache key")
    return cache_key, normalized_metadata


def _calculator_metadata_record(metadata: Mapping[str, Any]) -> dict[str, object]:
    """Retain only path-free calculator evidence used by cache auditing."""

    if not isinstance(metadata, Mapping):
        raise TypeError("calculator metadata must be a mapping")
    record = {
        field_name: metadata[field_name]
        for field_name in _CALCULATOR_METADATA_FIELDS
        if field_name in metadata
    }
    raw_segments = metadata.get("segments")
    if raw_segments is not None:
        if not isinstance(raw_segments, (list, tuple)):
            raise TypeError("calculator segment metadata must be a sequence")
        segments: list[dict[str, object]] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, Mapping):
                raise TypeError("calculator segment metadata must be mappings")
            segments.append(
                {
                    field_name: raw_segment[field_name]
                    for field_name in _CALCULATOR_SEGMENT_FIELDS
                    if field_name in raw_segment
                }
            )
        record["segments"] = segments
    raw_seams = metadata.get("seams")
    if raw_seams is not None:
        if not isinstance(raw_seams, (list, tuple)):
            raise TypeError("calculator seam metadata must be a sequence")
        seams: list[dict[str, object]] = []
        for raw_seam in raw_seams:
            if not isinstance(raw_seam, Mapping):
                raise TypeError("calculator seam metadata must be mappings")
            seams.append(
                {
                    field_name: raw_seam[field_name]
                    for field_name in _CALCULATOR_SEAM_FIELDS
                    if field_name in raw_seam
                }
            )
        record["seams"] = seams
    return record


def _convert_result_grid(
    result: LblrtmOpticalDepthSpectrum,
    *,
    vac2air: bool,
    wavelength_conversion_formula: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelength_nm, optical_depth)`` of an LBLRTM run, converted to air
    with *wavelength_conversion_formula* if *vac2air*; checked finite and increasing."""
    if not isinstance(result, LblrtmOpticalDepthSpectrum):
        raise TypeError("calculate_lblrtm_optical_depth returned an unexpected result type")
    if result.wavelength_medium != "vacuum":
        raise ValueError("LBLRTM cache runs must return vacuum wavelengths")

    source_wavelength = np.asarray(result.wavelength_nm, dtype=np.float64)
    if vac2air:
        try:
            wavelength = np.asarray(
                vac_to_air(
                    source_wavelength,
                    formula=wavelength_conversion_formula,
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            raise ValueError("cannot convert the cache wavelength grid to air") from exc
    else:
        wavelength = np.array(source_wavelength, dtype=np.float64, copy=True)
    optical_depth = np.array(
        result.optical_depth,
        dtype=np.float64,
        copy=True,
    )

    if (
        wavelength.ndim != 1
        or optical_depth.ndim != 1
        or wavelength.shape != optical_depth.shape
        or wavelength.size < 2
        or wavelength.size > _MAXIMUM_CACHE_GRID_POINTS
    ):
        raise ValueError("LBLRTM cache run returned invalid array shapes")
    if not np.all(np.isfinite(optical_depth)):
        raise ValueError(
            "LBLRTM optical depth contains non-finite values; the complete "
            "cache build is rejected before any regridding"
        )
    if (
        not np.all(np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError(
            "LBLRTM cache wavelength grid must be finite, positive, and strictly increasing"
        )
    return wavelength, optical_depth


def _full_cell_edges(wavelength: np.ndarray) -> np.ndarray:
    """Return midpoint cells for one strictly increasing centre grid."""

    grid = np.asarray(wavelength, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("full-cell construction requires a finite strictly increasing grid")
    edges = np.empty(grid.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return edges


def _common_full_cell_slice(
    cache_wavelength: np.ndarray,
    *,
    coverage_lower_nm: float,
    coverage_upper_nm: float,
) -> slice:
    """Select contiguous first-run cells wholly inside common source support."""

    cache_grid = np.asarray(cache_wavelength, dtype=np.float64)
    if (
        cache_grid.ndim != 1
        or cache_grid.size < 2
        or not np.all(np.isfinite(cache_grid))
        or np.any(np.diff(cache_grid) <= 0.0)
        or not math.isfinite(coverage_lower_nm)
        or not math.isfinite(coverage_upper_nm)
        or not coverage_upper_nm > coverage_lower_nm
    ):
        raise ValueError("common full-cell coverage inputs are invalid")

    tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(float(cache_grid[0])),
            abs(float(cache_grid[-1])),
            abs(float(coverage_lower_nm)),
            abs(float(coverage_upper_nm)),
        )
    )
    start = int(np.searchsorted(cache_grid, coverage_lower_nm, side="left"))
    stop = int(np.searchsorted(cache_grid, coverage_upper_nm, side="right"))
    while stop - start >= 2:
        left_edge = float(cache_grid[start] - 0.5 * (cache_grid[start + 1] - cache_grid[start]))
        right_edge = float(
            cache_grid[stop - 1] + 0.5 * (cache_grid[stop - 1] - cache_grid[stop - 2])
        )
        if left_edge < coverage_lower_nm - tolerance:
            start += 1
            continue
        if right_edge > coverage_upper_nm + tolerance:
            stop -= 1
            continue
        return slice(start, stop)
    raise ValueError(
        "LBLRTM grids have no common first-run interval containing at least "
        "two complete cache cells; extrapolation is forbidden"
    )


def _conservative_resample(
    source_wavelength: np.ndarray,
    source_values: np.ndarray,
    target_wavelength: np.ndarray,
) -> np.ndarray:
    """Rebin centre-sampled values by integrating full target cells."""

    source_x = np.asarray(source_wavelength, dtype=np.float64)
    source_y = np.asarray(source_values, dtype=np.float64)
    target_x = np.asarray(target_wavelength, dtype=np.float64)
    if (
        source_x.ndim != 1
        or source_y.ndim != 1
        or target_x.ndim != 1
        or source_x.shape != source_y.shape
        or source_x.size < 2
        or target_x.size < 2
    ):
        raise ValueError("conservative resampling requires valid one-dimensional grids")
    if (
        not np.all(np.isfinite(source_x))
        or not np.all(np.isfinite(target_x))
        or np.any(np.diff(source_x) <= 0.0)
        or np.any(np.diff(target_x) <= 0.0)
    ):
        raise ValueError("conservative resampling grids must be strictly increasing")

    edges = _full_cell_edges(target_x)
    tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(float(source_x[0])),
            abs(float(source_x[-1])),
            abs(float(edges[0])),
            abs(float(edges[-1])),
        )
    )
    if edges[0] < source_x[0] - tolerance or edges[-1] > source_x[-1] + tolerance:
        raise ValueError(
            "a later LBLRTM grid does not cover the first-run cache grid; "
            "extrapolation is forbidden"
        )

    rebinned = np.empty(target_x.shape, dtype=np.float64)
    for index in range(target_x.size):
        left = float(edges[index])
        right = float(edges[index + 1])
        if not right > left:
            raise ValueError("target resampling cells must have positive width")
        first = int(np.searchsorted(source_x, left, side="right"))
        stop = int(np.searchsorted(source_x, right, side="left"))
        interior_x = source_x[first:stop]
        interior_y = source_y[first:stop]
        nodes_x = np.concatenate(([left], interior_x, [right]))
        nodes_y = np.concatenate(
            (
                [np.interp(left, source_x, source_y)],
                interior_y,
                [np.interp(right, source_x, source_y)],
            )
        )
        with np.errstate(invalid="ignore", over="ignore"):
            integral = np.sum(0.5 * (nodes_y[:-1] + nodes_y[1:]) * np.diff(nodes_x))
            rebinned[index] = integral / (right - left)
    return rebinned


def _read_float_array(
    archive,
    key: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Array *key* of the cache archive as float64 (optionally of *expected_size*)."""
    try:
        array = np.array(archive[key], dtype=np.float64, copy=True)
    except KeyError as exc:
        raise ValueError(f"cache archive is missing array {key!r}") from exc
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise ValueError(f"cache array {key!r} is invalid") from exc
    if array.ndim != 1:
        raise ValueError(f"cache array {key!r} must be one-dimensional")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"cache array {key!r} has {array.size} values; expected {expected_size}")
    array.setflags(write=False)
    return array


def _cache_path(path: os.PathLike[str] | str) -> Path:
    """*path* with a ``.npz`` suffix."""
    try:
        output = Path(path)
    except (TypeError, ValueError) as exc:
        raise TypeError("cache path must be path-like") from exc
    if not str(output).endswith(".npz"):
        output = Path(str(output) + ".npz")
    return output


def _cache_lock_path(cache_path: Path) -> Path:
    """Lock file of a cache file (hidden sibling ``.<name>.lock``)."""
    return cache_path.with_name(f".{cache_path.name}.lock")


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry to disk (after an atomic rename)."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_loaded_build_provenance(
    value: object,
    *,
    cache_wavegrid: np.ndarray,
    fit_molecules: tuple[str, ...],
    fixed_columns: Mapping[str, float],
    column_scale_denominators: Mapping[str, float],
    optical_depth_grid_spacing_cm1: float,
    profile_digest: str,
    resource_fingerprint: str,
) -> object:
    """Check the build provenance of a loaded cache against its grid,
    molecules, columns and construction settings."""
    if not isinstance(value, Mapping):
        raise ValueError("optical-depth cache: its build record is not a table")
    record = dict(value)
    expected_fields = {
        "schema",
        "run_count",
        "fit_molecules",
        "fixed_columns",
        "spectral_output",
        "iemit",
        "iod",
        "optical_depth_grid_spacing_cm1",
        "nonfinite_optical_depth_policy",
        "finite_negative_optical_depth_policy",
        "optical_depth_policy",
        "first_run_defines_grid",
        "different_grid_policy",
        "grid_difference_observed",
        "common_full_cell_coverage_nm",
        "final_cache_grid_range_nm",
        "final_cache_cell_coverage_nm",
        "profile_sha256",
        "resource_fingerprint",
        "runs",
    }
    if set(record) != expected_fields:
        raise ValueError("optical-depth cache: its build record has wrong fields")
    try:
        recorded_fit_molecules = _normalize_molecule_collection(record["fit_molecules"])
        recorded_fixed_columns = _normalize_fixed_columns(record["fixed_columns"])
        run_count = int(record["run_count"])
        runs = tuple(record["runs"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("optical-depth cache: its build record is malformed") from exc
    if (
        record["schema"] != _CACHE_SCHEMA
        or recorded_fit_molecules != fit_molecules
        or recorded_fixed_columns != dict(fixed_columns)
        or run_count != len(fit_molecules) + 1
        or len(runs) != run_count
        or record["spectral_output"] != "optical_depth"
        or record["iemit"] != 0
        or record["iod"] != 1
        or _coerce_finite_float(
            record["optical_depth_grid_spacing_cm1"],
            field_name="saved optical_depth_grid_spacing_cm1",
            minimum=0.0,
            strict_minimum=True,
        )
        != optical_depth_grid_spacing_cm1
        or record["nonfinite_optical_depth_policy"] != "reject_complete_build"
        or record["finite_negative_optical_depth_policy"] != "retain_unchanged"
        or record["optical_depth_policy"] != _DIRECT_OPTICAL_DEPTH_POLICY
        or record["first_run_defines_grid"] is not True
        or record["different_grid_policy"] != _REGRID_POLICY
        or record["grid_difference_observed"] is not False
        or record["profile_sha256"] != profile_digest
        or record["resource_fingerprint"] != resource_fingerprint
    ):
        raise ValueError("optical-depth cache: its build record contradicts its metadata")

    cache_grid = np.asarray(cache_wavegrid, dtype=np.float64)
    if (
        cache_grid.ndim != 1
        or cache_grid.size < 2
        or not np.all(np.isfinite(cache_grid))
        or np.any(np.diff(cache_grid) <= 0.0)
    ):
        raise ValueError("optical-depth cache: invalid wavenumber grid")
    if not isinstance(record["grid_difference_observed"], bool):
        raise ValueError("optical-depth cache: its build record has an invalid grid flag")

    def wavelength_range(field_name: str) -> tuple[float, float]:
        """``(lower, upper)`` of a recorded wavelength range, checked."""
        try:
            values = tuple(record[field_name])
            lower = float(values[0])
            upper = float(values[1])
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(f"optical-depth cache: build record {field_name} is invalid") from exc
        if (
            len(values) != 2
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or not upper > lower
        ):
            raise ValueError(f"optical-depth cache: build record {field_name} is invalid")
        return lower, upper

    common_coverage = wavelength_range("common_full_cell_coverage_nm")
    final_grid_range = wavelength_range("final_cache_grid_range_nm")
    final_cell_coverage = wavelength_range("final_cache_cell_coverage_nm")
    calculated_cell_edges = _full_cell_edges(cache_grid)
    if final_grid_range != (
        float(cache_grid[0]),
        float(cache_grid[-1]),
    ) or final_cell_coverage != (
        float(calculated_cell_edges[0]),
        float(calculated_cell_edges[-1]),
    ):
        raise ValueError("optical-depth cache: the grid range differs from its build record")
    coverage_tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            *(abs(value) for value in (*common_coverage, *final_cell_coverage)),
        )
    )
    if (
        final_cell_coverage[0] < common_coverage[0] - coverage_tolerance
        or final_cell_coverage[1] > common_coverage[1] + coverage_tolerance
    ):
        raise ValueError(
            "optical-depth cache: the grid extends beyond the range all molecules cover"
        )

    zero_columns = {molecule: 0.0 for molecule in MOLECULES}
    expected_run_fields = {
        "role",
        "molecule",
        "relative_columns",
        "source_grid_size",
        "cache_grid_size_before",
        "cache_grid_size_after",
        "final_cache_grid_size",
        "edge_cache_points_removed_left",
        "edge_cache_points_removed_right",
        "conservative_resampling_applied",
        "calculator_metadata",
    }
    prior_cache_grid_size: int | None = None
    total_removed = 0
    any_resampling = False
    for run_index, raw_run in enumerate(runs):
        if not isinstance(raw_run, Mapping):
            raise ValueError("optical-depth cache: its LBLRTM run records are not tables")
        run = dict(raw_run)
        if set(run) != expected_run_fields:
            raise ValueError("optical-depth cache: its LBLRTM run records have wrong fields")
        if run_index < len(fit_molecules):
            expected_role = "fit_molecule"
            expected_molecule: str | None = fit_molecules[run_index]
            expected_columns = dict(zero_columns)
            expected_columns[expected_molecule] = (
                _REFERENCE_COLUMNS[expected_molecule] / column_scale_denominators[expected_molecule]
            )
        else:
            expected_role = "fixed_molecules"
            expected_molecule = None
            expected_columns = dict(zero_columns)
            for molecule, column in fixed_columns.items():
                expected_columns[molecule] = float(column) / column_scale_denominators[molecule]
        try:
            recorded_columns = _normalize_fixed_columns(run["relative_columns"])
            source_grid_size = int(run["source_grid_size"])
            cache_grid_size_before = int(run["cache_grid_size_before"])
            cache_grid_size_after = int(run["cache_grid_size_after"])
            final_cache_grid_size = int(run["final_cache_grid_size"])
            removed_left = int(run["edge_cache_points_removed_left"])
            removed_right = int(run["edge_cache_points_removed_right"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("optical-depth cache: its LBLRTM run records are malformed") from exc
        integer_fields = (
            run["source_grid_size"],
            run["cache_grid_size_before"],
            run["cache_grid_size_after"],
            run["final_cache_grid_size"],
            run["edge_cache_points_removed_left"],
            run["edge_cache_points_removed_right"],
        )
        resampled = run["conservative_resampling_applied"]
        if (
            run["role"] != expected_role
            or run["molecule"] != expected_molecule
            or recorded_columns != expected_columns
            or any(isinstance(item, bool) for item in integer_fields)
            or not 2 <= source_grid_size <= _MAXIMUM_CACHE_GRID_POINTS
            or not 2 <= cache_grid_size_before <= _MAXIMUM_CACHE_GRID_POINTS
            or not 2 <= cache_grid_size_after <= cache_grid_size_before
            or final_cache_grid_size != cache_grid.size
            or final_cache_grid_size > cache_grid_size_after
            or removed_left < 0
            or removed_right < 0
            or cache_grid_size_after != cache_grid_size_before - removed_left - removed_right
            or not isinstance(resampled, bool)
            or (not resampled and (removed_left != 0 or removed_right != 0))
            or (
                prior_cache_grid_size is not None
                and cache_grid_size_before != prior_cache_grid_size
            )
        ):
            raise ValueError("optical-depth cache: its LBLRTM run records contradict its metadata")
        prior_cache_grid_size = cache_grid_size_after
        total_removed += removed_left + removed_right
        any_resampling = any_resampling or resampled
        calculator_metadata = run["calculator_metadata"]
        if not isinstance(calculator_metadata, Mapping):
            raise ValueError("optical-depth cache: its calculation record is not a table")
        normalized_calculator_metadata = _calculator_metadata_record(calculator_metadata)
        if _json_ready(calculator_metadata) != _json_ready(normalized_calculator_metadata):
            raise ValueError("optical-depth cache: its calculation record has unknown fields")
    if (
        any_resampling is not record["grid_difference_observed"]
        or prior_cache_grid_size != cache_grid.size
        or total_removed != int(runs[0]["cache_grid_size_after"]) - int(cache_grid.size)
    ):
        raise ValueError("optical-depth cache: its grid history is inconsistent")
    return _freeze_json(record)


class TauCache:
    """Cache per-molecule optical depth for one immutable atmosphere and path.

    ``pressure``, ``temperature``, and ``lat`` are retained for public-contract
    compatibility and provenance.  They do not modify the supplied profile.
    The complete 121-level profile is the sole pressure, temperature, and
    molecular state passed to LBLRTM.
    """

    def __init__(
        self,
        *,
        lowfreq,
        highfreq,
        pressure=795.0,
        temperature=283.0,
        angle=45.0,
        airmass=None,
        lat=30.6,
        alt=2.1,
        vac2air=True,
        atmosphere_profile,
        resources,
        wavelength_conversion_formula="espresso",
        wave_margin_nm=0.0,
        continuum_mode="none",
        optical_depth_grid_spacing_cm1=0.002,
        timeout_s=None,
        verbose=False,
    ):
        low = _coerce_finite_float(
            lowfreq,
            field_name="lowfreq",
            minimum=0.0,
            strict_minimum=True,
        )
        high = _coerce_finite_float(
            highfreq,
            field_name="highfreq",
            minimum=0.0,
            strict_minimum=True,
        )
        if high <= low:
            raise ValueError("highfreq must be greater than lowfreq")
        self.window = {"lowfreq": low, "highfreq": high}

        normalized_pressure = _coerce_finite_float(
            pressure,
            field_name="pressure",
            minimum=0.0,
            strict_minimum=True,
        )
        normalized_temperature = _coerce_finite_float(
            temperature,
            field_name="temperature",
            minimum=0.0,
            strict_minimum=True,
        )
        normalized_angle = _coerce_finite_float(
            angle,
            field_name="angle",
            minimum=0.0,
        )
        if normalized_angle >= 90.0:
            raise ValueError("angle must be less than 90 degrees")
        normalized_latitude = _coerce_finite_float(lat, field_name="lat")
        if not -90.0 <= normalized_latitude <= 90.0:
            raise ValueError("lat must lie in [-90, 90] degrees")
        normalized_altitude = _coerce_finite_float(alt, field_name="alt")

        if airmass is None:
            cosine = math.cos(math.radians(normalized_angle))
            if not cosine > 0.0:
                raise ValueError("angle does not define a finite upward-looking airmass")
            effective_airmass = 1.0 / cosine
            input_airmass = None
        else:
            effective_airmass = _coerce_finite_float(
                airmass,
                field_name="airmass",
                minimum=1.0,
            )
            input_airmass = effective_airmass

        if not isinstance(vac2air, (bool, np.bool_)):
            raise TypeError("vac2air must be bool")
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be bool")
        if not isinstance(wavelength_conversion_formula, str):
            raise TypeError("wavelength_conversion_formula must be a string")
        formula = wavelength_conversion_formula.strip().lower()
        if formula not in _SUPPORTED_WAVELENGTH_FORMULAS:
            raise ValueError(
                "wavelength_conversion_formula must be one of "
                f"{sorted(_SUPPORTED_WAVELENGTH_FORMULAS)}"
            )
        margin = _coerce_finite_float(
            wave_margin_nm,
            field_name="wave_margin_nm",
            minimum=0.0,
        )
        expanded_vacuum_range_nm = (
            1.0e7 / high - margin,
            1.0e7 / low + margin,
        )
        if expanded_vacuum_range_nm[0] <= 0.0:
            raise ValueError("wave_margin_nm produces a non-positive vacuum wavelength")
        if bool(vac2air) and formula == "ciddor1996":
            ciddor_lower, ciddor_upper = _CIDDOR_WAVELENGTH_RANGE_NM
            if (
                expanded_vacuum_range_nm[0] < ciddor_lower
                or expanded_vacuum_range_nm[1] > ciddor_upper
            ):
                raise ValueError(
                    "ciddor1996 conversion requires the complete padded "
                    f"vacuum wavelength interval to lie in "
                    f"[{ciddor_lower}, {ciddor_upper}] nm"
                )
        if continuum_mode != "none":
            raise ValueError("the optical-depth cache supports only continuum_mode='none'")
        optical_depth_spacing = _coerce_finite_float(
            optical_depth_grid_spacing_cm1,
            field_name="optical_depth_grid_spacing_cm1",
            minimum=0.0,
            strict_minimum=True,
        )
        if timeout_s is None:
            normalized_timeout = None
        else:
            normalized_timeout = _coerce_finite_float(
                timeout_s,
                field_name="timeout_s",
                minimum=0.0,
                strict_minimum=True,
            )

        profile = _snapshot_profile(atmosphere_profile)
        profile_lower = float(profile["z_km"][0])
        profile_upper = float(profile["z_km"][-1])
        if not profile_lower <= normalized_altitude < profile_upper:
            raise ValueError(
                "alt must lie within the atmosphere_profile altitude range and "
                "below its final level"
            )
        column_scale_denominators = _column_scale_denominators(
            profile,
            normalized_altitude,
        )

        resource_record = _resource_provenance(resources)
        self.resources: ResolvedLblrtmResourceFiles | None = resources
        self.resource_provenance = resource_record
        self.atmosphere_profile = profile
        self.column_scale_denominators = column_scale_denominators
        self.atmosphere = {
            "pressure": normalized_pressure,
            "temperature": normalized_temperature,
            "lat": normalized_latitude,
            "alt": normalized_altitude,
        }
        if input_airmass is None:
            self.atmosphere["angle"] = normalized_angle
        else:
            self.atmosphere["airmass"] = input_airmass
        self.angle = normalized_angle
        self.effective_airmass = effective_airmass
        self.vac2air = bool(vac2air)
        self.wavelength_conversion_formula = formula
        self.wave_margin_nm = margin
        self.continuum_mode = continuum_mode
        self.optical_depth_grid_spacing_cm1 = optical_depth_spacing
        self.timeout_s = normalized_timeout
        self.verbose = bool(verbose)

        self.wavegrid: np.ndarray | None = None
        self.fit_molecules: tuple[str, ...] = ()
        self.ref: dict[str, float] = {}
        self.fixed: dict[str, float] = {}
        self.tau_unit: dict[str, np.ndarray] = {}
        self.tau_fixed: np.ndarray | None = None
        self.build_provenance: Mapping[str, Any] | None = None
        self.cache_metadata: Mapping[str, Any] | None = None
        self.cache_key: str | None = None

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Resource, build and cache-key provenance of this cache."""
        return MappingProxyType(
            {
                "resource": self.resource_provenance,
                "build": self.build_provenance,
                "cache_metadata": self.cache_metadata,
                "cache_key": self.cache_key,
            }
        )

    def expected_cache_identity(
        self,
        fit_molecules=("h2o",),
        fixed=None,
    ) -> tuple[str, Mapping[str, Any]]:
        """Return the caller-owned key and metadata without running LBLRTM."""

        molecules, fixed_columns, references = _cache_component_contract(
            fit_molecules,
            fixed,
        )
        cache_metadata = _cache_key_metadata(
            resource_provenance=self.resource_provenance,
            profile_digest=_profile_sha256(self.atmosphere_profile),
            window=self.window,
            effective_airmass=self.effective_airmass,
            observer_altitude_km=self.atmosphere["alt"],
            path_end_altitude_km=float(self.atmosphere_profile["z_km"][-1]),
            reference_latitude_deg=self.atmosphere["lat"],
            vac2air=self.vac2air,
            wavelength_conversion_formula=self.wavelength_conversion_formula,
            wave_margin_nm=self.wave_margin_nm,
            continuum_mode=self.continuum_mode,
            optical_depth_grid_spacing_cm1=(self.optical_depth_grid_spacing_cm1),
            fit_molecules=molecules,
            reference_columns=references,
            fixed_columns=fixed_columns,
            column_scale_denominators=self.column_scale_denominators,
        )
        return compute_tau_cache_key(cache_metadata), cache_metadata

    def _run_lblrtm(
        self,
        relative_columns: Mapping[str, float],
    ) -> LblrtmOpticalDepthSpectrum:
        """One IEMIT=0 LBLRTM run with *relative_columns*, on the resources this
        cache was keyed with."""
        if self.resources is None:
            raise RuntimeError(
                "this loaded cache has no explicitly validated resource object; "
                "it can be evaluated but cannot be rebuilt"
            )
        self.resources.require_fully_validated()
        expected_fingerprint = self.resource_provenance["manifest_fingerprint"]
        if self.resources.manifest_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "internal error: the LBLRTM bundle changed before the cache was built"
            )

        vacuum_wavelength_range_nm = (
            1.0e7 / self.window["highfreq"],
            1.0e7 / self.window["lowfreq"],
        )
        line_of_sight = LblrtmLineOfSight(
            airmass=self.effective_airmass,
            observer_altitude_km=self.atmosphere["alt"],
            path_end_altitude_km=float(self.atmosphere_profile["z_km"][-1]),
            reference_latitude_deg=self.atmosphere["lat"],
        )
        result = calculate_lblrtm_optical_depth(
            full_profile=self.atmosphere_profile,
            wavelength_range_nm=vacuum_wavelength_range_nm,
            wavelength_medium="vacuum",
            wavelength_conversion_formula=self.wavelength_conversion_formula,
            line_of_sight=line_of_sight,
            molecule_relative_columns=relative_columns,
            resources=self.resources,
            wave_margin_nm=self.wave_margin_nm,
            continuum_mode=self.continuum_mode,
            optical_depth_grid_spacing_cm1=(self.optical_depth_grid_spacing_cm1),
            timeout_s=self.timeout_s,
            verbose=self.verbose,
        )
        if not isinstance(result, LblrtmOpticalDepthSpectrum):
            raise TypeError("LBLRTM IEMIT=0 path did not return optical-depth data")
        return result

    def build(self, fit_molecules=("h2o",), fixed=None):
        """Build ``len(fit_molecules) + 1`` optical-depth components."""

        expected_cache_key, expected_cache_metadata = self.expected_cache_identity(
            fit_molecules, fixed
        )
        molecules = tuple(expected_cache_metadata["fit"]["molecules"])
        fixed_columns = dict(expected_cache_metadata["fixed_columns"])
        references = dict(expected_cache_metadata["fit"]["reference_columns"])
        zero_relative_columns = {molecule: 0.0 for molecule in MOLECULES}

        cache_grid: np.ndarray | None = None
        common_coverage_lower_nm: float | None = None
        common_coverage_upper_nm: float | None = None
        grid_difference_observed = False
        unit_depths: dict[str, np.ndarray] = {}
        run_records: list[dict[str, object]] = []

        def normalize_run(
            result: LblrtmOpticalDepthSpectrum,
            *,
            role: str,
            molecule: str | None,
            relative_columns: Mapping[str, float],
        ) -> np.ndarray:
            """Put one run on the common cache grid (trimming edge cells, resampling
            conservatively if the grids differ), record it, and return its optical depth."""
            nonlocal cache_grid
            nonlocal common_coverage_lower_nm
            nonlocal common_coverage_upper_nm
            nonlocal grid_difference_observed
            run_grid, optical_depth = _convert_result_grid(
                result,
                vac2air=self.vac2air,
                wavelength_conversion_formula=self.wavelength_conversion_formula,
            )
            source_grid_size = int(run_grid.size)
            cache_grid_size_before = source_grid_size
            removed_left = 0
            removed_right = 0
            resampled = False
            if cache_grid is None:
                cache_grid = np.array(run_grid, dtype=np.float64, copy=True)
                cache_grid.setflags(write=False)
                first_cell_edges = _full_cell_edges(cache_grid)
                common_coverage_lower_nm = float(first_cell_edges[0])
                common_coverage_upper_nm = float(first_cell_edges[-1])
            elif not np.array_equal(run_grid, cache_grid):
                grid_difference_observed = True
                raise ValueError(
                    "IEMIT=0 TauCache component grids differ despite the "
                    "shared IOD=1 DVOUT control; refusing to regrid direct "
                    "optical depth"
                )
            elif optical_depth.shape != cache_grid.shape:
                raise ValueError("LBLRTM cache run grid shape changed unexpectedly")

            run_records.append(
                {
                    "role": role,
                    "molecule": molecule,
                    "relative_columns": dict(relative_columns),
                    "source_grid_size": source_grid_size,
                    "cache_grid_size_before": cache_grid_size_before,
                    "cache_grid_size_after": int(cache_grid.size),
                    "edge_cache_points_removed_left": removed_left,
                    "edge_cache_points_removed_right": removed_right,
                    "conservative_resampling_applied": resampled,
                    "calculator_metadata": _calculator_metadata_record(result.metadata),
                }
            )
            return optical_depth

        for molecule in molecules:
            relative_columns = dict(zero_relative_columns)
            # H2O has denominator 1, so its reference run uses the supplied
            # full profile unchanged.  A non-water molecule uniformly scales
            # its supplied full profile until the observer-altitude VMR is
            # exactly REF_COLUMN[molecule].
            relative_columns[molecule] = (
                references[molecule] / self.column_scale_denominators[molecule]
            )
            result = self._run_lblrtm(relative_columns)
            reference_depth = normalize_run(
                result,
                role="fit_molecule",
                molecule=molecule,
                relative_columns=relative_columns,
            )
            unit_depth = np.array(
                # The fit supplies C_m = s_m * REF_COLUMN[m].  Persisting
                # tau_ref / REF_COLUMN therefore preserves the public fit
                # parameter while the reference tau itself comes directly
                # from IEMIT=0 rather than from -log(transmission).
                reference_depth / _REFERENCE_COLUMNS[molecule],
                dtype=np.float64,
                copy=True,
            )
            unit_depth.setflags(write=False)
            unit_depths[molecule] = unit_depth

        fixed_relative_columns = dict(zero_relative_columns)
        for molecule, column in fixed_columns.items():
            fixed_relative_columns[molecule] = column / self.column_scale_denominators[molecule]
        fixed_result = self._run_lblrtm(fixed_relative_columns)
        fixed_depth = normalize_run(
            fixed_result,
            role="fixed_molecules",
            molecule=None,
            relative_columns=fixed_relative_columns,
        )
        if cache_grid is None:  # pragma: no cover - fixed run always defines it
            raise RuntimeError("LBLRTM cache build produced no wavelength grid")
        if (
            common_coverage_lower_nm is None or common_coverage_upper_nm is None
        ):  # pragma: no cover - first run establishes both values
            raise RuntimeError("cache common coverage is unavailable")
        final_cell_edges = _full_cell_edges(cache_grid)
        for run_record in run_records:
            run_record["final_cache_grid_size"] = int(cache_grid.size)

        build_record = {
            "schema": _CACHE_SCHEMA,
            "run_count": len(molecules) + 1,
            "fit_molecules": list(molecules),
            "fixed_columns": fixed_columns,
            "spectral_output": "optical_depth",
            "iemit": 0,
            "iod": 1,
            "optical_depth_grid_spacing_cm1": (self.optical_depth_grid_spacing_cm1),
            "nonfinite_optical_depth_policy": "reject_complete_build",
            "finite_negative_optical_depth_policy": "retain_unchanged",
            "optical_depth_policy": _DIRECT_OPTICAL_DEPTH_POLICY,
            "first_run_defines_grid": True,
            "different_grid_policy": _REGRID_POLICY,
            "grid_difference_observed": grid_difference_observed,
            "common_full_cell_coverage_nm": [
                common_coverage_lower_nm,
                common_coverage_upper_nm,
            ],
            "final_cache_grid_range_nm": [
                float(cache_grid[0]),
                float(cache_grid[-1]),
            ],
            "final_cache_cell_coverage_nm": [
                float(final_cell_edges[0]),
                float(final_cell_edges[-1]),
            ],
            "profile_sha256": _profile_sha256(self.atmosphere_profile),
            "resource_fingerprint": self.resource_provenance["manifest_fingerprint"],
            "runs": run_records,
        }
        # Commit state only after all runs, rebinning, and provenance capture
        # have succeeded, so a failed rebuild never exposes a partial cache.
        self.wavegrid = cache_grid
        self.fit_molecules = molecules
        self.ref = references
        self.fixed = fixed_columns
        self.tau_unit = unit_depths
        self.tau_fixed = fixed_depth
        self.build_provenance = _freeze_json(build_record)
        self.cache_metadata = expected_cache_metadata
        self.cache_key = expected_cache_key
        return self

    def optical_depth(self, **scales):
        """Return total optical depth for public fit-molecule columns."""

        if self.wavegrid is None or self.tau_fixed is None:
            raise RuntimeError("cache not built; call build() first")
        normalized_scales: dict[str, float] = {}
        for supplied_name, supplied_value in scales.items():
            try:
                molecule = normalize_lblrtm_molecule_name(supplied_name).lower()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid scale molecule {supplied_name!r}: {exc}") from exc
            if molecule in normalized_scales:
                raise ValueError(f"duplicate scale molecule after normalization: {molecule}")
            normalized_scales[molecule] = _coerce_finite_float(
                supplied_value,
                field_name=f"column for {molecule}",
                minimum=0.0,
            )
        unknown = set(normalized_scales) - set(self.fit_molecules)
        if unknown:
            raise ValueError(f"{sorted(unknown)} not in fit_molecules {self.fit_molecules}")

        total = np.array(self.tau_fixed, dtype=np.float64, copy=True)
        for molecule in self.fit_molecules:
            column = normalized_scales.get(molecule, self.ref[molecule])
            with np.errstate(invalid="ignore", over="ignore"):
                total += column * self.tau_unit[molecule]
        return total

    def transmittance(self, **scales):
        """Return cached telluric transmission for public molecule columns."""

        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            return np.exp(-self.optical_depth(**scales))

    def save(self, path, *, overwrite=False):
        """Validate and atomically publish a schema-v3 cache archive."""

        if self.wavegrid is None or self.tau_fixed is None:
            raise RuntimeError("cache not built; call build() before save()")
        if self.build_provenance is None or self.cache_metadata is None or self.cache_key is None:
            raise RuntimeError("internal error: the built cache has no build record")
        if not isinstance(overwrite, (bool, np.bool_)):
            raise TypeError("overwrite must be bool")

        current_cache_metadata = _cache_key_metadata(
            resource_provenance=self.resource_provenance,
            profile_digest=_profile_sha256(self.atmosphere_profile),
            window=self.window,
            effective_airmass=self.effective_airmass,
            observer_altitude_km=self.atmosphere["alt"],
            path_end_altitude_km=float(self.atmosphere_profile["z_km"][-1]),
            reference_latitude_deg=self.atmosphere["lat"],
            vac2air=self.vac2air,
            wavelength_conversion_formula=self.wavelength_conversion_formula,
            wave_margin_nm=self.wave_margin_nm,
            continuum_mode=self.continuum_mode,
            optical_depth_grid_spacing_cm1=(self.optical_depth_grid_spacing_cm1),
            fit_molecules=self.fit_molecules,
            reference_columns=self.ref,
            fixed_columns=self.fixed,
            column_scale_denominators=self.column_scale_denominators,
        )
        if _json_ready(current_cache_metadata) != _json_ready(self.cache_metadata):
            raise RuntimeError(
                "internal error: the cache inputs changed after it was built; rebuild it "
                "before saving"
            )
        if compute_tau_cache_key(current_cache_metadata) != self.cache_key:
            raise RuntimeError("internal error: cache_key does not match the built cache")

        output_path = _cache_path(path)
        if not output_path.parent.exists():
            raise FileNotFoundError(f"cache output directory does not exist: {output_path.parent}")
        if not output_path.parent.is_dir():
            raise NotADirectoryError(
                f"cache output parent is not a directory: {output_path.parent}"
            )

        lock_path = _cache_lock_path(output_path)
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                "cache publication lock already exists; refusing a "
                "concurrent or stale last-writer-wins save: "
                f"{lock_path}"
            ) from exc

        temporary_path: Path | None = None
        try:
            if output_path.exists() and not bool(overwrite):
                raise FileExistsError(
                    "cache output already exists; pass overwrite=True only "
                    f"after validating the replacement target: {output_path}"
                )

            arrays: dict[str, np.ndarray] = {
                "wavegrid": np.asarray(self.wavegrid, dtype=np.float64),
                "tau_fixed": np.asarray(self.tau_fixed, dtype=np.float64),
            }
            for molecule in self.fit_molecules:
                arrays[f"tauunit_{molecule}"] = np.asarray(
                    self.tau_unit[molecule],
                    dtype=np.float64,
                )
            for field_name in _PROFILE_FIELDS:
                arrays[f"profile_{field_name}"] = np.asarray(
                    self.atmosphere_profile[field_name],
                    dtype=np.float64,
                )

            array_hashes = {key: _array_sha256(array) for key, array in sorted(arrays.items())}
            metadata = {
                "schema": _CACHE_SCHEMA,
                "cache_key": self.cache_key,
                "cache_key_metadata": self.cache_metadata,
                "window": dict(self.window),
                "atmosphere": dict(self.atmosphere),
                "angle": self.angle,
                "effective_airmass": self.effective_airmass,
                "vac2air": self.vac2air,
                "wavelength_conversion_formula": (self.wavelength_conversion_formula),
                "wave_margin_nm": self.wave_margin_nm,
                "continuum_mode": self.continuum_mode,
                "optical_depth_grid_spacing_cm1": (self.optical_depth_grid_spacing_cm1),
                "timeout_s": self.timeout_s,
                "verbose": self.verbose,
                "fit_molecules": list(self.fit_molecules),
                "ref": dict(self.ref),
                "fixed": dict(self.fixed),
                "profile_fields": list(_PROFILE_FIELDS),
                "profile_level_count": _PROFILE_LEVEL_COUNT,
                "profile_sha256": _profile_sha256(self.atmosphere_profile),
                "resource_provenance": self.resource_provenance,
                "build_provenance": self.build_provenance,
                "array_sha256": array_hashes,
            }
            encoded_metadata = _json_bytes(metadata)
            arrays["_meta"] = np.frombuffer(
                encoded_metadata,
                dtype=np.uint8,
            ).copy()

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp.npz",
                dir=output_path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())

            TauCache.load(
                temporary_path,
                resources=self.resources,
                expected_cache_key=self.cache_key,
                expected_metadata=self.cache_metadata,
            )
            if output_path.exists() and not bool(overwrite):
                raise FileExistsError(
                    "cache output appeared during publication; refusing "
                    f"last-writer-wins replacement: {output_path}"
                )
            if bool(overwrite):
                os.replace(temporary_path, output_path)
            else:
                try:
                    os.link(temporary_path, output_path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        "cache output appeared during atomic publication; "
                        "refusing last-writer-wins replacement: "
                        f"{output_path}"
                    ) from exc
                temporary_path.unlink()
            temporary_path = None
            _fsync_directory(output_path.parent)
        finally:
            if temporary_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    temporary_path.unlink()
            os.close(lock_descriptor)
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()
            _fsync_directory(output_path.parent)
        return output_path

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_cache_key=None,
        expected_metadata=None,
        resources=None,
    ):
        """Load a cache with ``allow_pickle=False`` and verify all array hashes.

        ``resources`` is optional because evaluating an existing cache performs
        no external lookup or LBLRTM execution.  Supplying resources verifies
        their complete-validator identity and enables a later explicit rebuild.
        At least one caller-owned expected identity gate is mandatory.
        """

        if expected_cache_key is None and expected_metadata is None:
            raise ValueError("loading a cache requires expected_cache_key or expected_metadata")
        normalized_expected_key = (
            None
            if expected_cache_key is None
            else _normalize_sha256(
                expected_cache_key,
                field_name="expected_cache_key",
            )
        )
        if expected_metadata is None:
            normalized_expected_metadata = None
        else:
            if not isinstance(expected_metadata, Mapping):
                raise TypeError("expected_metadata must be a mapping or None")
            normalized_expected_metadata = _freeze_json(expected_metadata)
            metadata_expected_key = compute_tau_cache_key(normalized_expected_metadata)
            if (
                normalized_expected_key is not None
                and normalized_expected_key != metadata_expected_key
            ):
                raise ValueError("expected_cache_key contradicts expected_metadata")

        input_path = _cache_path(path)
        try:
            archive_context = np.load(input_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot open cache archive {input_path}") from exc

        with archive_context as archive:
            try:
                metadata_array = np.asarray(archive["_meta"])
            except KeyError as exc:
                raise ValueError("cache archive is missing '_meta'") from exc
            if (
                metadata_array.dtype != np.uint8
                or metadata_array.ndim != 1
                or metadata_array.size > _MAXIMUM_METADATA_BYTES
            ):
                raise ValueError("cache metadata array is invalid")
            try:
                metadata = json.loads(metadata_array.tobytes().decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("cache metadata is not valid ASCII JSON") from exc
            if not isinstance(metadata, dict) or metadata.get("schema") != _CACHE_SCHEMA:
                raise ValueError(
                    "optical-depth cache: unsupported file format (written by another "
                    "gtellcorr version?)"
                )
            expected_metadata_fields = {
                "schema",
                "cache_key",
                "cache_key_metadata",
                "window",
                "atmosphere",
                "angle",
                "effective_airmass",
                "vac2air",
                "wavelength_conversion_formula",
                "wave_margin_nm",
                "continuum_mode",
                "optical_depth_grid_spacing_cm1",
                "timeout_s",
                "verbose",
                "fit_molecules",
                "ref",
                "fixed",
                "profile_fields",
                "profile_level_count",
                "profile_sha256",
                "resource_provenance",
                "build_provenance",
                "array_sha256",
            }
            if set(metadata) != expected_metadata_fields:
                raise ValueError("optical-depth cache: unexpected metadata fields")
            cache_key, cache_metadata = _require_expected_cache_identity(
                recorded_cache_key=metadata["cache_key"],
                recorded_metadata=metadata["cache_key_metadata"],
                expected_cache_key=normalized_expected_key,
                expected_metadata=normalized_expected_metadata,
            )

            try:
                molecules = _normalize_molecule_collection(metadata["fit_molecules"])
                recorded_fields = tuple(metadata["profile_fields"])
                recorded_level_count = int(metadata["profile_level_count"])
                recorded_hashes = dict(metadata["array_sha256"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("cache metadata is incomplete") from exc
            if recorded_fields != _PROFILE_FIELDS:
                raise ValueError(
                    "optical-depth cache: the atmospheric profile has unexpected fields"
                )
            if recorded_level_count != _PROFILE_LEVEL_COUNT:
                raise ValueError(
                    "optical-depth cache: the atmospheric profile has unexpected levels"
                )

            expected_array_keys = {
                "wavegrid",
                "tau_fixed",
                "_meta",
                *(f"tauunit_{molecule}" for molecule in molecules),
                *(f"profile_{field_name}" for field_name in _PROFILE_FIELDS),
            }
            if set(archive.files) != expected_array_keys:
                missing = sorted(expected_array_keys - set(archive.files))
                unexpected = sorted(set(archive.files) - expected_array_keys)
                raise ValueError(
                    "cache archive arrays do not match metadata; "
                    f"missing={missing}, unexpected={unexpected}"
                )

            wavegrid = _read_float_array(archive, "wavegrid")
            if (
                wavegrid.size < 2
                or wavegrid.size > _MAXIMUM_CACHE_GRID_POINTS
                or not np.all(np.isfinite(wavegrid))
                or np.any(wavegrid <= 0.0)
                or np.any(np.diff(wavegrid) <= 0.0)
            ):
                raise ValueError("cache wavegrid is invalid")
            tau_fixed = _read_float_array(
                archive,
                "tau_fixed",
                expected_size=wavegrid.size,
            )
            tau_unit = {
                molecule: _read_float_array(
                    archive,
                    f"tauunit_{molecule}",
                    expected_size=wavegrid.size,
                )
                for molecule in molecules
            }
            if not np.all(np.isfinite(tau_fixed)) or any(
                not np.all(np.isfinite(tau_unit[molecule])) for molecule in molecules
            ):
                raise ValueError("cache optical-depth arrays contain non-finite values")
            profile_input = {
                field_name: _read_float_array(
                    archive,
                    f"profile_{field_name}",
                    expected_size=_PROFILE_LEVEL_COUNT,
                )
                for field_name in _PROFILE_FIELDS
            }
            profile = _snapshot_profile(profile_input)

            loaded_arrays = {
                "wavegrid": wavegrid,
                "tau_fixed": tau_fixed,
                **{f"tauunit_{molecule}": tau_unit[molecule] for molecule in molecules},
                **{f"profile_{field_name}": profile[field_name] for field_name in _PROFILE_FIELDS},
            }
            if set(recorded_hashes) != set(loaded_arrays):
                raise ValueError("cache array-hash inventory does not match")
            for key, array in loaded_arrays.items():
                recorded_digest = recorded_hashes.get(key)
                if not isinstance(recorded_digest, str) or _array_sha256(array) != recorded_digest:
                    raise ValueError(f"cache array hash mismatch for {key!r}")

        profile_digest = _profile_sha256(profile)
        if metadata.get("profile_sha256") != profile_digest:
            raise ValueError("optical-depth cache: the stored atmospheric profile is damaged")
        try:
            ref = {str(key): float(value) for key, value in dict(metadata["ref"]).items()}
            fixed = {str(key): float(value) for key, value in dict(metadata["fixed"]).items()}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cache reference-column metadata is invalid") from exc
        expected_ref = {molecule: float(_REFERENCE_COLUMNS[molecule]) for molecule in molecules}
        if ref != expected_ref:
            raise ValueError("cache reference columns do not match this implementation")
        expected_fixed_names = set(MOLECULES) - set(molecules)
        if set(fixed) != expected_fixed_names or any(
            not math.isfinite(value) or value < 0.0 for value in fixed.values()
        ):
            raise ValueError("cache fixed-column metadata is invalid")

        try:
            resource_record = _normalize_loaded_resource_provenance(metadata["resource_provenance"])
            recorded_fingerprint = resource_record["manifest_fingerprint"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("optical-depth cache: its LBLRTM bundle record is invalid") from exc

        if resources is None:
            validated_resources = None
        else:
            current_record = _resource_provenance(resources)
            recorded_resource_identity = {
                "manifest_fingerprint": recorded_fingerprint,
                "lblrtm_precision": resource_record["lblrtm_precision"],
                "tape12_abi": resource_record["tape12_abi"],
            }
            current_resource_identity = {
                "manifest_fingerprint": current_record["manifest_fingerprint"],
                "lblrtm_precision": current_record["lblrtm_precision"],
                "tape12_abi": current_record["tape12_abi"],
            }
            if _json_ready(current_resource_identity) != _json_ready(recorded_resource_identity):
                raise ValueError(
                    "the optical-depth cache was made with a different LBLRTM bundle (resource "
                    "fingerprint, precision or TAPE12 layout differ)"
                )
            validated_resources = resources

        try:
            window = {
                "lowfreq": _coerce_finite_float(
                    metadata["window"]["lowfreq"],
                    field_name="saved lowfreq",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "highfreq": _coerce_finite_float(
                    metadata["window"]["highfreq"],
                    field_name="saved highfreq",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            }
            if window["highfreq"] <= window["lowfreq"]:
                raise ValueError("saved highfreq must exceed saved lowfreq")
            atmosphere = {
                str(key): float(value) for key, value in dict(metadata["atmosphere"]).items()
            }
            angle = _coerce_finite_float(
                metadata["angle"],
                field_name="saved angle",
                minimum=0.0,
            )
            effective_airmass = _coerce_finite_float(
                metadata["effective_airmass"],
                field_name="saved effective_airmass",
                minimum=1.0,
            )
            vac2air = metadata["vac2air"]
            verbose = metadata["verbose"]
            if not isinstance(vac2air, bool) or not isinstance(verbose, bool):
                raise ValueError("saved boolean controls are invalid")
            formula = str(metadata["wavelength_conversion_formula"])
            if formula not in _SUPPORTED_WAVELENGTH_FORMULAS:
                raise ValueError("saved wavelength formula is unsupported")
            margin = _coerce_finite_float(
                metadata["wave_margin_nm"],
                field_name="saved wave_margin_nm",
                minimum=0.0,
            )
            continuum_mode = metadata["continuum_mode"]
            if continuum_mode != "none":
                raise ValueError("the optical-depth cache supports only continuum_mode='none'")
            optical_depth_spacing = _coerce_finite_float(
                metadata["optical_depth_grid_spacing_cm1"],
                field_name="saved optical_depth_grid_spacing_cm1",
                minimum=0.0,
                strict_minimum=True,
            )
            saved_timeout = metadata["timeout_s"]
            timeout = (
                None
                if saved_timeout is None
                else _coerce_finite_float(
                    saved_timeout,
                    field_name="saved timeout_s",
                    minimum=0.0,
                    strict_minimum=True,
                )
            )
            build_record = metadata["build_provenance"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cache configuration metadata is invalid") from exc

        required_atmosphere_keys = {"pressure", "temperature", "lat", "alt"}
        if not required_atmosphere_keys.issubset(atmosphere):
            raise ValueError("cache atmosphere metadata is incomplete")
        if ("angle" in atmosphere) == ("airmass" in atmosphere):
            raise ValueError("cache atmosphere metadata must contain exactly one of angle/airmass")
        column_scale_denominators = _column_scale_denominators(
            profile,
            atmosphere["alt"],
        )
        validated_build_record = _validate_loaded_build_provenance(
            build_record,
            cache_wavegrid=wavegrid,
            fit_molecules=molecules,
            fixed_columns=fixed,
            column_scale_denominators=column_scale_denominators,
            optical_depth_grid_spacing_cm1=optical_depth_spacing,
            profile_digest=profile_digest,
            resource_fingerprint=recorded_fingerprint,
        )
        reconstructed_cache_metadata = _cache_key_metadata(
            resource_provenance=resource_record,
            profile_digest=profile_digest,
            window=window,
            effective_airmass=effective_airmass,
            observer_altitude_km=atmosphere["alt"],
            path_end_altitude_km=float(profile["z_km"][-1]),
            reference_latitude_deg=atmosphere["lat"],
            vac2air=vac2air,
            wavelength_conversion_formula=formula,
            wave_margin_nm=margin,
            continuum_mode=continuum_mode,
            optical_depth_grid_spacing_cm1=optical_depth_spacing,
            fit_molecules=molecules,
            reference_columns=ref,
            fixed_columns=fixed,
            column_scale_denominators=column_scale_denominators,
        )
        if _json_ready(reconstructed_cache_metadata) != _json_ready(cache_metadata):
            raise ValueError("cache key metadata contradicts the serialized cache state")
        if compute_tau_cache_key(reconstructed_cache_metadata) != cache_key:
            raise ValueError("serialized cache state does not reproduce cache_key")

        obj = cls.__new__(cls)
        obj.window = window
        obj.atmosphere = atmosphere
        obj.angle = angle
        obj.effective_airmass = effective_airmass
        obj.vac2air = vac2air
        obj.atmosphere_profile = profile
        obj.column_scale_denominators = column_scale_denominators
        obj.resources = validated_resources
        obj.resource_provenance = _freeze_json(resource_record)
        obj.wavelength_conversion_formula = formula
        obj.wave_margin_nm = margin
        obj.continuum_mode = continuum_mode
        obj.optical_depth_grid_spacing_cm1 = optical_depth_spacing
        obj.timeout_s = timeout
        obj.verbose = verbose
        obj.wavegrid = wavegrid
        obj.fit_molecules = molecules
        obj.ref = ref
        obj.fixed = fixed
        obj.tau_unit = tau_unit
        obj.tau_fixed = tau_fixed
        obj.build_provenance = validated_build_record
        obj.cache_metadata = reconstructed_cache_metadata
        obj.cache_key = cache_key
        return obj
