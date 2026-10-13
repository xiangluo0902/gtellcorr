"""Load, locate, checksum and check LBLRTM resource bundles.

The resource manifest records the
exact LNFL/LBLRTM inputs and products needed by gtellcorr and validates every
declared file before a radiative-transfer backend is allowed to use it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .lblrtm_errors import (
    LblrtmManifestError,
    LblrtmResourceIntegrityError,
)
from .lnfl_molecule_selection import (
    LNFL_MOLECULE_SLOT_COUNT,
    load_lnfl_molecule_registry,
    validate_lnfl_molecule_selection_mask,
)
from .tape3_metadata import (
    Tape3FormatError,
    Tape3ValidationError,
    compute_tape3_scientific_sha256,
    compute_tape3_semantic_sha256,
    read_tape3_header,
    validate_tape3_metadata,
)

DEFAULT_MANIFEST_FILENAME: Final = (
    "lblrtm_v12_17_lnfl_v3_2_aer_v3_9_gnu_double_project_runtime_manifest.json"
)
SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1, 2})
_BUNDLE_VALIDATOR_ID: Final = "gtellcorr.validate_lblrtm_resource_bundle/v2"
_VALIDATED_RESOURCE_BUNDLE_TOKEN: Final = object()
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SIZE_BYTES_PATTERN: Final = re.compile(r"^(?:0|[1-9][0-9]*)$")
# The AER LBLRTM v12.17 README.md (lines 214 and 222) requires the
# LNFL/TAPE3 interval to extend at least 25 cm-1 beyond both ends of every
# LBLRTM calculation interval.
_LBLRTM_REQUIRED_TAPE3_MARGIN_CM1: Final[float] = 25.0
_RUNTIME_RESOURCE_PRODUCT_NAMES: Final = (
    "lblrtm_executable",
    "tape3",
    "mt_ckd_h2o_continuum_coefficients",
)
_TAPE5_BOUNDS_PATTERN: Final = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_TAPE6_SELECTION_PATTERN: Final = re.compile(r"^\s*([A-Za-z0-9+]+)\s*=\s*([01])\s*$")
_TAPE6_LINE_COUNT_PATTERN: Final = re.compile(r"^\s*([A-Za-z0-9+]+)\s*=\s*(\d+)(?:\s+.*)?$")
_TAPE6_BOUNDS_PATTERN: Final = re.compile(
    r"LOWEST LINE\s*=\s*([0-9.+\-Ee]+)\s*CM-1,\s*"
    r"HIGHEST LINE\s*=\s*([0-9.+\-Ee]+)\s*CM-1,\s*"
    r"TOTAL NUMBER OF LINES\s*=\s*(\d+)"
)
_ALLOWED_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "$schema",
        "schema_version",
        "resource_set_id",
        "status",
        "created_utc",
        "scientific_contract",
        "sources",
        "build_contract",
        "products",
        "compatibility",
        "golden_compatibility",
        "licenses",
        "citations",
        "resolved_discrepancies",
        "validation",
    }
)
#: Manifest fields that enter the manifest checksum (see
#: :func:`compute_lblrtm_resource_fingerprint`).
_FINGERPRINT_FIELDS: Final = (
    "schema_version",
    "resource_set_id",
    "sources",
    "products",
    "build_contract",
    "scientific_contract",
    "licenses",
)
_REQUIRED_SOURCE_RECORDS: Final = frozenset(
    {
        "aer_line_file_archive",
        "aer_line_file",
        "lnfl_source_archive",
        "lblrtm_source_archive",
    }
)
_REQUIRED_PRODUCT_RECORDS: Final = frozenset(
    {
        "lnfl_executable",
        "lblrtm_executable",
        "lnfl_tape5",
        "lnfl_tape6",
        "tape3",
        "mt_ckd_h2o_continuum_coefficients",
        "build_recipe",
        "lblrtm_addlibs_template",
    }
)
_TAPE12_ABI_EVIDENCE_KEYS: Final = (
    "build_contract.tape12_abi.byte_order",
    "build_contract.tape12_abi.record_marker_bytes",
    "build_contract.tape12_abi.real_bytes",
    "build_contract.tape12_abi.integer_bytes",
)


@dataclass(frozen=True, slots=True)
class Tape12AbiContract:
    """Binary layout of an LBLRTM TAPE12 file (read-only)."""

    byte_order: str
    record_marker_bytes: int
    real_bytes: int
    integer_bytes: int

    def __post_init__(self) -> None:
        if self.byte_order not in {"little", "big"}:
            raise ValueError("TAPE12 byte_order must be 'little' or 'big'")
        for field_name, value in (
            ("record_marker_bytes", self.record_marker_bytes),
            ("real_bytes", self.real_bytes),
            ("integer_bytes", self.integer_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value not in {4, 8}:
                raise ValueError(f"TAPE12 {field_name} must be 4 or 8")


def _tape12_abi_from_manifest_data(
    manifest_data: Mapping[str, Any],
) -> Tape12AbiContract:
    """Return a validated TAPE12 ABI, including the frozen v1 inference."""

    schema_version = manifest_data.get("schema_version")
    build_contract = _require_mapping(
        manifest_data.get("build_contract"),
        field="build_contract",
    )
    if schema_version == 1:
        if "precision" in build_contract and build_contract["precision"] != "single":
            raise LblrtmManifestError("schema v1 build_contract.precision must be 'single'")
        if "byte_order" in build_contract and build_contract["byte_order"] != "little":
            raise LblrtmManifestError("schema v1 build_contract.byte_order must be 'little'")
        if (
            "fortran_record_marker_bytes" in build_contract
            and build_contract["fortran_record_marker_bytes"] != 4
        ):
            raise LblrtmManifestError(
                "schema v1 build_contract.fortran_record_marker_bytes must be 4"
            )
        return Tape12AbiContract(
            byte_order="little",
            record_marker_bytes=4,
            real_bytes=4,
            integer_bytes=4,
        )
    if schema_version != 2:
        raise LblrtmManifestError(
            f"cannot tell the TAPE12 binary layout of manifest schema_version {schema_version!r}"
        )
    raw_abi = _require_mapping(
        build_contract.get("tape12_abi"),
        field="build_contract.tape12_abi",
    )
    try:
        return Tape12AbiContract(
            byte_order=raw_abi.get("byte_order"),
            record_marker_bytes=raw_abi.get("record_marker_bytes"),
            real_bytes=raw_abi.get("real_bytes"),
            integer_bytes=raw_abi.get("integer_bytes"),
        )
    except (TypeError, ValueError) as exc:
        raise LblrtmManifestError(
            f"invalid build_contract.tape12_abi (TAPE12 binary layout): {exc}"
        ) from exc


def _tape12_abi_from_evidence(
    evidence: Mapping[str, str],
) -> Tape12AbiContract:
    """Rebuild the TAPE12 layout from the result of the full bundle check only."""

    present_keys = {key for key in _TAPE12_ABI_EVIDENCE_KEYS if key in evidence}
    if present_keys != set(_TAPE12_ABI_EVIDENCE_KEYS):
        missing = sorted(set(_TAPE12_ABI_EVIDENCE_KEYS) - present_keys)
        raise LblrtmResourceIntegrityError(
            "internal error: the bundle check did not record the TAPE12 binary layout: "
            + ", ".join(missing)
        )
    try:
        return Tape12AbiContract(
            byte_order=evidence[_TAPE12_ABI_EVIDENCE_KEYS[0]],
            record_marker_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[1]]),
            real_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[2]]),
            integer_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[3]]),
        )
    except (TypeError, ValueError) as exc:
        raise LblrtmResourceIntegrityError(
            f"internal error: the bundle check recorded an invalid TAPE12 binary layout: {exc}"
        ) from exc


def _has_complete_runtime_resource_evidence(
    evidence: Mapping[str, str],
) -> bool:
    """Return whether the runtime file checksums and wavenumber limits are recorded."""

    try:
        tape12_abi = _tape12_abi_from_evidence(evidence)
    except LblrtmResourceIntegrityError:
        return False
    schema_version = evidence.get("manifest_schema_version")
    if schema_version not in {"1", "2"}:
        return False
    if schema_version == "1" and tape12_abi != Tape12AbiContract(
        byte_order="little",
        record_marker_bytes=4,
        real_bytes=4,
        integer_bytes=4,
    ):
        return False
    if schema_version == "2" and (
        tape12_abi.byte_order != "little"
        or tape12_abi.record_marker_bytes != 4
        or tape12_abi.real_bytes != tape12_abi.integer_bytes
    ):
        return False
    for product_name in _RUNTIME_RESOURCE_PRODUCT_NAMES:
        prefix = f"products.{product_name}"
        size_bytes = evidence.get(f"{prefix}.size_bytes")
        digest = evidence.get(f"{prefix}.sha256")
        if (
            size_bytes is None
            or _SIZE_BYTES_PATTERN.fullmatch(size_bytes) is None
            or digest is None
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            return False
    try:
        requested_minimum = float(evidence["scientific_contract.lnfl_requested_vmin_cm1"])
        requested_maximum = float(evidence["scientific_contract.lnfl_requested_vmax_cm1"])
        required_margin = float(evidence["scientific_contract.required_line_margin_cm1"])
        safe_minimum = float(evidence["scientific_contract.safe_calculation_vmin_cm1"])
        safe_maximum = float(evidence["scientific_contract.safe_calculation_vmax_cm1"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return not (
        not all(
            math.isfinite(value)
            for value in (
                requested_minimum,
                requested_maximum,
                required_margin,
                safe_minimum,
                safe_maximum,
            )
        )
        or requested_minimum <= 0.0
        or requested_maximum <= requested_minimum
        or required_margin != _LBLRTM_REQUIRED_TAPE3_MARGIN_CM1
        or safe_minimum != requested_minimum + required_margin
        or safe_maximum != requested_maximum - required_margin
        or safe_maximum <= safe_minimum
    )


def _runtime_resource_validation_evidence(
    manifest_data: Mapping[str, Any],
) -> dict[str, str]:
    """Copy the checked runtime file checksums and wavenumber limits into a read-only record."""

    products = manifest_data["products"]
    tape12_abi = _tape12_abi_from_manifest_data(manifest_data)
    evidence: dict[str, str] = {}
    evidence.update(
        {
            "manifest_schema_version": str(manifest_data["schema_version"]),
            "build_contract.tape12_abi.byte_order": tape12_abi.byte_order,
            "build_contract.tape12_abi.record_marker_bytes": str(tape12_abi.record_marker_bytes),
            "build_contract.tape12_abi.real_bytes": str(tape12_abi.real_bytes),
            "build_contract.tape12_abi.integer_bytes": str(tape12_abi.integer_bytes),
        }
    )
    for product_name in _RUNTIME_RESOURCE_PRODUCT_NAMES:
        record = products[product_name]
        prefix = f"products.{product_name}"
        evidence[f"{prefix}.size_bytes"] = str(record["size_bytes"])
        evidence[f"{prefix}.sha256"] = str(record["sha256"])
    requested_bounds = manifest_data["scientific_contract"]["requested_bounds_cm1"]
    requested_minimum = float(requested_bounds[0])
    requested_maximum = float(requested_bounds[1])
    safe_minimum = requested_minimum + _LBLRTM_REQUIRED_TAPE3_MARGIN_CM1
    safe_maximum = requested_maximum - _LBLRTM_REQUIRED_TAPE3_MARGIN_CM1
    evidence["scientific_contract.lnfl_requested_vmin_cm1"] = str(requested_bounds[0])
    evidence["scientific_contract.lnfl_requested_vmax_cm1"] = str(requested_bounds[1])
    evidence["scientific_contract.required_line_margin_cm1"] = str(
        _LBLRTM_REQUIRED_TAPE3_MARGIN_CM1
    )
    evidence["scientific_contract.safe_calculation_vmin_cm1"] = str(safe_minimum)
    evidence["scientific_contract.safe_calculation_vmax_cm1"] = str(safe_maximum)
    return evidence


@dataclass(frozen=True, slots=True)
class LblrtmResourceManifest:
    """Checked JSON manifest and the checksum of its physics part."""

    path: Path
    _serialized_data: str
    _resource_set_id: str
    _status: str
    fingerprint: str

    @property
    def data(self) -> Mapping[str, Any]:
        """Return a copy, so that changing it cannot invalidate the recorded checksum."""

        value = json.loads(self._serialized_data)
        if not isinstance(value, Mapping):  # pragma: no cover - checked by the constructor
            raise AssertionError("internal error: the manifest did not serialize to a JSON object")
        return value

    @property
    def resource_set_id(self) -> str:
        """Return the human-readable name of the resource set."""

        return self._resource_set_id

    @property
    def status(self) -> str:
        """Return the lifecycle status: candidate, accepted, or retired."""

        return self._status

    @property
    def tape12_abi(self) -> Tape12AbiContract:
        """Return the TAPE12 binary layout declared by this manifest."""

        return _tape12_abi_from_manifest_data(self.data)


@dataclass(frozen=True, slots=True)
class ResolvedLblrtmResourceFiles:
    """Absolute resource paths with explicit resolution/validation state.

    ``resolve_lblrtm_resource_files`` fills ``manifest_fingerprint`` but
    leaves ``validation_evidence`` unset.  Only
    ``validate_lblrtm_resource_bundle`` returns an instance whose
    ``is_fully_validated`` property is true.
    """

    root: Path
    files: Mapping[str, Path]
    manifest_fingerprint: str | None = None
    validation_evidence: Mapping[str, str] | None = None
    _validation_token: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self,
            "files",
            MappingProxyType(dict(self.files)),
        )
        if (
            self.manifest_fingerprint is not None
            and _SHA256_PATTERN.fullmatch(self.manifest_fingerprint) is None
        ):
            raise ValueError("manifest_fingerprint must contain 64 lowercase hexadecimal digits")
        if self.validation_evidence is not None:
            if not isinstance(self.validation_evidence, Mapping):
                raise TypeError("internal error: validation_evidence must be a mapping or None")
            evidence = dict(self.validation_evidence)
            if any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in evidence.items()
            ):
                raise TypeError(
                    "internal error: validation_evidence keys and values must be strings"
                )
            object.__setattr__(
                self,
                "validation_evidence",
                MappingProxyType(evidence),
            )

    def require(self, logical_name: str) -> Path:
        """Return a declared file or raise a precise missing-key error."""

        try:
            return self.files[logical_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.files))
            raise KeyError(
                f"resource {logical_name!r} is not declared; available: {available}"
            ) from exc

    @property
    def is_fully_validated(self) -> bool:
        """Return whether the complete bundle validator produced this state."""

        evidence = self.validation_evidence
        return (
            self.manifest_fingerprint is not None
            and evidence is not None
            and evidence.get("validator") == _BUNDLE_VALIDATOR_ID
            and evidence.get("manifest_fingerprint") == self.manifest_fingerprint
            and _has_complete_runtime_resource_evidence(evidence)
            and self._validation_token is _VALIDATED_RESOURCE_BUNDLE_TOKEN
        )

    def require_fully_validated(self) -> ResolvedLblrtmResourceFiles:
        """Return ``self`` or reject a resolve-only resource collection."""

        if not self.is_fully_validated:
            raise LblrtmResourceIntegrityError(
                "LBLRTM resources were resolved but have not passed validate_lblrtm_resource_bundle"
            )
        return self

    def require_validated_calculation_bounds_cm1(self) -> tuple[float, float]:
        """Return the wavenumber limits LBLRTM can be run within, from the full bundle check.

        The full check first compares the LNFL-requested interval with the
        manifest, control files, build recipe and TAPE3 metadata, then removes the
        25 cm-1 line margin that the AER LBLRTM v12.17 README requires at each end
        of a calculation.  These are therefore
        narrower than the nominal LNFL/TAPE3 interval.
        """

        self.require_fully_validated()
        evidence = self.validation_evidence
        if evidence is None:  # pragma: no cover - set by require_fully_validated
            raise LblrtmResourceIntegrityError("internal error: the bundle check result is missing")
        try:
            minimum_wavenumber = float(evidence["scientific_contract.safe_calculation_vmin_cm1"])
            maximum_wavenumber = float(evidence["scientific_contract.safe_calculation_vmax_cm1"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise LblrtmResourceIntegrityError(
                "internal error: the bundle check did not record the TAPE3 wavenumber limits"
            ) from exc
        if (
            not math.isfinite(minimum_wavenumber)
            or not math.isfinite(maximum_wavenumber)
            or minimum_wavenumber <= 0.0
            or maximum_wavenumber <= minimum_wavenumber
        ):
            raise LblrtmResourceIntegrityError(
                "internal error: the bundle check recorded invalid TAPE3 wavenumber limits"
            )
        return minimum_wavenumber, maximum_wavenumber

    def require_validated_tape12_abi(self) -> Tape12AbiContract:
        """Return the TAPE12 binary layout recorded by the full bundle check."""

        self.require_fully_validated()
        evidence = self.validation_evidence
        if evidence is None:  # pragma: no cover - set by require_fully_validated
            raise LblrtmResourceIntegrityError("internal error: the bundle check result is missing")
        return _tape12_abi_from_evidence(evidence)


@dataclass(frozen=True, slots=True)
class LnflTape5Summary:
    """Scientific inputs encoded by an LNFL TAPE5 control file."""

    requested_vmin_cm1: float
    requested_vmax_cm1: float
    molecule_selection_mask: str

    @property
    def selected_molecule_ids(self) -> tuple[int, ...]:
        """Return one-based IDs whose mask flag is enabled."""

        return tuple(
            index for index, flag in enumerate(self.molecule_selection_mask, 1) if flag == "1"
        )


@dataclass(frozen=True, slots=True)
class LnflTape6Summary:
    """Molecule selection and line counts reported by LNFL."""

    molecule_selection_mask: str
    reported_slot_count: int
    actual_vmin_cm1: float
    actual_vmax_cm1: float
    total_line_count: int
    line_count_by_molecule_id: Mapping[int, int]

    @property
    def nonzero_molecule_ids(self) -> tuple[int, ...]:
        """Return molecule IDs with at least one line in the requested range."""

        return tuple(
            molecule_id
            for molecule_id, line_count in self.line_count_by_molecule_id.items()
            if line_count > 0
        )


def _reject_duplicate_json_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a dict while rejecting otherwise silent duplicate JSON keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LblrtmManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    """*value*, which must be a JSON object."""
    if not isinstance(value, Mapping):
        raise LblrtmManifestError(f"{field} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, *, field: str) -> str:
    """*value*, which must be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise LblrtmManifestError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    """*value*, which must be an integer >= *minimum* (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LblrtmManifestError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    """*value*, which must be a JSON array."""
    if not isinstance(value, list):
        raise LblrtmManifestError(f"{field} must be a JSON array")
    return value


def _validate_manifest_tape12_contract(
    *,
    schema_version: int,
    build_contract: Mapping[str, Any],
    products: Mapping[str, Any],
) -> Tape12AbiContract:
    """Validate precision, build targets, products, and the TAPE12 ABI."""

    if schema_version == 1:
        return _tape12_abi_from_manifest_data(
            {
                "schema_version": schema_version,
                "build_contract": build_contract,
            }
        )

    required_fields = {
        "lnfl_precision",
        "lblrtm_precision",
        "lnfl_make_target",
        "lblrtm_make_target",
        "lnfl_product_basename",
        "lblrtm_product_basename",
        "tape12_abi",
    }
    missing_fields = required_fields - set(build_contract)
    if missing_fields:
        raise LblrtmManifestError(
            "schema v2 build_contract is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )
    lnfl_precision = build_contract.get("lnfl_precision")
    lblrtm_precision = build_contract.get("lblrtm_precision")
    if lnfl_precision != "single":
        raise LblrtmManifestError("build_contract.lnfl_precision must be 'single'")
    if lblrtm_precision not in {"single", "double"}:
        raise LblrtmManifestError("build_contract.lblrtm_precision must be 'single' or 'double'")
    # accept the Linux GNU targets as well as macOS
    target_prefix = (
        "linux" if str(build_contract.get("lnfl_make_target", "")).startswith("linux") else "osx"
    )
    expected_targets = {
        "lnfl_make_target": f"{target_prefix}GNUsgl",
        "lblrtm_make_target": (
            f"{target_prefix}GNUdbl" if lblrtm_precision == "double" else f"{target_prefix}GNUsgl"
        ),
    }
    for field_name, expected_value in expected_targets.items():
        if build_contract.get(field_name) != expected_value:
            raise LblrtmManifestError(f"build_contract.{field_name} must be {expected_value!r}")

    raw_abi = _require_mapping(
        build_contract.get("tape12_abi"),
        field="build_contract.tape12_abi",
    )
    expected_abi_fields = {
        "byte_order",
        "record_marker_bytes",
        "real_bytes",
        "integer_bytes",
    }
    if set(raw_abi) != expected_abi_fields:
        raise LblrtmManifestError(
            "build_contract.tape12_abi must contain exactly: "
            + ", ".join(sorted(expected_abi_fields))
        )
    tape12_abi = _tape12_abi_from_manifest_data(
        {
            "schema_version": schema_version,
            "build_contract": build_contract,
        }
    )
    expected_numeric_bytes = 8 if lblrtm_precision == "double" else 4
    if tape12_abi != Tape12AbiContract(
        byte_order="little",
        record_marker_bytes=4,
        real_bytes=expected_numeric_bytes,
        integer_bytes=expected_numeric_bytes,
    ):
        raise LblrtmManifestError(
            f"build_contract.tape12_abi is inconsistent with lblrtm_precision={lblrtm_precision!r}"
        )
    # schema-1 field names, still accepted if they agree with the v2 fields
    legacy_aliases = {
        "precision": lblrtm_precision,
        "byte_order": tape12_abi.byte_order,
        "fortran_record_marker_bytes": tape12_abi.record_marker_bytes,
    }
    for field_name, expected_value in legacy_aliases.items():
        if field_name in build_contract and build_contract[field_name] != expected_value:
            raise LblrtmManifestError(
                f"build_contract.{field_name} contradicts the precision and TAPE12 layout "
                "given in the same manifest"
            )

    expected_suffixes = {
        "lnfl_product_basename": "_sgl",
        "lblrtm_product_basename": ("_dbl" if lblrtm_precision == "double" else "_sgl"),
    }
    for contract_field, product_name in (
        ("lnfl_product_basename", "lnfl_executable"),
        ("lblrtm_product_basename", "lblrtm_executable"),
    ):
        basename = _require_nonempty_string(
            build_contract.get(contract_field),
            field=f"build_contract.{contract_field}",
        )
        if PurePosixPath(basename).name != basename or "/" in basename:
            raise LblrtmManifestError(f"build_contract.{contract_field} must be a file basename")
        if not basename.endswith(expected_suffixes[contract_field]):
            raise LblrtmManifestError(
                f"build_contract.{contract_field} must end with "
                f"{expected_suffixes[contract_field]!r}"
            )
        product_record = _require_mapping(
            products.get(product_name),
            field=f"products.{product_name}",
        )
        product_path = _require_nonempty_string(
            product_record.get("relative_path"),
            field=f"products.{product_name}.relative_path",
        )
        if PurePosixPath(product_path).name != basename:
            raise LblrtmManifestError(
                f"build_contract.{contract_field} differs from "
                f"products.{product_name}.relative_path basename"
            )
    return tape12_abi


def _canonical_manifest_payload(data: Mapping[str, Any]) -> bytes:
    """Return the bytes (sorted-key JSON) from which the manifest checksum is computed.

    Only what determines a calculation enters it: the schema version, the
    resource-set id, the source and product files (sizes and hashes), the build
    settings, the physics settings (``scientific_contract``) and the licences. Descriptive or
    administrative fields (creation time, status, validation records,
    compatibility notes, citations) do not, so editing them leaves every
    optical-depth cache valid.
    """

    payload = {key: data[key] for key in _FINGERPRINT_FIELDS if key in data}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_lblrtm_resource_fingerprint(data: Mapping[str, Any]) -> str:
    """Compute the checksum of the parts of a manifest that determine a calculation."""

    return sha256(_canonical_manifest_payload(data)).hexdigest()


def _iter_primary_file_records(
    data: Mapping[str, Any],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """``(dotted name, record)`` of every source and product file."""
    for section_name in ("sources", "products"):
        section = _require_mapping(data.get(section_name), field=section_name)
        for logical_name, raw_record in section.items():
            record = _require_mapping(
                raw_record,
                field=f"{section_name}.{logical_name}",
            )
            yield f"{section_name}.{logical_name}", record


def _iter_file_records(
    data: Mapping[str, Any],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield every bundle file with a recorded checksum, including licence notices."""

    yield from _iter_primary_file_records(data)
    licenses = _require_mapping(data.get("licenses"), field="licenses")
    for license_name, raw_license in licenses.items():
        license_record = _require_mapping(
            raw_license,
            field=f"licenses.{license_name}",
        )
        if "notice_file" not in license_record:
            continue
        notice_record = _require_mapping(
            license_record["notice_file"],
            field=f"licenses.{license_name}.notice_file",
        )
        yield f"licenses.{license_name}.notice_file", notice_record


def validate_lblrtm_resource_manifest(
    data: Mapping[str, Any],
) -> None:
    """Check the manifest format and that its physics fields agree with each other."""

    unknown_fields = set(data) - _ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        raise LblrtmManifestError(
            "unknown top-level manifest fields: " + ", ".join(sorted(unknown_fields))
        )
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise LblrtmManifestError(
            "schema_version must be one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}, got {schema_version!r}"
        )
    _require_nonempty_string(data.get("resource_set_id"), field="resource_set_id")
    if data.get("status") not in {"candidate", "accepted", "retired"}:
        raise LblrtmManifestError("status must be one of: candidate, accepted, retired")
    _require_nonempty_string(data.get("created_utc"), field="created_utc")

    contract = _require_mapping(
        data.get("scientific_contract"),
        field="scientific_contract",
    )
    selection_field_width = _require_integer(
        contract.get("selection_field_width"),
        field="scientific_contract.selection_field_width",
        minimum=1,
    )
    if selection_field_width != LNFL_MOLECULE_SLOT_COUNT:
        raise LblrtmManifestError(
            "scientific_contract.selection_field_width must be 47 for LNFL v3.2"
        )
    try:
        mask = validate_lnfl_molecule_selection_mask(
            contract.get("molecule_selection_mask"),
            expected_slot_count=selection_field_width,
        )
    except (TypeError, ValueError) as exc:
        raise LblrtmManifestError(
            f"invalid scientific_contract.molecule_selection_mask: {exc}"
        ) from exc
    selected_ids = [index for index, flag in enumerate(mask, 1) if flag == "1"]
    if not selected_ids:
        raise LblrtmManifestError(
            "scientific_contract.molecule_selection_mask must select at least one molecule"
        )
    if contract.get("selected_molecule_ids") != selected_ids:
        raise LblrtmManifestError("selected_molecule_ids does not match molecule_selection_mask")
    if contract.get("selected_molecule_count") != len(selected_ids):
        raise LblrtmManifestError("selected_molecule_count does not match molecule_selection_mask")
    tape3_linmol = _require_integer(
        contract.get("tape3_linmol"),
        field="scientific_contract.tape3_linmol",
        minimum=1,
    )
    expected_linmol = max(7, max(selected_ids))
    if tape3_linmol != expected_linmol:
        raise LblrtmManifestError("tape3_linmol must equal max(7, highest selected molecule ID)")
    if tape3_linmol > selection_field_width:
        raise LblrtmManifestError("tape3_linmol cannot exceed selection_field_width")
    runtime_count = _require_integer(
        contract.get("runtime_profile_count"),
        field="scientific_contract.runtime_profile_count",
        minimum=1,
    )
    if runtime_count > tape3_linmol:
        raise LblrtmManifestError("runtime_profile_count cannot exceed tape3_linmol")
    line_counts = _require_mapping(
        contract.get("line_count_by_molecule_id"),
        field="scientific_contract.line_count_by_molecule_id",
    )
    expected_line_count_keys = {
        str(molecule_id) for molecule_id in range(1, LNFL_MOLECULE_SLOT_COUNT + 1)
    }
    if set(line_counts) != expected_line_count_keys:
        raise LblrtmManifestError(
            "line_count_by_molecule_id must contain exactly molecule IDs 1..47"
        )
    validated_line_counts = [
        _require_integer(
            line_counts[str(molecule_id)],
            field=(f"scientific_contract.line_count_by_molecule_id.{molecule_id}"),
            minimum=0,
        )
        for molecule_id in range(1, LNFL_MOLECULE_SLOT_COUNT + 1)
    ]
    for molecule_id, (flag, count) in enumerate(
        zip(mask, validated_line_counts, strict=False),
        start=1,
    ):
        if flag == "0" and count != 0:
            raise LblrtmManifestError(
                "line_count_by_molecule_id contains nonzero lines for "
                f"unselected molecule ID {molecule_id}"
            )
    total_line_count = _require_integer(
        contract.get("total_line_count"),
        field="scientific_contract.total_line_count",
        minimum=1,
    )
    if sum(validated_line_counts) != total_line_count:
        raise LblrtmManifestError("total_line_count does not equal the sum of per-molecule counts")
    nonzero_count = _require_integer(
        contract.get("nonzero_molecule_count_in_range"),
        field="scientific_contract.nonzero_molecule_count_in_range",
        minimum=0,
    )
    if sum(count > 0 for count in validated_line_counts) != nonzero_count:
        raise LblrtmManifestError(
            "nonzero_molecule_count_in_range does not match per-molecule counts"
        )

    actual_bounds = contract.get("actual_line_bounds_cm1")
    requested_bounds = contract.get("requested_bounds_cm1")
    wavelength_bounds = contract.get("nominal_wavelength_coverage_nm")
    for name, bounds in (
        ("scientific_contract.requested_bounds_cm1", requested_bounds),
        ("scientific_contract.actual_line_bounds_cm1", actual_bounds),
        (
            "scientific_contract.nominal_wavelength_coverage_nm",
            wavelength_bounds,
        ),
    ):
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bounds)
            or not float(bounds[0]) < float(bounds[1])
        ):
            raise LblrtmManifestError(f"{name} must be a two-number increasing array")
    if float(actual_bounds[0]) < float(requested_bounds[0]) or float(actual_bounds[1]) > float(
        requested_bounds[1]
    ):
        raise LblrtmManifestError("actual_line_bounds_cm1 must lie inside requested_bounds_cm1")
    if (
        float(requested_bounds[1]) - float(requested_bounds[0])
        <= 2.0 * _LBLRTM_REQUIRED_TAPE3_MARGIN_CM1
    ):
        raise LblrtmManifestError(
            "requested_bounds_cm1 must span more than 50 cm-1: LBLRTM needs lines "
            "25 cm-1 beyond each end of a calculation"
        )

    # golden_compatibility and resolved_discrepancies are optional records
    required_sections = (
        "sources",
        "build_contract",
        "products",
        "compatibility",
        "licenses",
        "citations",
    )
    for section_name in required_sections:
        if section_name not in data:
            raise LblrtmManifestError(f"required manifest section is missing: {section_name}")
    sources = _require_mapping(data["sources"], field="sources")
    products = _require_mapping(data["products"], field="products")
    missing_sources = _REQUIRED_SOURCE_RECORDS - set(sources)
    if missing_sources:
        raise LblrtmManifestError(
            "sources is missing required records: " + ", ".join(sorted(missing_sources))
        )
    missing_products = _REQUIRED_PRODUCT_RECORDS - set(products)
    if missing_products:
        raise LblrtmManifestError(
            "products is missing required records: " + ", ".join(sorted(missing_products))
        )
    tape3_record = _require_mapping(
        products["tape3"],
        field="products.tape3",
    )
    for digest_name in ("semantic_sha256", "scientific_sha256"):
        digest = tape3_record.get(digest_name)
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise LblrtmManifestError(
                f"products.tape3.{digest_name} must be 64 lowercase hexadecimal digits"
            )

    build_contract = _require_mapping(
        data["build_contract"],
        field="build_contract",
    )
    if build_contract.get("runtime_distribution_status") not in {
        "host-bound",
        "relocatable",
    }:
        raise LblrtmManifestError(
            "build_contract.runtime_distribution_status must be 'host-bound' or 'relocatable'"
        )
    _validate_manifest_tape12_contract(
        schema_version=schema_version,
        build_contract=build_contract,
        products=products,
    )
    compatibility = _require_mapping(
        data["compatibility"],
        field="compatibility",
    )
    runtime_profile_ids = _require_list(
        compatibility.get("runtime_profile_molecule_ids"),
        field="compatibility.runtime_profile_molecule_ids",
    )
    if (
        len(runtime_profile_ids) != runtime_count
        or any(
            isinstance(molecule_id, bool)
            or not isinstance(molecule_id, int)
            or molecule_id not in selected_ids
            for molecule_id in runtime_profile_ids
        )
        or len(set(runtime_profile_ids)) != len(runtime_profile_ids)
    ):
        raise LblrtmManifestError(
            "compatibility.runtime_profile_molecule_ids must contain exactly "
            "runtime_profile_count unique selected molecule IDs"
        )
    _require_integer(
        compatibility.get("wrapper_name_count"),
        field="compatibility.wrapper_name_count",
        minimum=1,
    )
    golden_compatibility = _require_mapping(
        data.get("golden_compatibility", {}),
        field="golden_compatibility",
    )
    for digest_field in (
        ()
        if not golden_compatibility
        else (
            "reference_manifest_sha256",
            "golden_cases_sha256",
            "reference_tape3_sha256",
            "reference_tape3_semantic_sha256",
            "reference_tape3_scientific_sha256",
        )
    ):
        digest = golden_compatibility.get(digest_field)
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise LblrtmManifestError(
                f"golden_compatibility.{digest_field} must be 64 lowercase hexadecimal digits"
            )
    licenses = _require_mapping(data["licenses"], field="licenses")
    if not licenses:
        raise LblrtmManifestError("licenses must not be empty")
    for license_name, raw_license in licenses.items():
        license_record = _require_mapping(
            raw_license,
            field=f"licenses.{license_name}",
        )
        if "notice_file" not in license_record and "statement" not in license_record:
            raise LblrtmManifestError(
                f"licenses.{license_name} must declare notice_file or statement"
            )
        if "statement" in license_record:
            _require_nonempty_string(
                license_record["statement"],
                field=f"licenses.{license_name}.statement",
            )
        if "redistribution" in license_record:
            _require_nonempty_string(
                license_record["redistribution"],
                field=f"licenses.{license_name}.redistribution",
            )
    _require_list(data["citations"], field="citations")
    _require_list(
        data.get("resolved_discrepancies", []),
        field="resolved_discrepancies",
    )
    validation = (
        _require_mapping(data["validation"], field="validation") if "validation" in data else {}
    )
    if data["status"] == "accepted":
        golden_gate_name = (
            "independent_backend_golden_comparison"
            if schema_version == 2
            else "native_backend_golden_comparison"
        )
        required_acceptance_results = {
            "clean_source_build": "passed",
            "runtime_linkage_portability": "passed",
        }
        if golden_compatibility:
            required_acceptance_results[golden_gate_name] = "passed"
        failed_gates = [
            gate
            for gate, required_result in required_acceptance_results.items()
            if validation.get(gate) != required_result
        ]
        if build_contract["runtime_distribution_status"] != "relocatable":
            failed_gates.append("build_contract.runtime_distribution_status")
        if golden_compatibility and golden_compatibility.get("verification_status") != "passed":
            failed_gates.append("golden_compatibility.verification_status")
        if failed_gates:
            raise LblrtmManifestError(
                "accepted manifest has incomplete acceptance gates: " + ", ".join(failed_gates)
            )

    seen_paths: dict[str, str] = {}
    file_record_count = 0
    for logical_name, record in _iter_file_records(data):
        file_record_count += 1
        relative_path = _require_nonempty_string(
            record.get("relative_path"),
            field=f"{logical_name}.relative_path",
        )
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise LblrtmManifestError(f"{logical_name}.relative_path must remain inside the bundle")
        if relative_path != pure_path.as_posix() or relative_path.startswith("./"):
            raise LblrtmManifestError(
                f"{logical_name}.relative_path is not a normalized POSIX path"
            )
        if relative_path in seen_paths:
            raise LblrtmManifestError(
                f"{logical_name} reuses {relative_path!r}, already declared by "
                f"{seen_paths[relative_path]}"
            )
        seen_paths[relative_path] = logical_name
        _require_integer(
            record.get("size_bytes"),
            field=f"{logical_name}.size_bytes",
            minimum=0,
        )
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise LblrtmManifestError(
                f"{logical_name}.sha256 must be 64 lowercase hexadecimal digits"
            )
        _require_nonempty_string(
            record.get("role"),
            field=f"{logical_name}.role",
        )
        _require_nonempty_string(
            record.get("license_ref"),
            field=f"{logical_name}.license_ref",
        )
        if record["license_ref"] not in licenses:
            raise LblrtmManifestError(
                f"{logical_name}.license_ref names an undeclared license: {record['license_ref']!r}"
            )
    if file_record_count < len(_REQUIRED_SOURCE_RECORDS | _REQUIRED_PRODUCT_RECORDS):
        raise LblrtmManifestError("manifest declares too few managed files")


def load_lblrtm_resource_manifest(
    manifest_path: str | Path | None = None,
) -> LblrtmResourceManifest:
    """Read and validate an independent LBLRTM resource manifest."""

    path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(__file__).with_name(DEFAULT_MANIFEST_FILENAME)
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LblrtmManifestError(f"cannot read manifest {path}: {exc}") from exc
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise LblrtmManifestError(f"invalid JSON in manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise LblrtmManifestError("manifest top level must be a JSON object")
    validate_lblrtm_resource_manifest(data)
    return LblrtmResourceManifest(
        path=path.resolve(),
        _serialized_data=json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        _resource_set_id=str(data["resource_set_id"]),
        _status=str(data["status"]),
        fingerprint=compute_lblrtm_resource_fingerprint(data),
    )


def resolve_lblrtm_resource_files(
    manifest: LblrtmResourceManifest,
    resource_root: str | Path,
) -> ResolvedLblrtmResourceFiles:
    """Resolve every declared file without permitting path or symlink escapes."""

    root = Path(resource_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise LblrtmResourceIntegrityError(f"resource root is not a directory: {root}")
    resolved: dict[str, Path] = {}
    for logical_name, record in _iter_file_records(manifest.data):
        relative_path = str(record["relative_path"])
        candidate = root.joinpath(*PurePosixPath(relative_path).parts)
        relative_parts = PurePosixPath(relative_path).parts
        path_cursor = root
        symlink_component: Path | None = None
        for part in relative_parts:
            path_cursor = path_cursor / part
            if path_cursor.is_symlink():
                symlink_component = path_cursor
                break
        if symlink_component is not None:
            raise LblrtmResourceIntegrityError(
                "an LBLRTM bundle must not contain symbolic links: "
                f"{symlink_component.relative_to(root)}"
            )
        try:
            actual = candidate.resolve(strict=True)
        except OSError as exc:
            raise LblrtmResourceIntegrityError(
                f"declared resource is missing: {relative_path}"
            ) from exc
        if root not in actual.parents:
            raise LblrtmResourceIntegrityError(
                f"declared resource escapes bundle root: {relative_path}"
            )
        if not actual.is_file():
            raise LblrtmResourceIntegrityError(
                f"declared resource is not a regular file: {relative_path}"
            )
        resolved[logical_name] = actual
    return ResolvedLblrtmResourceFiles(
        root=root,
        files=MappingProxyType(resolved),
        manifest_fingerprint=manifest.fingerprint,
    )


def compute_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading large TAPE3 or line-list assets into memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lnfl_tape5(path: str | Path) -> LnflTape5Summary:
    """Parse the three-line LNFL v3.2 TAPE5 that gtellcorr uses."""

    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LblrtmResourceIntegrityError(f"cannot read LNFL TAPE5: {exc}") from exc
    if len(lines) < 3:
        raise LblrtmResourceIntegrityError(
            "LNFL TAPE5 must contain title, bounds, and 47-slot mask lines"
        )
    bounds_match = _TAPE5_BOUNDS_PATTERN.fullmatch(lines[1])
    if bounds_match is None:
        raise LblrtmResourceIntegrityError("LNFL TAPE5 line 2 must contain exactly VMIN and VMAX")
    try:
        mask = validate_lnfl_molecule_selection_mask(lines[2])
    except (TypeError, ValueError) as exc:
        raise LblrtmResourceIntegrityError(f"invalid LNFL TAPE5 molecule mask: {exc}") from exc
    requested_vmin_cm1 = float(bounds_match.group(1))
    requested_vmax_cm1 = float(bounds_match.group(2))
    if not requested_vmin_cm1 < requested_vmax_cm1:
        raise LblrtmResourceIntegrityError("LNFL TAPE5 VMIN must be lower than VMAX")
    return LnflTape5Summary(
        requested_vmin_cm1=requested_vmin_cm1,
        requested_vmax_cm1=requested_vmax_cm1,
        molecule_selection_mask=mask,
    )


def parse_lnfl_tape6(path: str | Path) -> LnflTape6Summary:
    """Parse LNFL's selection echo and per-molecule line-count report."""

    try:
        lines = Path(path).read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LblrtmResourceIntegrityError(f"cannot read LNFL TAPE6: {exc}") from exc

    registry = load_lnfl_molecule_registry()
    selections: list[int] = []
    selection_names: list[str] = []
    line_counts: list[int] = []
    line_count_names: list[str] = []
    bounds: tuple[float, float, int] | None = None
    saw_tape_number = False
    saw_table_header = False
    for line in lines:
        if "TAPE NO." in line:
            saw_tape_number = True
        if not saw_tape_number:
            selection_match = _TAPE6_SELECTION_PATTERN.fullmatch(line)
            if selection_match is not None:
                selection_names.append(selection_match.group(1))
                selections.append(int(selection_match.group(2)))
            continue
        bounds_match = _TAPE6_BOUNDS_PATTERN.search(line)
        if bounds_match is not None:
            bounds = (
                float(bounds_match.group(1)),
                float(bounds_match.group(2)),
                int(bounds_match.group(3)),
            )
        if "MOL" in line and "LINES" in line and "STRENGTH" in line:
            saw_table_header = True
            continue
        if saw_table_header and len(line_counts) < LNFL_MOLECULE_SLOT_COUNT:
            count_match = _TAPE6_LINE_COUNT_PATTERN.fullmatch(line)
            if count_match is not None:
                line_count_names.append(count_match.group(1))
                line_counts.append(int(count_match.group(2)))

    reported_slot_count = len(selections)
    if not 7 <= reported_slot_count <= LNFL_MOLECULE_SLOT_COUNT:
        raise LblrtmResourceIntegrityError(
            f"LNFL TAPE6 must echo 7..47 molecule selection flags; found {reported_slot_count}"
        )
    if bounds is None:
        raise LblrtmResourceIntegrityError("LNFL TAPE6 is missing the line-range summary")
    if len(line_counts) != reported_slot_count:
        raise LblrtmResourceIntegrityError(
            "LNFL TAPE6 line-count row count must match its selection echo; "
            f"found {len(line_counts)} counts for {reported_slot_count} flags"
        )
    expected_names = [molecule.formula.upper() for molecule in registry]
    if [name.upper() for name in selection_names] != expected_names[:reported_slot_count]:
        raise LblrtmResourceIntegrityError(
            "LNFL TAPE6 molecule selection names are not in registry order"
        )
    if [name.upper() for name in line_count_names] != expected_names[:reported_slot_count]:
        raise LblrtmResourceIntegrityError("LNFL TAPE6 line-count names are not in registry order")
    mask = validate_lnfl_molecule_selection_mask(
        "".join(str(flag) for flag in selections)
        + ("0" * (LNFL_MOLECULE_SLOT_COUNT - reported_slot_count))
    )
    padded_line_counts = line_counts + ([0] * (LNFL_MOLECULE_SLOT_COUNT - reported_slot_count))
    return LnflTape6Summary(
        molecule_selection_mask=mask,
        reported_slot_count=reported_slot_count,
        actual_vmin_cm1=bounds[0],
        actual_vmax_cm1=bounds[1],
        total_line_count=bounds[2],
        line_count_by_molecule_id=MappingProxyType(
            {
                molecule.molecule_id: padded_line_counts[molecule.molecule_id - 1]
                for molecule in registry
            }
        ),
    )


def _assert_float_close(
    actual: float,
    expected: float,
    *,
    field: str,
    absolute_tolerance: float = 5e-4,
) -> None:
    """Check that *actual* is within *absolute_tolerance* of *expected*."""
    if abs(actual - expected) > absolute_tolerance:
        raise LblrtmResourceIntegrityError(f"{field} mismatch: expected {expected}, got {actual}")


def _require_recipe_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    """*value*, which must be a JSON object of the build recipe."""
    if not isinstance(value, Mapping):
        raise LblrtmResourceIntegrityError(f"build recipe {field} must be a JSON object")
    return value


def _require_recipe_string(value: Any, *, field: str) -> str:
    """*value*, which must be a non-empty string of the build recipe."""
    if not isinstance(value, str) or not value.strip():
        raise LblrtmResourceIntegrityError(f"build recipe {field} must be a non-empty string")
    return value


def _require_recipe_string_list(value: Any, *, field: str) -> list[str]:
    """*value*, which must be a list of non-empty strings."""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise LblrtmResourceIntegrityError(
            f"build recipe {field} must be an array of non-empty strings"
        )
    return value


def _validate_recipe_sha256(value: Any, *, field: str) -> str:
    """*value*, which must be 64 lowercase hexadecimal digits."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise LblrtmResourceIntegrityError(
            f"build recipe {field} must be 64 lowercase hexadecimal digits"
        )
    return value


def _validate_v2_recipe_build_contract(
    recipe: Mapping[str, Any],
    manifest_data: Mapping[str, Any],
) -> Tape12AbiContract:
    """Check every v2 recipe precision and TAPE12 layout against the manifest products."""

    builder = _require_recipe_string(
        recipe.get("builder"),
        field="builder",
    )
    if builder not in {
        "tools/lbl/build_lblrtm_resource_bundle.py",
        "gtellcorr/telluric/resources/_build.py",
    }:
        raise LblrtmResourceIntegrityError(
            "the build recipe names a different build script than the manifest expects"
        )
    _validate_recipe_sha256(
        recipe.get("builder_sha256"),
        field="builder_sha256",
    )

    scalar_fields = (
        "lnfl_precision",
        "lblrtm_precision",
        "lnfl_make_target",
        "lblrtm_make_target",
        "lnfl_product_basename",
        "lblrtm_product_basename",
    )
    manifest_build_contract = _require_mapping(
        manifest_data.get("build_contract"),
        field="build_contract",
    )
    for field_name in scalar_fields:
        recipe_value = _require_recipe_string(
            recipe.get(field_name),
            field=field_name,
        )
        if recipe_value != manifest_build_contract.get(field_name):
            raise LblrtmResourceIntegrityError(
                f"build recipe {field_name} differs from build_contract in the manifest"
            )

    raw_recipe_abi = _require_recipe_mapping(
        recipe.get("tape12_abi"),
        field="tape12_abi",
    )
    expected_abi_fields = {
        "byte_order",
        "record_marker_bytes",
        "real_bytes",
        "integer_bytes",
    }
    if set(raw_recipe_abi) != expected_abi_fields:
        raise LblrtmResourceIntegrityError(
            "build recipe tape12_abi must contain exactly: "
            + ", ".join(sorted(expected_abi_fields))
        )
    try:
        recipe_abi = Tape12AbiContract(
            byte_order=raw_recipe_abi.get("byte_order"),
            record_marker_bytes=raw_recipe_abi.get("record_marker_bytes"),
            real_bytes=raw_recipe_abi.get("real_bytes"),
            integer_bytes=raw_recipe_abi.get("integer_bytes"),
        )
        manifest_abi = _tape12_abi_from_manifest_data(manifest_data)
    except (LblrtmManifestError, TypeError, ValueError) as exc:
        raise LblrtmResourceIntegrityError(
            f"invalid tape12_abi (TAPE12 binary layout) in the build recipe: {exc}"
        ) from exc
    if recipe_abi != manifest_abi:
        raise LblrtmResourceIntegrityError(
            "build recipe tape12_abi differs from build_contract in the manifest"
        )

    products = _require_mapping(
        manifest_data.get("products"),
        field="products",
    )
    for recipe_field, product_name in (
        ("lnfl_product_basename", "lnfl_executable"),
        ("lblrtm_product_basename", "lblrtm_executable"),
    ):
        product_record = _require_mapping(
            products.get(product_name),
            field=f"products.{product_name}",
        )
        product_path = _require_nonempty_string(
            product_record.get("relative_path"),
            field=f"products.{product_name}.relative_path",
        )
        if PurePosixPath(product_path).name != recipe[recipe_field]:
            raise LblrtmResourceIntegrityError(
                f"build recipe {recipe_field} differs from manifest product {product_name} basename"
            )
    return recipe_abi


def validate_lblrtm_build_recipe(
    path: str | Path,
    manifest_data: Mapping[str, Any],
    *,
    expected_requested_bounds_cm1: tuple[float, float] | None = None,
    expected_molecule_selection_mask: str | None = None,
    require_manifest_distribution_status_match: bool = True,
) -> Mapping[str, Any]:
    """Check a build recipe against the manifest of its bundle.

    The optional expected bounds and mask are used by the freeze tool while it
    measures a bundle whose physics settings may differ from the template.
    The freeze tool also disables the manifest distribution-status comparison
    because it derives that status from the recipe before rewriting the
    template. Normal bundle validation keeps all comparisons enabled.
    """

    recipe_path = Path(path)
    try:
        text = recipe_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LblrtmResourceIntegrityError(
            f"cannot read build recipe {recipe_path}: {exc}"
        ) from exc
    try:
        recipe = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, LblrtmManifestError) as exc:
        raise LblrtmResourceIntegrityError(
            f"invalid JSON in build recipe {recipe_path}: {exc}"
        ) from exc
    if not isinstance(recipe, Mapping):
        raise LblrtmResourceIntegrityError("build recipe top level must be a JSON object")
    recipe_schema_version = recipe.get("schema_version")
    if isinstance(recipe_schema_version, bool) or recipe_schema_version not in {1, 2}:
        raise LblrtmResourceIntegrityError("build recipe schema_version must be 1 or 2")
    manifest_schema_version = manifest_data.get("schema_version")
    if recipe_schema_version != manifest_schema_version:
        raise LblrtmResourceIntegrityError(
            "build recipe schema_version differs from manifest schema_version"
        )
    if recipe_schema_version == 2:
        _validate_v2_recipe_build_contract(recipe, manifest_data)

    sources = _require_mapping(
        manifest_data.get("sources"),
        field="sources",
    )
    recipe_source_sha256 = _require_recipe_mapping(
        recipe.get("source_sha256"),
        field="source_sha256",
    )
    expected_source_sha256: dict[str, Any] = {}
    for logical_name, raw_record in sources.items():
        source_record = _require_mapping(
            raw_record,
            field=f"sources.{logical_name}",
        )
        expected_source_sha256[logical_name] = source_record["sha256"]
    if set(recipe_source_sha256) != set(expected_source_sha256):
        raise LblrtmResourceIntegrityError(
            "build recipe source_sha256 keys differ from manifest sources"
        )
    for logical_name, expected_digest in expected_source_sha256.items():
        actual_digest = _validate_recipe_sha256(
            recipe_source_sha256[logical_name],
            field=f"source_sha256.{logical_name}",
        )
        if actual_digest != expected_digest:
            raise LblrtmResourceIntegrityError(
                f"build recipe source_sha256 differs from manifest for {logical_name}"
            )

    contract = _require_mapping(
        manifest_data.get("scientific_contract"),
        field="scientific_contract",
    )
    expected_bounds = (
        expected_requested_bounds_cm1
        if expected_requested_bounds_cm1 is not None
        else (
            float(contract["requested_bounds_cm1"][0]),
            float(contract["requested_bounds_cm1"][1]),
        )
    )
    recipe_bounds = recipe.get("requested_bounds_cm1")
    if (
        not isinstance(recipe_bounds, list)
        or len(recipe_bounds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in recipe_bounds
        )
    ):
        raise LblrtmResourceIntegrityError(
            "build recipe requested_bounds_cm1 must be a two-number array"
        )
    _assert_float_close(
        float(recipe_bounds[0]),
        float(expected_bounds[0]),
        field="build recipe requested VMIN",
    )
    _assert_float_close(
        float(recipe_bounds[1]),
        float(expected_bounds[1]),
        field="build recipe requested VMAX",
    )

    expected_mask = (
        expected_molecule_selection_mask
        if expected_molecule_selection_mask is not None
        else contract["molecule_selection_mask"]
    )
    try:
        recipe_mask = validate_lnfl_molecule_selection_mask(recipe.get("molecule_selection_mask"))
        validated_expected_mask = validate_lnfl_molecule_selection_mask(expected_mask)
    except (TypeError, ValueError) as exc:
        raise LblrtmResourceIntegrityError(
            f"invalid build recipe molecule selection: {exc}"
        ) from exc
    if recipe_mask != validated_expected_mask:
        raise LblrtmResourceIntegrityError(
            "build recipe molecule_selection_mask differs from scientific_contract in the manifest"
        )

    runtime_linkage = _require_recipe_mapping(
        recipe.get("runtime_linkage"),
        field="runtime_linkage",
    )
    expected_executable_names = {
        "lnfl_executable",
        "lblrtm_executable",
    }
    if set(runtime_linkage) != expected_executable_names:
        raise LblrtmResourceIntegrityError(
            "build recipe runtime_linkage must contain exactly "
            "lnfl_executable and lblrtm_executable"
        )
    products = _require_mapping(
        manifest_data.get("products"),
        field="products",
    )
    executable_relocatability: list[bool] = []
    linked_libraries_by_executable: dict[str, list[str]] = {}
    for executable_name in sorted(expected_executable_names):
        linkage_record = _require_recipe_mapping(
            runtime_linkage[executable_name],
            field=f"runtime_linkage.{executable_name}",
        )
        product_record = _require_mapping(
            products.get(executable_name),
            field=f"products.{executable_name}",
        )
        recipe_executable_size = linkage_record.get("size_bytes")
        if (
            isinstance(recipe_executable_size, bool)
            or not isinstance(recipe_executable_size, int)
            or recipe_executable_size < 1
        ):
            raise LblrtmResourceIntegrityError(
                "build recipe runtime_linkage."
                f"{executable_name}.size_bytes must be a positive integer"
            )
        if recipe_executable_size != product_record.get("size_bytes"):
            raise LblrtmResourceIntegrityError(
                f"build recipe executable size differs from manifest product for {executable_name}"
            )
        recipe_executable_sha256 = _validate_recipe_sha256(
            linkage_record.get("sha256"),
            field=f"runtime_linkage.{executable_name}.sha256",
        )
        if recipe_executable_sha256 != product_record.get("sha256"):
            raise LblrtmResourceIntegrityError(
                "build recipe executable SHA-256 differs from manifest "
                f"product for {executable_name}"
            )
        linked_libraries = _require_recipe_string_list(
            linkage_record.get("linked_libraries"),
            field=f"runtime_linkage.{executable_name}.linked_libraries",
        )
        linked_libraries_by_executable[executable_name] = linked_libraries
        lc_rpaths = _require_recipe_string_list(
            linkage_record.get("lc_rpaths"),
            field=f"runtime_linkage.{executable_name}.lc_rpaths",
        )
        absolute_rpaths = _require_recipe_string_list(
            linkage_record.get("absolute_rpaths"),
            field=f"runtime_linkage.{executable_name}.absolute_rpaths",
        )
        non_system_absolute_libraries = _require_recipe_string_list(
            linkage_record.get("non_system_absolute_libraries"),
            field=(f"runtime_linkage.{executable_name}.non_system_absolute_libraries"),
        )
        expected_absolute_rpaths = [rpath for rpath in lc_rpaths if rpath.startswith("/")]
        if absolute_rpaths != expected_absolute_rpaths:
            raise LblrtmResourceIntegrityError(
                f"build recipe absolute_rpaths does not match lc_rpaths for {executable_name}"
            )
        expected_non_system_libraries = [
            library
            for library in linked_libraries
            if library.startswith("/") and not library.startswith(("/usr/lib/", "/System/Library/"))
        ]
        if non_system_absolute_libraries != expected_non_system_libraries:
            raise LblrtmResourceIntegrityError(
                "build recipe non_system_absolute_libraries does not match "
                f"linked_libraries for {executable_name}"
            )
        relocatable = linkage_record.get("relocatable")
        if not isinstance(relocatable, bool):
            raise LblrtmResourceIntegrityError(
                f"build recipe runtime_linkage.{executable_name}.relocatable must be boolean"
            )
        derived_relocatable = not (expected_absolute_rpaths or expected_non_system_libraries)
        if relocatable != derived_relocatable:
            raise LblrtmResourceIntegrityError(
                f"build recipe relocatable flag contradicts linkage paths for {executable_name}"
            )
        executable_relocatability.append(relocatable)

    derived_distribution_status = "relocatable" if all(executable_relocatability) else "host-bound"
    recipe_distribution_status = recipe.get("runtime_distribution_status")
    if recipe_distribution_status != derived_distribution_status:
        raise LblrtmResourceIntegrityError(
            "build recipe runtime_distribution_status contradicts executable linkage"
        )
    manifest_build_contract = _require_mapping(
        manifest_data.get("build_contract"),
        field="build_contract",
    )
    manifest_distribution_status = manifest_build_contract.get("runtime_distribution_status")
    if (
        require_manifest_distribution_status_match
        and manifest_distribution_status in {"host-bound", "relocatable"}
        and manifest_distribution_status != derived_distribution_status
    ):
        raise LblrtmResourceIntegrityError(
            "build recipe runtime_distribution_status differs from build_contract in the manifest"
        )

    netcdf_provenance = _require_recipe_mapping(
        recipe.get("netcdf_provenance"),
        field="netcdf_provenance",
    )
    configuration_tools = _require_recipe_mapping(
        netcdf_provenance.get("configuration_tools"),
        field="netcdf_provenance.configuration_tools",
    )
    missing_configuration_tools = _require_recipe_string_list(
        netcdf_provenance.get("missing_configuration_tools"),
        field="netcdf_provenance.missing_configuration_tools",
    )
    if set(configuration_tools) & set(missing_configuration_tools):
        raise LblrtmResourceIntegrityError(
            "build recipe NetCDF configuration tools cannot be both present and missing"
        )
    if set(configuration_tools) | set(missing_configuration_tools) != {"nc-config", "nf-config"}:
        raise LblrtmResourceIntegrityError(
            "the build recipe must record where nc-config and nf-config (NetCDF) came from"
        )
    for tool_name, raw_tool_record in configuration_tools.items():
        tool_record = _require_recipe_mapping(
            raw_tool_record,
            field=f"netcdf_provenance.configuration_tools.{tool_name}",
        )
        _validate_recipe_sha256(
            tool_record.get("sha256"),
            field=(f"netcdf_provenance.configuration_tools.{tool_name}.sha256"),
        )
        _require_recipe_string(
            tool_record.get("version"),
            field=(f"netcdf_provenance.configuration_tools.{tool_name}.version"),
        )
    linked_library_files = _require_recipe_mapping(
        netcdf_provenance.get("linked_library_files"),
        field="netcdf_provenance.linked_library_files",
    )
    if not linked_library_files:
        raise LblrtmResourceIntegrityError(
            "build recipe NetCDF linked_library_files must not be empty"
        )
    expected_netcdf_install_names = {
        library
        for library in linked_libraries_by_executable["lblrtm_executable"]
        if PurePosixPath(library).name.startswith(("libnetcdf.", "libnetcdff."))
    }
    if set(linked_library_files) != expected_netcdf_install_names:
        raise LblrtmResourceIntegrityError(
            "build recipe NetCDF linked_library_files differs from LBLRTM runtime linkage"
        )
    for install_name, raw_library_record in linked_library_files.items():
        _require_recipe_string(
            install_name,
            field="netcdf_provenance.linked_library_files key",
        )
        library_record = _require_recipe_mapping(
            raw_library_record,
            field=f"netcdf_provenance.linked_library_files.{install_name}",
        )
        for basename_field in ("requested_basename", "resolved_basename"):
            _require_recipe_string(
                library_record.get(basename_field),
                field=(f"netcdf_provenance.linked_library_files.{install_name}.{basename_field}"),
            )
        if library_record["requested_basename"] != PurePosixPath(install_name).name:
            raise LblrtmResourceIntegrityError(
                "build recipe NetCDF requested_basename differs from its "
                f"install name: {install_name}"
            )
        size_bytes = library_record.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise LblrtmResourceIntegrityError(
                "build recipe NetCDF library size_bytes must be a positive "
                f"integer for {install_name}"
            )
        _validate_recipe_sha256(
            library_record.get("sha256"),
            field=(f"netcdf_provenance.linked_library_files.{install_name}.sha256"),
        )

    return MappingProxyType(dict(recipe))


def validate_lblrtm_resource_bundle(
    manifest: LblrtmResourceManifest,
    resource_root: str | Path,
) -> ResolvedLblrtmResourceFiles:
    """Validate hashes, control files, TAPE3 metadata, and cross-file semantics."""

    if not isinstance(manifest, LblrtmResourceManifest):
        raise TypeError("internal error: manifest must be an LblrtmResourceManifest")
    manifest_data = manifest.data
    validate_lblrtm_resource_manifest(manifest_data)
    computed_fingerprint = compute_lblrtm_resource_fingerprint(manifest_data)
    if manifest.fingerprint != computed_fingerprint:
        raise LblrtmResourceIntegrityError(
            "the manifest was changed after its checksum was recorded: "
            f"expected {computed_fingerprint}, got "
            f"{manifest.fingerprint}"
        )
    if manifest.resource_set_id != manifest_data["resource_set_id"]:
        raise LblrtmResourceIntegrityError(
            "internal error: manifest resource_set_id does not match the manifest file"
        )
    if manifest.status != manifest_data["status"]:
        raise LblrtmResourceIntegrityError(
            "internal error: manifest status does not match the manifest file"
        )

    resolved = resolve_lblrtm_resource_files(manifest, resource_root)
    for logical_name, record in _iter_file_records(manifest_data):
        path = resolved.require(logical_name)
        size = path.stat().st_size
        if size != record["size_bytes"]:
            raise LblrtmResourceIntegrityError(
                f"{logical_name} size mismatch: expected {record['size_bytes']}, got {size}"
            )
        digest = compute_file_sha256(path)
        if digest != record["sha256"]:
            raise LblrtmResourceIntegrityError(
                f"{logical_name} SHA-256 mismatch: expected {record['sha256']}, got {digest}"
            )

    contract = manifest_data["scientific_contract"]
    tape5 = parse_lnfl_tape5(resolved.require("products.lnfl_tape5"))
    validate_lblrtm_build_recipe(
        resolved.require("products.build_recipe"),
        manifest_data,
        expected_requested_bounds_cm1=(
            tape5.requested_vmin_cm1,
            tape5.requested_vmax_cm1,
        ),
        expected_molecule_selection_mask=tape5.molecule_selection_mask,
    )
    tape6 = parse_lnfl_tape6(resolved.require("products.lnfl_tape6"))
    if tape5.molecule_selection_mask != contract["molecule_selection_mask"]:
        raise LblrtmResourceIntegrityError(
            "TAPE5 molecule selection differs from the one in the manifest"
        )
    if tape6.molecule_selection_mask != tape5.molecule_selection_mask:
        raise LblrtmResourceIntegrityError("TAPE6 selection echo differs from TAPE5")
    _assert_float_close(
        tape5.requested_vmin_cm1,
        float(contract["requested_bounds_cm1"][0]),
        field="TAPE5 requested VMIN",
    )
    _assert_float_close(
        tape5.requested_vmax_cm1,
        float(contract["requested_bounds_cm1"][1]),
        field="TAPE5 requested VMAX",
    )
    _assert_float_close(
        tape6.actual_vmin_cm1,
        float(contract["actual_line_bounds_cm1"][0]),
        field="TAPE6 actual VMIN",
    )
    _assert_float_close(
        tape6.actual_vmax_cm1,
        float(contract["actual_line_bounds_cm1"][1]),
        field="TAPE6 actual VMAX",
    )
    if tape6.total_line_count != contract["total_line_count"]:
        raise LblrtmResourceIntegrityError("the line count in TAPE6 differs from the manifest")
    expected_counts = {
        int(key): int(value) for key, value in contract["line_count_by_molecule_id"].items()
    }
    if dict(tape6.line_count_by_molecule_id) != expected_counts:
        raise LblrtmResourceIntegrityError(
            "the per-molecule line counts in TAPE6 differ from the manifest"
        )
    if tape6.reported_slot_count != contract["tape3_linmol"]:
        raise LblrtmResourceIntegrityError(
            "TAPE6 reported slot count differs from manifest tape3_linmol"
        )
    if len(tape6.nonzero_molecule_ids) != contract["nonzero_molecule_count_in_range"]:
        raise LblrtmResourceIntegrityError(
            "the number of molecules with lines in TAPE6 differs from the manifest"
        )

    tape3_record = manifest_data["products"]["tape3"]
    tape3_path = resolved.require("products.tape3")
    try:
        tape3_header = read_tape3_header(tape3_path)
        tape3_linmol = int(contract["tape3_linmol"])
        validate_tape3_metadata(
            tape3_header,
            expected_linmol=tape3_linmol,
            expected_actual_line_bounds_cm_minus_1=(
                float(contract["actual_line_bounds_cm1"][0]),
                float(contract["actual_line_bounds_cm1"][1]),
            ),
            bounds_tolerance_cm_minus_1=5e-4,
            expected_total_line_count=contract["total_line_count"],
            expected_molecule_line_counts={
                int(key): int(value)
                for key, value in contract["line_count_by_molecule_id"].items()
                if int(key) <= tape3_linmol
            },
        )
        expected_names = tuple(
            molecule.formula.upper() for molecule in load_lnfl_molecule_registry()[:tape3_linmol]
        )
        actual_names = tuple(name.upper() for name in tape3_header.molecule_names)
        if actual_names != expected_names:
            raise LblrtmResourceIntegrityError(
                "TAPE3 molecule names are not in the LNFL v3.2 registry order"
            )
        semantic_digest = compute_tape3_semantic_sha256(tape3_path)
        scientific_digest = compute_tape3_scientific_sha256(tape3_path)
    except (Tape3FormatError, Tape3ValidationError) as exc:
        raise LblrtmResourceIntegrityError(f"TAPE3 metadata validation failed: {exc}") from exc
    if semantic_digest != tape3_record["semantic_sha256"]:
        raise LblrtmResourceIntegrityError(
            "the TAPE3 line file differs from the manifest (SHA-256 without the LNFL "
            "date and time): expected "
            f"{tape3_record['semantic_sha256']}, got {semantic_digest}"
        )
    if scientific_digest != tape3_record["scientific_sha256"]:
        raise LblrtmResourceIntegrityError(
            "the TAPE3 line data differ from the manifest (SHA-256 without the LNFL "
            "date, time and title): expected "
            f"{tape3_record['scientific_sha256']}, got {scientific_digest}"
        )
    validated = ResolvedLblrtmResourceFiles(
        root=resolved.root,
        files=resolved.files,
        manifest_fingerprint=manifest.fingerprint,
        validation_evidence=MappingProxyType(
            {
                "validator": _BUNDLE_VALIDATOR_ID,
                "manifest_fingerprint": manifest.fingerprint,
                "resource_set_id": manifest.resource_set_id,
                "scope": (
                    "declared-file-hashes+build-contract+tape12-abi+"
                    "lnfl-controls+tape3-scientific-metadata+"
                    "runtime-file-identities+validated-spectral-support"
                ),
                **_runtime_resource_validation_evidence(manifest_data),
            }
        ),
    )
    object.__setattr__(
        validated,
        "_validation_token",
        _VALIDATED_RESOURCE_BUNDLE_TOKEN,
    )
    return validated
