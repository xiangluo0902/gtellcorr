#!/usr/bin/env python3
"""Freeze an assembled LBLRTM bundle into an immutable manifest.

This command performs no download.
All paths are resolved from the supplied bundle root and manifest template.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

_LBLRTM_PACKAGE = Path(__file__).resolve().parents[1] / "lblrtm"

# Build targets and product names follow the host platform (macOS or Linux).
_HOST_SYSTEM = __import__("platform").system()
_TARGET_PREFIX = "linux" if _HOST_SYSTEM == "Linux" else "osx"
_PRODUCT_TAG = "linux" if _HOST_SYSTEM == "Linux" else "OS_X"

from gtellcorr.telluric.lblrtm.lblrtm_resources import (  # noqa: E402
    DEFAULT_MANIFEST_FILENAME,
    LblrtmResourceIntegrityError,
    compute_file_sha256,
    compute_lblrtm_resource_fingerprint,
    load_lblrtm_resource_manifest,
    parse_lnfl_tape5,
    parse_lnfl_tape6,
    validate_lblrtm_build_recipe,
    validate_lblrtm_resource_manifest,
)
from gtellcorr.telluric.lblrtm.lnfl_molecule_selection import (  # noqa: E402
    load_lnfl_molecule_registry,
)
from gtellcorr.telluric.lblrtm.tape3_metadata import (  # noqa: E402
    compute_tape3_scientific_sha256,
    compute_tape3_semantic_sha256,
    read_tape3_header,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Command-line options of the freezer."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute every LBLRTM file hash and scientific summary "
            "from an already assembled resource bundle."
        )
    )
    parser.add_argument(
        "--resource-root",
        required=True,
        type=Path,
        help="Bundle root containing bin/, data/, sources/, and provenance/.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=(_LBLRTM_PACKAGE / DEFAULT_MANIFEST_FILENAME),
        help="Manifest whose relative paths and descriptive metadata are retained.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON. Required unless --check is used.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare computed content with --template instead of writing JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing --output file.",
    )
    return parser


def _iter_file_records(
    data: Mapping[str, Any],
) -> Iterable[tuple[str, dict[str, Any]]]:
    """``(dotted name, record)`` of every file of the manifest: sources, products
    and licence notices."""
    for section_name in ("sources", "products"):
        section = data[section_name]
        for logical_name, record in section.items():
            yield f"{section_name}.{logical_name}", record
    for license_name, license_record in data["licenses"].items():
        if "notice_file" in license_record:
            yield (
                f"licenses.{license_name}.notice_file",
                license_record["notice_file"],
            )


def _resolve_bundle_file(root: Path, relative_path: str) -> Path:
    """A manifest path inside *root*: relative, no ``..``, no symlinks on the way."""
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise LblrtmResourceIntegrityError(
            f"manifest path {relative_path!r} must stay inside the bundle"
        )
    candidate = root.joinpath(*pure_path.parts)
    path_cursor = root
    for part in pure_path.parts:
        path_cursor = path_cursor / part
        if path_cursor.is_symlink():
            raise LblrtmResourceIntegrityError(
                f"an LBLRTM bundle must not contain symbolic links: {path_cursor.relative_to(root)}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LblrtmResourceIntegrityError(f"declared file is missing: {relative_path}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise LblrtmResourceIntegrityError(
            f"declared file is outside the resource root: {relative_path}"
        )
    return resolved


_EXPECTED_LBLRTM_BUILD_CONTRACTS = {
    "single": {
        "make_target": f"{_TARGET_PREFIX}GNUsgl",
        "product_basename": f"lblrtm_v12.17_{_PRODUCT_TAG}_gnu_sgl",
        "tape12_abi": {
            "byte_order": "little",
            "record_marker_bytes": 4,
            "real_bytes": 4,
            "integer_bytes": 4,
        },
    },
    "double": {
        "make_target": f"{_TARGET_PREFIX}GNUdbl",
        "product_basename": f"lblrtm_v12.17_{_PRODUCT_TAG}_gnu_dbl",
        "tape12_abi": {
            "byte_order": "little",
            "record_marker_bytes": 4,
            "real_bytes": 8,
            "integer_bytes": 8,
        },
    },
}
_EXPECTED_LNFL_BUILD_CONTRACT = {
    "precision": "single",
    "make_target": f"{_TARGET_PREFIX}GNUsgl",
    "product_basename": f"lnfl_v3.2_{_PRODUCT_TAG}_gnu_sgl",
}
# Builders whose recipes are accepted, with their expected SHA-256: the builder
# script of the original implementation (which built the existing bundles) and
# this package's _build.py.
_LEGACY_BUILDER_SHA256 = "4b4f2557438065a82369da8d8addeb27296af84bc91b882668e87f800a766fa0"
_KNOWN_BUILDERS = {
    "tools/lbl/build_lblrtm_resource_bundle.py": lambda: _LEGACY_BUILDER_SHA256,
    "gtellcorr/telluric/resources/_build.py": lambda: compute_file_sha256(
        Path(__file__).with_name("_build.py")
    ),
}


def _require_recipe_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    """*value*, which must be a JSON object."""
    if not isinstance(value, Mapping):
        raise LblrtmResourceIntegrityError(f"build recipe {field} must be a JSON object")
    return value


def _manifest_product_basename(
    data: Mapping[str, Any],
    *,
    logical_name: str,
) -> str:
    """File name of product *logical_name* in the manifest."""
    products = _require_recipe_mapping(
        data.get("products"),
        field="manifest products",
    )
    product = _require_recipe_mapping(
        products.get(logical_name),
        field=f"manifest products.{logical_name}",
    )
    relative_path = product.get("relative_path")
    if not isinstance(relative_path, str):
        raise LblrtmResourceIntegrityError(
            f"manifest products.{logical_name}.relative_path must be a string"
        )
    return PurePosixPath(relative_path).name


def _require_recipe_contract_value(
    recipe: Mapping[str, Any],
    *,
    field: str,
    expected: Any,
) -> Any:
    """``recipe[field]``, which must equal *expected*."""
    actual = recipe.get(field)
    if actual != expected:
        raise LblrtmResourceIntegrityError(
            f"build recipe {field} mismatch: expected {expected!r}, got {actual!r}"
        )
    return actual


def _validate_v2_build_contract(
    recipe: Mapping[str, Any],
    data: Mapping[str, Any],
    resolved: Mapping[str, Path],
) -> None:
    """Check a schema-2 build recipe: a known builder with its hash, and every
    recorded build setting against the manifest and the bundle files."""
    builder_identity = recipe.get("builder")
    if builder_identity not in _KNOWN_BUILDERS:
        raise LblrtmResourceIntegrityError(
            "build recipe builder is not a known gtellcorr resource builder"
        )
    builder_sha256 = recipe.get("builder_sha256")
    if (
        not isinstance(builder_sha256, str)
        or len(builder_sha256) != 64
        or any(character not in "0123456789abcdef" for character in builder_sha256)
    ):
        raise LblrtmResourceIntegrityError(
            "build recipe builder_sha256 must be 64 lowercase hexadecimal digits"
        )
    current_builder_sha256 = _KNOWN_BUILDERS[builder_identity]()
    if builder_sha256 != current_builder_sha256:
        raise LblrtmResourceIntegrityError(
            "build recipe builder_sha256 differs from the current resource builder"
        )

    lblrtm_precision = recipe.get("lblrtm_precision")
    if lblrtm_precision not in _EXPECTED_LBLRTM_BUILD_CONTRACTS:
        raise LblrtmResourceIntegrityError(
            "build recipe lblrtm_precision must be 'single' or 'double'"
        )
    lblrtm_expected = _EXPECTED_LBLRTM_BUILD_CONTRACTS[str(lblrtm_precision)]
    _require_recipe_contract_value(
        recipe,
        field="lnfl_precision",
        expected=_EXPECTED_LNFL_BUILD_CONTRACT["precision"],
    )
    _require_recipe_contract_value(
        recipe,
        field="lnfl_make_target",
        expected=_EXPECTED_LNFL_BUILD_CONTRACT["make_target"],
    )
    _require_recipe_contract_value(
        recipe,
        field="lblrtm_make_target",
        expected=lblrtm_expected["make_target"],
    )
    _require_recipe_contract_value(
        recipe,
        field="lnfl_product_basename",
        expected=_EXPECTED_LNFL_BUILD_CONTRACT["product_basename"],
    )
    _require_recipe_contract_value(
        recipe,
        field="lblrtm_product_basename",
        expected=lblrtm_expected["product_basename"],
    )
    recipe_abi = _require_recipe_mapping(
        recipe.get("tape12_abi"),
        field="tape12_abi",
    )
    if dict(recipe_abi) != lblrtm_expected["tape12_abi"]:
        raise LblrtmResourceIntegrityError(
            "build recipe tape12_abi (TAPE12 binary layout) contradicts lblrtm_precision"
        )

    build_contract = _require_recipe_mapping(
        data.get("build_contract"),
        field="manifest build_contract",
    )
    manifest_lnfl_precision = build_contract.get(
        "lnfl_precision",
        _EXPECTED_LNFL_BUILD_CONTRACT["precision"],
    )
    manifest_lblrtm_precision = build_contract.get(
        "lblrtm_precision",
        build_contract.get("precision"),
    )
    manifest_lnfl_target = build_contract.get(
        "lnfl_make_target",
        _EXPECTED_LNFL_BUILD_CONTRACT["make_target"],
    )
    manifest_lblrtm_target = build_contract.get(
        "lblrtm_make_target",
        lblrtm_expected["make_target"],
    )
    if manifest_lnfl_precision != recipe["lnfl_precision"]:
        raise LblrtmResourceIntegrityError("build recipe LNFL precision differs from manifest")
    if manifest_lblrtm_precision != lblrtm_precision:
        raise LblrtmResourceIntegrityError("build recipe LBLRTM precision differs from manifest")
    if manifest_lnfl_target != recipe["lnfl_make_target"]:
        raise LblrtmResourceIntegrityError("build recipe LNFL target differs from manifest")
    if manifest_lblrtm_target != recipe["lblrtm_make_target"]:
        raise LblrtmResourceIntegrityError("build recipe LBLRTM target differs from manifest")

    manifest_abi = build_contract.get("tape12_abi")
    if manifest_abi is None:
        manifest_abi = {
            "byte_order": build_contract.get("byte_order"),
            "record_marker_bytes": build_contract.get("fortran_record_marker_bytes"),
            "real_bytes": 4 if manifest_lblrtm_precision == "single" else 8,
            "integer_bytes": (4 if manifest_lblrtm_precision == "single" else 8),
        }
    if not isinstance(manifest_abi, Mapping) or dict(manifest_abi) != dict(recipe_abi):
        raise LblrtmResourceIntegrityError(
            "build recipe TAPE12 binary layout differs from the manifest"
        )

    expected_basenames = {
        "lnfl_executable": recipe["lnfl_product_basename"],
        "lblrtm_executable": recipe["lblrtm_product_basename"],
    }
    for logical_name, expected_basename in expected_basenames.items():
        if _manifest_product_basename(data, logical_name=logical_name) != expected_basename:
            raise LblrtmResourceIntegrityError(
                f"build recipe {logical_name} basename differs from manifest"
            )
        resolved_path = resolved[f"products.{logical_name}"]
        if resolved_path.name != expected_basename:
            raise LblrtmResourceIntegrityError(
                f"resolved {logical_name} basename differs from build recipe"
            )

    tape3_provenance = _require_recipe_mapping(
        recipe.get("tape3_provenance"),
        field="tape3_provenance",
    )
    tape3_path = resolved["products.tape3"]
    expected_tape3_values = {
        "size_bytes": tape3_path.stat().st_size,
        "sha256": compute_file_sha256(tape3_path),
        "semantic_sha256": compute_tape3_semantic_sha256(tape3_path),
        "scientific_sha256": compute_tape3_scientific_sha256(tape3_path),
    }
    for field, expected in expected_tape3_values.items():
        _require_recipe_contract_value(
            tape3_provenance,
            field=field,
            expected=expected,
        )
    source_policy = tape3_provenance.get("source_policy")
    if not isinstance(source_policy, str) or not source_policy.strip():
        raise LblrtmResourceIntegrityError(
            "build recipe tape3_provenance.source_policy (where TAPE3 came from) must not be empty"
        )


def _validate_v1_build_contract(
    recipe: Mapping[str, Any],
    data: Mapping[str, Any],
    _resolved: Mapping[str, Path],
) -> None:
    """Accept a schema-1 recipe (older single-precision bundles), checking the
    fields that can be inferred."""

    build_contract = _require_recipe_mapping(
        data.get("build_contract"),
        field="manifest build_contract",
    )
    if "precision" in build_contract and build_contract["precision"] != "single":
        raise LblrtmResourceIntegrityError(
            "a build recipe with schema_version 1 can only describe a "
            "single-precision LNFL/LBLRTM build"
        )
    expected = _EXPECTED_LBLRTM_BUILD_CONTRACTS["single"]
    _require_recipe_contract_value(
        recipe,
        field="make_target",
        expected=expected["make_target"],
    )


def _cross_validate_recipe_build_contract(
    recipe: Mapping[str, Any],
    data: Mapping[str, Any],
    resolved: Mapping[str, Path],
) -> None:
    """Check the build recipe against the manifest (by recipe schema version)."""
    schema_version = recipe.get("schema_version")
    if schema_version == 1:
        _validate_v1_build_contract(recipe, data, resolved)
    elif schema_version == 2:
        _validate_v2_build_contract(recipe, data, resolved)
    else:
        raise LblrtmResourceIntegrityError("build recipe schema_version must be 1 or 2")


def freeze_manifest(
    resource_root: Path,
    template_path: Path,
) -> dict[str, Any]:
    """Return a freshly measured candidate manifest for *resource_root*.

    A newly measured manifest is always a candidate and never inherits
    clean-build, portability, or golden-comparison claims from its template.
    """

    template = load_lblrtm_resource_manifest(template_path)
    data: dict[str, Any] = json.loads(json.dumps(template.data))
    # Executable names and formats follow the host platform.
    for _name in ("lnfl_executable", "lblrtm_executable"):
        _record = data["products"][_name]
        _record["relative_path"] = _record["relative_path"].replace("_OS_X_", f"_{_PRODUCT_TAG}_")
        if _HOST_SYSTEM == "Linux":
            _record["executable_format"] = f"ELF {__import__('platform').machine()}"
    # The template's build contract names the macOS targets and products; the
    # recipe of a Linux build is compared against it, so translate it as well.
    _contract = data["build_contract"]
    for _key in ("lnfl_make_target", "lblrtm_make_target"):
        if _key in _contract:
            _contract[_key] = _contract[_key].replace("osx", _TARGET_PREFIX, 1)
    for _key in ("lnfl_product_basename", "lblrtm_product_basename"):
        if _key in _contract:
            _contract[_key] = _contract[_key].replace("_OS_X_", f"_{_PRODUCT_TAG}_")
    if _HOST_SYSTEM == "Linux":
        _contract["target_platform"] = "Linux"
        _contract["target_architecture"] = __import__("platform").machine()
    root = resource_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise LblrtmResourceIntegrityError(f"resource root is not a directory: {root}")

    resolved: dict[str, Path] = {}
    for logical_name, record in _iter_file_records(data):
        path = _resolve_bundle_file(root, record["relative_path"])
        resolved[logical_name] = path
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = compute_file_sha256(path)

    tape5 = parse_lnfl_tape5(resolved["products.lnfl_tape5"])
    build_recipe = validate_lblrtm_build_recipe(
        resolved["products.build_recipe"],
        data,
        expected_requested_bounds_cm1=(
            tape5.requested_vmin_cm1,
            tape5.requested_vmax_cm1,
        ),
        expected_molecule_selection_mask=tape5.molecule_selection_mask,
        require_manifest_distribution_status_match=False,
    )
    _cross_validate_recipe_build_contract(
        build_recipe,
        data,
        resolved,
    )
    tape6 = parse_lnfl_tape6(resolved["products.lnfl_tape6"])
    if tape5.molecule_selection_mask != tape6.molecule_selection_mask:
        raise LblrtmResourceIntegrityError("TAPE5 selection and TAPE6 selection echo disagree")
    tape3_path = resolved["products.tape3"]
    tape3_header = read_tape3_header(tape3_path)
    if tape3_header.linmol != tape6.reported_slot_count:
        raise LblrtmResourceIntegrityError(
            "TAPE3 LINMOL does not match the TAPE6 reported slot count"
        )
    if (
        tuple(tape3_header.molecule_line_counts)
        != tuple(tape6.line_count_by_molecule_id.values())[: tape3_header.linmol]
    ):
        raise LblrtmResourceIntegrityError("TAPE3 per-molecule line counts disagree with TAPE6")
    if tape3_header.total_line_count != tape6.total_line_count:
        raise LblrtmResourceIntegrityError("TAPE3 total line count disagrees with TAPE6")
    for label, tape3_bound, tape6_bound in zip(
        ("lower", "upper"),
        tape3_header.actual_line_bounds_cm_minus_1,
        (tape6.actual_vmin_cm1, tape6.actual_vmax_cm1),
        strict=False,
    ):
        if abs(tape3_bound - tape6_bound) > 5e-4:
            raise LblrtmResourceIntegrityError(f"TAPE3 {label} line bound disagrees with TAPE6")
    expected_names = tuple(
        molecule.formula.upper()
        for molecule in load_lnfl_molecule_registry()[: tape3_header.linmol]
    )
    if tuple(name.upper() for name in tape3_header.molecule_names) != expected_names:
        raise LblrtmResourceIntegrityError(
            "TAPE3 molecule names are not in the LNFL v3.2 registry order"
        )

    contract = data["scientific_contract"]
    contract["requested_bounds_cm1"] = [
        tape5.requested_vmin_cm1,
        tape5.requested_vmax_cm1,
    ]
    contract["nominal_wavelength_coverage_nm"] = [
        round(1.0e7 / tape5.requested_vmax_cm1, 3),
        round(1.0e7 / tape5.requested_vmin_cm1, 3),
    ]
    contract["actual_line_bounds_cm1"] = [
        tape3_header.actual_line_bounds_cm_minus_1[0],
        tape3_header.actual_line_bounds_cm_minus_1[1],
    ]
    contract["selection_field_width"] = len(tape5.molecule_selection_mask)
    contract["molecule_selection_mask"] = tape5.molecule_selection_mask
    contract["selected_molecule_ids"] = list(tape5.selected_molecule_ids)
    contract["selected_molecule_count"] = len(tape5.selected_molecule_ids)
    contract["tape3_linmol"] = tape3_header.linmol
    contract["nonzero_molecule_count_in_range"] = len(tape6.nonzero_molecule_ids)
    contract["total_line_count"] = tape6.total_line_count
    contract["line_count_by_molecule_id"] = {
        str(molecule_id): count for molecule_id, count in tape6.line_count_by_molecule_id.items()
    }
    data["products"]["tape3"]["semantic_sha256"] = compute_tape3_semantic_sha256(tape3_path)
    data["products"]["tape3"]["scientific_sha256"] = compute_tape3_scientific_sha256(tape3_path)
    data["status"] = "candidate"
    build_contract = data["build_contract"]
    retained_build_contract = {
        key: build_contract[key]
        for key in (
            "target_platform",
            "target_architecture",
            "precision",
            "byte_order",
            "fortran_record_marker_bytes",
            "source_tree_policy",
        )
        if key in build_contract
    }
    if build_recipe.get("schema_version") == 2:
        for legacy_field in (
            "precision",
            "byte_order",
            "fortran_record_marker_bytes",
        ):
            retained_build_contract.pop(legacy_field, None)
        tape12_abi = dict(
            _require_recipe_mapping(
                build_recipe.get("tape12_abi"),
                field="tape12_abi",
            )
        )
        retained_build_contract.update(
            {
                "lnfl_precision": build_recipe["lnfl_precision"],
                "lblrtm_precision": build_recipe["lblrtm_precision"],
                "lnfl_make_target": build_recipe["lnfl_make_target"],
                "lblrtm_make_target": build_recipe["lblrtm_make_target"],
                "lnfl_product_basename": build_recipe["lnfl_product_basename"],
                "lblrtm_product_basename": build_recipe["lblrtm_product_basename"],
                "tape12_abi": tape12_abi,
            }
        )
    retained_build_contract["measurement_status"] = (
        "toolchain versions, binary linkage, clean-source status, and "
        "portability were not established by the freeze tool"
    )
    retained_build_contract["runtime_distribution_status"] = build_recipe[
        "runtime_distribution_status"
    ]
    retained_build_contract["runtime_distribution_note"] = (
        "classification was measured by the hash-pinned build recipe; the "
        "freeze tool did not independently rerun the portability audit"
    )
    data["build_contract"] = retained_build_contract
    data["validation"] = {
        "bundle_internal_consistency": "passed",
        "clean_source_build": "not established by freeze tool",
        "runtime_linkage_portability": "not established by freeze tool",
    }
    data["created_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    validate_lblrtm_resource_manifest(data)
    return data


def _comparable_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest without the fields that are not measured from the bundle
    (creation time, status, validation records)."""
    comparable = json.loads(json.dumps(data))
    for unmeasured_field in (
        "created_utc",
        "status",
        "golden_compatibility",
        "validation",
    ):
        comparable.pop(unmeasured_field, None)
    build_contract = comparable.get("build_contract")
    if isinstance(build_contract, Mapping):
        comparable["build_contract"] = {
            "runtime_distribution_status": build_contract.get("runtime_distribution_status")
        }
    return comparable


def _write_json_atomically(
    output_path: Path,
    data: Mapping[str, Any],
    *,
    force: bool,
) -> None:
    """Write *data* as JSON through a temporary file (refuse to overwrite unless
    *force*)."""
    output = output_path.expanduser()
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    """Run the manifest-freezing command-line interface."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if not args.check and args.output is None:
        parser.error("--output is required unless --check is supplied")
    frozen = freeze_manifest(
        args.resource_root,
        args.template,
    )
    if args.check:
        template = load_lblrtm_resource_manifest(args.template)
        if _comparable_manifest(frozen) != _comparable_manifest(template.data):
            print(
                "the bundle files differ from the recorded sizes and checksums",
                file=sys.stderr,
            )
            return 1
        print(f"bundle matches its recorded sizes and checksums: {template.fingerprint}")
        return 0
    _write_json_atomically(args.output, frozen, force=args.force)
    print(f"wrote {args.output}: {compute_lblrtm_resource_fingerprint(frozen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
