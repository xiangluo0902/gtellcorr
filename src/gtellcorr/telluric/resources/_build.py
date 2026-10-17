#!/usr/bin/env python3
"""Build a gtellcorr LNFL/LBLRTM resource bundle offline.

The builder uses three caller-supplied archives (checked against their SHA-256) and never
downloads resources.  It extracts source archives into a new staging tree,
removes archive-provided compiler outputs, builds explicit GNU targets (macOS
arm64 or Linux),
and creates a 47-slot LNFL TAPE3.  LNFL is always built in single precision;
LBLRTM can be built in single or double precision.

The resulting bundle is an input to the freezer (``gtellcorr resources freeze``;
``gtellcorr resources build`` freezes it automatically);
build-time values such as the TAPE3 timestamp mean a newly built bundle must
be frozen before it is treated as an accepted resource set.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

LBLRTM_PACKAGE = Path(__file__).resolve().parents[1] / "lblrtm"

# Build targets and product names follow the host platform (macOS or Linux).
_HOST_SYSTEM = platform.system()
_TARGET_PREFIX = "linux" if _HOST_SYSTEM == "Linux" else "osx"
_PRODUCT_TAG = "linux" if _HOST_SYSTEM == "Linux" else "OS_X"

from gtellcorr.telluric.lblrtm.lblrtm_errors import (  # noqa: E402
    Tape3FormatError,
    Tape3ValidationError,
)
from gtellcorr.telluric.lblrtm.lblrtm_resources import (  # noqa: E402
    compute_file_sha256,
    load_lblrtm_resource_manifest,
    parse_lnfl_tape5,
    parse_lnfl_tape6,
)
from gtellcorr.telluric.lblrtm.lnfl_molecule_selection import (  # noqa: E402
    LNFL_MOLECULE_SLOT_COUNT,
    build_lnfl_molecule_selection_mask,
)
from gtellcorr.telluric.lblrtm.tape3_metadata import (  # noqa: E402
    compute_tape3_scientific_sha256,
    compute_tape3_semantic_sha256,
    read_tape3_header,
    validate_tape3_metadata,
)

LNFL_SOURCE_KEY = "lnfl_source_archive"
LBLRTM_SOURCE_KEY = "lblrtm_source_archive"
LINE_FILE_SOURCE_KEY = "aer_line_file_archive"
LINE_FILE_MEMBER = PurePosixPath("aer_v_3.9/line_file/aer_v_3.9")
MT_CKD_H2O_CONTINUUM_PRODUCT_KEY = "mt_ckd_h2o_continuum_coefficients"
LBLRTM_MT_CKD_SOURCE_MEMBER = PurePosixPath("lblrtm/data/absco-ref_wv-mt-ckd.nc")
MT_CKD_H2O_CONTINUUM_SOURCE_RELATIVE_PATH = PurePosixPath("data/absco-ref_wv-mt-ckd.nc")
LNFL_TOP_DIRECTORY = "lnfl"
LBLRTM_TOP_DIRECTORY = "lblrtm"
LNFL_MAKEFILE = "make_lnfl"
LBLRTM_MAKEFILE = "make_lblrtm"
BUILDER_RELATIVE_PATH = PurePosixPath("gtellcorr/telluric/resources/_build.py")
LNFL_PRECISION = "single"
LNFL_MAKE_TARGET = f"{_TARGET_PREFIX}GNUsgl"
LNFL_EXECUTABLE_NAME = f"lnfl_v3.2_{_PRODUCT_TAG}_gnu_sgl"
REQUESTED_VMIN_CM1 = 2000.0
REQUESTED_VMAX_CM1 = 33333.333
LNFL_COMPILER_FLAGS = "-Wall -frecord-marker=4 -std=legacy -fallow-argument-mismatch"
LBLRTM_SINGLE_COMPILER_FLAGS = "-frecord-marker=4 -std=legacy -fallow-argument-mismatch"
LBLRTM_DOUBLE_COMPILER_FLAGS = (
    "-fdefault-integer-8 -fdefault-real-8 -fdefault-double-8 "
    "-frecord-marker=4 -std=legacy -fallow-argument-mismatch"
)


@dataclass(frozen=True, slots=True)
class SourcePatch:
    """Exact text replacements in one file of a source tree, pinned by the file's
    SHA-256 before and after."""

    patch_id: str
    relative_path: PurePosixPath
    sha256_before: str
    sha256_after: str
    replacements: tuple[tuple[bytes, bytes], ...]
    reason: str

    def record(self) -> dict[str, str]:
        """The patch as written to the build recipe."""
        return {
            "patch_id": self.patch_id,
            "file": self.relative_path.as_posix(),
            "sha256_before": self.sha256_before,
            "sha256_after": self.sha256_after,
            "reason": self.reason,
        }


#: The water-continuum module of LBLRTM v12.17 (MT_CKD 4.3) searches its coefficient
#: table (-20 to 20000 cm-1) without a bound: for a run whose range reaches beyond
#: 20000 cm-1 (shorter than 500 nm) it reads past the end of the table, and the
#: process dies with SIGBUS when the table ends at an unmapped page, or it uses values
#: from beyond the table near 20000 cm-1. The patch stops both searches at the last
#: coefficient and skips the interpolation term past the end of the coefficients (its
#: weight is zero there), as the MT_CKD 3.5 code of LBLRTM v12.11 did. Ranges that
#: stay below 20000 cm-1 take the same code path as before.
MT_CKD_H2O_BOUNDS_PATCH = SourcePatch(
    patch_id="mt_ckd_h2o_bounds",
    relative_path=PurePosixPath("src/mt_ckd_h2o_module.f90"),
    sha256_before="d95d01074c6b188d7c37efd775f955ed33b84a83fa61e262ef8e917571a89615",
    sha256_after="5408b78314183411a532bebf0b2897a9504cc1a4e9ef0cf90913b8f336bbcca1",
    replacements=(
        (
            b"   do while (wvn(i) <= (wv1abs-2*dvc))\n",
            b"   do while (i < ncoeffin .and. wvn(i) <= (wv1abs-2*dvc))\n",
        ),
        (
            b"   do while (wvn(i) < (wv2abs+2*dvc))\n",
            b"   do while (i < ncoeffin .and. wvn(i) < (wv2abs+2*dvc))\n",
        ),
        (
            b"      CONTI = -A(J-1)*B1+A(J)*(1.-C+B2)+A(J+1)*(C+B1)-A(J+2)*B2\n",
            b"      IF (J+2 .LE. SIZE(A)) THEN\n"
            b"         CONTI = -A(J-1)*B1+A(J)*(1.-C+B2)+A(J+1)*(C+B1)-A(J+2)*B2\n"
            b"      ELSE\n"
            b"         CONTI = -A(J-1)*B1+A(J)*(1.-C+B2)+A(J+1)*(C+B1)\n"
            b"      END IF\n",
        ),
    ),
    reason=(
        "MT_CKD 4.3 water continuum: bound the coefficient-table search at 20000 cm-1 "
        "(out-of-bounds read, SIGBUS, for runs extending below 500 nm)"
    ),
)
LBLRTM_SOURCE_PATCHES: tuple[SourcePatch, ...] = (MT_CKD_H2O_BOUNDS_PATCH,)

_MAKE_TARGET_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+:\s*$")
_UNSAFE_COMMAND_CHARACTER_PATTERN = re.compile(r"[\s\x00\r\n]")
_UNSAFE_MAKE_SHELL_PATH_PATTERN = re.compile(r"[^A-Za-z0-9_./:+@=-]")


class ResourceBuildError(RuntimeError):
    """Raised when an offline resource build cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class BuildInputs:
    """Resolved, hash-validated input archives."""

    lnfl_archive: Path
    lblrtm_archive: Path
    line_file_archive: Path


@dataclass(frozen=True, slots=True)
class BuildToolchain:
    """Resolved executables and NetCDF paths used by the source builds."""

    make_command: str
    fortran_compiler: str
    otool_command: str
    sdk_root: Path | None
    netcdf_prefix: Path
    netcdf_library_directory: Path
    netcdf_include_directory: Path


@dataclass(frozen=True, slots=True)
class LblrtmBuildSpec:
    """One supported LBLRTM precision, target, product, and output ABI."""

    precision: str
    make_target: str
    executable_name: str
    compiler_flags: str
    tape12_real_bytes: int
    tape12_integer_bytes: int


@dataclass(frozen=True, slots=True)
class SharedTape3Identity:
    """Checked size and checksums of one shared TAPE3."""

    path: Path
    size_bytes: int
    sha256: str
    semantic_sha256: str
    scientific_sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """A tar member paired with its normalized bundle-relative path."""

    metadata: tarfile.TarInfo
    relative_path: PurePosixPath


_LBLRTM_BUILD_SPECS = {
    "single": LblrtmBuildSpec(
        precision="single",
        make_target=f"{_TARGET_PREFIX}GNUsgl",
        executable_name=f"lblrtm_v12.17_{_PRODUCT_TAG}_gnu_sgl",
        compiler_flags=LBLRTM_SINGLE_COMPILER_FLAGS,
        tape12_real_bytes=4,
        tape12_integer_bytes=4,
    ),
    "double": LblrtmBuildSpec(
        precision="double",
        make_target=f"{_TARGET_PREFIX}GNUdbl",
        executable_name=f"lblrtm_v12.17_{_PRODUCT_TAG}_gnu_dbl",
        compiler_flags=LBLRTM_DOUBLE_COMPILER_FLAGS,
        tape12_real_bytes=8,
        tape12_integer_bytes=8,
    ),
}


def _lblrtm_build_spec(precision: str) -> LblrtmBuildSpec:
    """The build spec of *precision* (``ResourceBuildError`` if unsupported)."""
    try:
        return _LBLRTM_BUILD_SPECS[precision]
    except KeyError as exc:
        supported = ", ".join(sorted(_LBLRTM_BUILD_SPECS))
        raise ResourceBuildError(
            f"unsupported LBLRTM precision {precision!r}; choose {supported}"
        ) from exc


def _build_recipe_precision_contract(
    lblrtm_build_spec: LblrtmBuildSpec,
) -> dict[str, Any]:
    """Return the mixed-precision build settings and the TAPE12 binary layout."""

    return {
        "lnfl_precision": LNFL_PRECISION,
        "lblrtm_precision": lblrtm_build_spec.precision,
        "lnfl_make_target": LNFL_MAKE_TARGET,
        "lblrtm_make_target": lblrtm_build_spec.make_target,
        "lnfl_product_basename": LNFL_EXECUTABLE_NAME,
        "lblrtm_product_basename": lblrtm_build_spec.executable_name,
        "tape12_abi": {
            "byte_order": "little",
            "record_marker_bytes": 4,
            "real_bytes": lblrtm_build_spec.tape12_real_bytes,
            "integer_bytes": lblrtm_build_spec.tape12_integer_bytes,
        },
    }


def _resolve_recipe_schema_version(
    requested_version: int | None,
    *,
    lblrtm_build_spec: LblrtmBuildSpec,
) -> int:
    """Select a compatible recipe schema for the requested precision."""

    version = (
        2
        if requested_version is None and lblrtm_build_spec.precision == "double"
        else 1
        if requested_version is None
        else requested_version
    )
    if version not in {1, 2}:
        raise ResourceBuildError("build recipe schema version must be 1 or 2")
    if lblrtm_build_spec.precision == "double" and version != 2:
        raise ResourceBuildError("double-precision LBLRTM requires build recipe schema version 2")
    return version


def _require_manifest_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    """*value*, which must be a JSON object of the built-in manifest."""
    if not isinstance(value, Mapping):
        raise ResourceBuildError(
            f"internal error: field {field!r} of the shipped LBLRTM manifest is not a table"
        )
    return value


def _source_records() -> Mapping[str, Mapping[str, Any]]:
    """The shipped manifest's records of the source archives."""
    manifest = load_lblrtm_resource_manifest()
    raw_sources = _require_manifest_mapping(
        manifest.data.get("sources"),
        field="sources",
    )
    records: dict[str, Mapping[str, Any]] = {}
    for logical_name in (
        LNFL_SOURCE_KEY,
        LBLRTM_SOURCE_KEY,
        LINE_FILE_SOURCE_KEY,
        "aer_line_file",
    ):
        records[logical_name] = _require_manifest_mapping(
            raw_sources.get(logical_name),
            field=f"sources.{logical_name}",
        )
    return records


def _product_records(
    *,
    lblrtm_build_spec: LblrtmBuildSpec,
) -> Mapping[str, Mapping[str, Any]]:
    """The built-in manifest records of the products to build, for this precision
    and platform."""
    manifest = load_lblrtm_resource_manifest()
    raw_products = _require_manifest_mapping(
        manifest.data.get("products"),
        field="products",
    )
    records: dict[str, Mapping[str, Any]] = {}
    for logical_name in (
        "lnfl_executable",
        "lblrtm_executable",
        "lnfl_tape5",
        "lnfl_tape6",
        "tape3",
        MT_CKD_H2O_CONTINUUM_PRODUCT_KEY,
        "build_recipe",
        "lblrtm_addlibs_template",
    ):
        record = _require_manifest_mapping(
            raw_products.get(logical_name),
            field=f"products.{logical_name}",
        )
        if logical_name == "lblrtm_executable":
            relative_path = record.get("relative_path")
            if not isinstance(relative_path, str):
                raise ResourceBuildError(
                    "internal error: products.lblrtm_executable.relative_path of the shipped "
                    "LBLRTM manifest is not a string"
                )
            record = dict(record)
            record["relative_path"] = str(
                PurePosixPath(relative_path).with_name(lblrtm_build_spec.executable_name)
            )
        if logical_name == "lnfl_executable":
            # platform-specific LNFL product name
            record = dict(record)
            record["relative_path"] = str(
                PurePosixPath(str(record["relative_path"])).with_name(LNFL_EXECUTABLE_NAME)
            )
        records[logical_name] = record
    return records


def _resolve_regular_input(path_value: str | Path, *, label: str) -> Path:
    """*path_value* resolved; it must exist and be a regular file."""
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ResourceBuildError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ResourceBuildError(f"{label} is not a regular file: {resolved}")
    return resolved


def _validate_shared_tape3(
    path_value: str | Path,
    *,
    product_record: Mapping[str, Any],
) -> SharedTape3Identity:
    """Check a caller-shared TAPE3 against the checksums in the shipped manifest."""

    path = _resolve_regular_input(path_value, label="shared TAPE3")
    expected_semantic = product_record.get("semantic_sha256")
    expected_scientific = product_record.get("scientific_sha256")
    if not isinstance(expected_semantic, str) or not isinstance(
        expected_scientific,
        str,
    ):
        raise ResourceBuildError(
            "internal error: the shipped LBLRTM manifest records no SHA-256 for TAPE3"
        )
    try:
        actual_semantic = compute_tape3_semantic_sha256(path)
        actual_scientific = compute_tape3_scientific_sha256(path)
    except (
        OSError,
        TypeError,
        ValueError,
        Tape3FormatError,
        Tape3ValidationError,
    ) as exc:
        raise ResourceBuildError(f"cannot inspect shared TAPE3 {path}: {exc}") from exc
    if actual_semantic != expected_semantic:
        raise ResourceBuildError(
            "the TAPE3 line file differs from the shipped manifest (SHA-256 without the "
            "LNFL date and time): "
            f"expected {expected_semantic}, got {actual_semantic}"
        )
    if actual_scientific != expected_scientific:
        raise ResourceBuildError(
            "the TAPE3 line data differ from the shipped manifest (SHA-256 without the "
            "LNFL date, time and title): "
            f"expected {expected_scientific}, got {actual_scientific}"
        )
    return SharedTape3Identity(
        path=path,
        size_bytes=path.stat().st_size,
        sha256=compute_file_sha256(path),
        semantic_sha256=actual_semantic,
        scientific_sha256=actual_scientific,
    )


def _validate_exact_file_identity(
    path: Path,
    *,
    label: str,
    size_bytes: int,
    sha256_digest: str,
) -> None:
    """Require a copied file to keep the size and checksum measured before."""

    actual_size = path.stat().st_size
    if actual_size != size_bytes:
        raise ResourceBuildError(
            f"{label} size mismatch: expected {size_bytes}, got {actual_size} ({path})"
        )
    actual_sha256 = compute_file_sha256(path)
    if actual_sha256 != sha256_digest:
        raise ResourceBuildError(
            f"{label} SHA-256 mismatch: expected {sha256_digest}, got {actual_sha256} ({path})"
        )


def _validate_pinned_input(
    path: Path,
    *,
    label: str,
    record: Mapping[str, Any],
) -> None:
    """Check that *path* has the size and SHA-256 recorded in *record*."""
    expected_size, expected_sha256 = _expected_pinned_identity(
        record,
        label=label,
    )
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ResourceBuildError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size} ({path})"
        )
    actual_sha256 = compute_file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ResourceBuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256} ({path})"
        )


def _expected_pinned_identity(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, str]:
    """``(size_bytes, sha256)`` recorded in a manifest *record*."""
    expected_size = record.get("size_bytes")
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_size, int) or not isinstance(expected_sha256, str):
        raise ResourceBuildError(
            f"internal error: the shipped LBLRTM manifest records no size and SHA-256 for {label}"
        )
    return expected_size, expected_sha256


def _compute_file_descriptor_sha256(file_descriptor: int) -> str:
    """SHA-256 of an open file, read from the start (the position is reset)."""
    digest = sha256()
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    for chunk in iter(lambda: os.read(file_descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_pinned_file_descriptor(
    file_descriptor: int,
    *,
    label: str,
    record: Mapping[str, Any],
) -> os.stat_result:
    """Validate one already-open regular file without reopening its path."""

    expected_size, expected_sha256 = _expected_pinned_identity(
        record,
        label=label,
    )
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ResourceBuildError(f"{label} is not a regular file")
    actual_size = metadata.st_size
    if actual_size != expected_size:
        raise ResourceBuildError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = _compute_file_descriptor_sha256(file_descriptor)
    if actual_sha256 != expected_sha256:
        raise ResourceBuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return metadata


def _open_directory_chain_without_symlinks(
    root: Path,
    relative_parts: Sequence[str],
    *,
    create: bool,
    label: str,
) -> int:
    """Open a directory chain by descriptor, rejecting every symlink."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    opened: list[int] = []
    try:
        opened.append(os.open(root, directory_flags))
        for part in relative_parts:
            if part in {"", ".", ".."} or "/" in part:
                raise ResourceBuildError(f"{label} contains an unsafe path component: {part!r}")
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, mode=0o755, dir_fd=opened[-1])
            opened.append(os.open(part, directory_flags, dir_fd=opened[-1]))
    except ResourceBuildError:
        for file_descriptor in reversed(opened):
            os.close(file_descriptor)
        raise
    except OSError as exc:
        for file_descriptor in reversed(opened):
            os.close(file_descriptor)
        raise ResourceBuildError(f"{label} is missing, unsafe, or not a directory: {exc}") from exc

    for file_descriptor in opened[:-1]:
        os.close(file_descriptor)
    return opened[-1]


def _copy_file_descriptor_bytes(
    source_file_descriptor: int,
    destination_file_descriptor: int,
) -> None:
    """Copy from the already-validated source descriptor without reopening."""

    os.lseek(source_file_descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_file_descriptor, 1024 * 1024)
        if not chunk:
            break
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(destination_file_descriptor, remaining)
            if written <= 0:
                raise OSError("a write returned zero bytes while copying a file into the bundle")
            remaining = remaining[written:]


def _validate_inputs(
    args: argparse.Namespace,
    source_records: Mapping[str, Mapping[str, Any]],
) -> BuildInputs:
    """The three source archives, resolved and checked against their recorded hashes."""
    inputs = BuildInputs(
        lnfl_archive=_resolve_regular_input(
            args.lnfl_archive,
            label="LNFL source archive",
        ),
        lblrtm_archive=_resolve_regular_input(
            args.lblrtm_archive,
            label="LBLRTM source archive",
        ),
        line_file_archive=_resolve_regular_input(
            args.line_file_archive,
            label="AER line-file archive",
        ),
    )
    _validate_pinned_input(
        inputs.lnfl_archive,
        label="LNFL source archive",
        record=source_records[LNFL_SOURCE_KEY],
    )
    _validate_pinned_input(
        inputs.lblrtm_archive,
        label="LBLRTM source archive",
        record=source_records[LBLRTM_SOURCE_KEY],
    )
    _validate_pinned_input(
        inputs.line_file_archive,
        label="AER line-file archive",
        record=source_records[LINE_FILE_SOURCE_KEY],
    )
    return inputs


def _canonical_output_root(path_value: str | Path) -> Path:
    """The output root resolved: no ``..``, not a symlink, absent or an empty directory."""
    path = Path(path_value).expanduser()
    if ".." in path.parts:
        raise ResourceBuildError(
            f"output root must not contain parent-directory components: {path}"
        )
    if path.is_symlink():
        raise ResourceBuildError(f"output root must not be a symbolic link: {path}")
    if path.exists():
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ResourceBuildError(f"cannot resolve output root: {path}") from exc
        if not resolved.is_dir():
            raise ResourceBuildError(f"output root exists but is not a directory: {resolved}")
        try:
            first_entry = next(resolved.iterdir(), None)
        except OSError as exc:
            raise ResourceBuildError(f"cannot inspect output root: {resolved}") from exc
        if first_entry is not None:
            raise ResourceBuildError(f"output root must be absent or empty: {resolved}")
        return resolved

    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise ResourceBuildError(f"cannot resolve output root: {path}") from exc
    parent = resolved.parent
    while not parent.exists():
        if parent == parent.parent:
            raise ResourceBuildError(f"cannot find an existing parent for output root: {resolved}")
        parent = parent.parent
    if not parent.is_dir():
        raise ResourceBuildError(f"output root parent is not a directory: {parent}")
    return resolved


def _resolve_command(command: str, *, label: str) -> str:
    """The absolute path of *command* on PATH, safe to use in the AER Makefiles."""
    if not command or _UNSAFE_COMMAND_CHARACTER_PATTERN.search(command):
        raise ResourceBuildError(f"{label} must be one executable name or path without whitespace")
    resolved = shutil.which(command)
    if resolved is None:
        raise ResourceBuildError(f"{label} is not executable or on PATH: {command}")
    resolved_path = str(Path(resolved).resolve())
    if _UNSAFE_MAKE_SHELL_PATH_PATTERN.search(resolved_path):
        raise ResourceBuildError(
            f"{label} resolves to a path unsafe for the AER Makefiles: {resolved_path}"
        )
    return resolved_path


def _validate_netcdf_prefix(path_value: str | Path) -> tuple[Path, Path, Path]:
    """``(prefix, lib, include)`` of a NetCDF-Fortran installation, checked to be
    usable and safe for the AER Makefiles."""
    prefix_text = os.fspath(path_value)
    if _UNSAFE_COMMAND_CHARACTER_PATTERN.search(
        prefix_text
    ) or _UNSAFE_MAKE_SHELL_PATH_PATTERN.search(prefix_text):
        raise ResourceBuildError("NetCDF prefix contains characters unsafe for the AER Makefiles")
    try:
        prefix = Path(path_value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ResourceBuildError(f"NetCDF prefix does not exist: {path_value}") from exc
    if not prefix.is_dir():
        raise ResourceBuildError(f"NetCDF prefix is not a directory: {prefix}")
    if _UNSAFE_MAKE_SHELL_PATH_PATTERN.search(str(prefix)):
        raise ResourceBuildError(
            "resolved NetCDF prefix contains characters unsafe for the AER Makefiles"
        )
    library_directory = prefix / "lib"
    include_directory = prefix / "include"
    if not library_directory.is_dir():
        raise ResourceBuildError(f"NetCDF library directory is missing: {library_directory}")
    if not include_directory.is_dir():
        raise ResourceBuildError(f"NetCDF include directory is missing: {include_directory}")
    if not any(library_directory.glob("libnetcdff*.dylib")) and not any(
        library_directory.glob("libnetcdff*.so*")
    ):
        raise ResourceBuildError(
            f"NetCDF-Fortran dynamic library is missing in {library_directory}"
        )
    if not any(library_directory.glob("libnetcdf*.dylib")) and not any(
        library_directory.glob("libnetcdf*.so*")
    ):
        raise ResourceBuildError(f"NetCDF-C dynamic library is missing in {library_directory}")
    if not any(
        path.name.lower() == "netcdf.mod" for path in include_directory.iterdir() if path.is_file()
    ):
        raise ResourceBuildError(f"Fortran module netcdf.mod is missing in {include_directory}")
    return prefix, library_directory, include_directory


def _validate_toolchain(args: argparse.Namespace) -> BuildToolchain:
    """The build toolchain: supported platform and architecture, NetCDF, compilers,
    make, and the macOS SDK."""
    if _HOST_SYSTEM == "Darwin":
        if platform.machine().lower() not in {"arm64", "aarch64"}:
            raise ResourceBuildError("on macOS the resource recipe targets Apple arm64")
    elif _HOST_SYSTEM == "Linux":
        if platform.machine().lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
            raise ResourceBuildError("on Linux the resource recipe targets x86_64 or aarch64")
    else:
        raise ResourceBuildError("resource builds are supported on macOS and Linux only")
    prefix, library_directory, include_directory = _validate_netcdf_prefix(args.netcdf_prefix)
    if _HOST_SYSTEM != "Darwin":
        sdk_root = None
    else:
        try:
            sdk_root = Path(args.sdk_root).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ResourceBuildError(f"macOS SDK root does not exist: {args.sdk_root}") from exc
    if sdk_root is not None and (not sdk_root.is_dir() or not (sdk_root / "usr" / "lib").is_dir()):
        raise ResourceBuildError(f"macOS SDK root is missing usr/lib: {sdk_root}")
    return BuildToolchain(
        make_command=_resolve_command(args.make_command, label="make command"),
        fortran_compiler=_resolve_command(
            args.fortran_compiler,
            label="Fortran compiler",
        ),
        otool_command=_resolve_command(args.otool_command, label="otool command"),
        sdk_root=sdk_root,
        netcdf_prefix=prefix,
        netcdf_library_directory=library_directory,
        netcdf_include_directory=include_directory,
    )


def _normalise_archive_path(
    raw_path: str,
    *,
    archive: Path,
    field: str,
) -> PurePosixPath:
    """A tar member path made relative and free of ``.`` / ``..`` (error if it
    escapes the extraction root)."""
    if not raw_path or "\x00" in raw_path:
        raise ResourceBuildError(f"{archive}: tar {field} contains an empty or NUL path")
    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute():
        raise ResourceBuildError(f"{archive}: tar {field} is absolute: {raw_path!r}")
    stack: list[str] = []
    for part in pure_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise ResourceBuildError(
                    f"{archive}: tar {field} escapes extraction root: {raw_path!r}"
                )
            stack.pop()
            continue
        stack.append(part)
    if not stack:
        raise ResourceBuildError(f"{archive}: tar {field} resolves to the extraction root")
    return PurePosixPath(*stack)


def _validate_archive_members(
    archive_path: Path,
) -> tuple[ArchiveMember, ...]:
    """The members of a tar archive, checked: safe paths, no duplicates, only
    regular files, directories and checked symlinks."""
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            raw_members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ResourceBuildError(f"cannot inspect tar archive {archive_path}: {exc}") from exc

    members: list[ArchiveMember] = []
    seen_paths: set[PurePosixPath] = set()
    for metadata in raw_members:
        relative_path = _normalise_archive_path(
            metadata.name,
            archive=archive_path,
            field="member name",
        )
        if relative_path in seen_paths:
            raise ResourceBuildError(
                f"{archive_path}: duplicate tar member path {relative_path.as_posix()!r}"
            )
        seen_paths.add(relative_path)
        if not (metadata.isdir() or metadata.isfile() or metadata.issym()):
            raise ResourceBuildError(
                f"{archive_path}: unsupported tar member type for "
                f"{relative_path.as_posix()!r}; only directories, regular "
                "files, and confined symbolic links are allowed"
            )
        members.append(
            ArchiveMember(
                metadata=metadata,
                relative_path=relative_path,
            )
        )

    symlink_paths = {member.relative_path for member in members if member.metadata.issym()}
    for member in members:
        for parent in member.relative_path.parents:
            if parent == PurePosixPath("."):
                break
            if parent in symlink_paths:
                raise ResourceBuildError(
                    f"{archive_path}: member {member.relative_path.as_posix()!r} "
                    f"is nested below symbolic link {parent.as_posix()!r}"
                )
        if not member.metadata.issym():
            continue
        link_target = _normalise_archive_path(
            (member.relative_path.parent / member.metadata.linkname).as_posix(),
            archive=archive_path,
            field=f"link target for {member.relative_path.as_posix()!r}",
        )
        if link_target not in seen_paths:
            raise ResourceBuildError(
                f"{archive_path}: symbolic link "
                f"{member.relative_path.as_posix()!r} targets an undeclared "
                f"archive member {link_target.as_posix()!r}"
            )
    return tuple(members)


def _destination_for_member(
    destination: Path,
    member: ArchiveMember,
) -> Path:
    """Where *member* is extracted below *destination*."""
    return destination.joinpath(*member.relative_path.parts)


def _extract_source_archive(
    archive_path: Path,
    destination: Path,
    members: Sequence[ArchiveMember],
) -> None:
    """Extract the validated *members* of a source archive into a new *destination*."""
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in members:
                if member.metadata.isdir():
                    _destination_for_member(destination, member).mkdir(
                        parents=True,
                        exist_ok=False,
                    )
            for member in members:
                metadata = member.metadata
                if not metadata.isfile():
                    continue
                output_path = _destination_for_member(destination, member)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(metadata)
                if source is None:
                    raise ResourceBuildError(
                        f"{archive_path}: cannot read regular member "
                        f"{member.relative_path.as_posix()!r}"
                    )
                with source, output_path.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                output_path.chmod(metadata.mode & 0o777)
            for member in members:
                metadata = member.metadata
                if not metadata.issym():
                    continue
                output_path = _destination_for_member(destination, member)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(metadata.linkname, output_path)
    except (OSError, tarfile.TarError) as exc:
        raise ResourceBuildError(f"cannot safely extract {archive_path}: {exc}") from exc


def _extract_line_file(
    archive_path: Path,
    destination: Path,
    members: Sequence[ArchiveMember],
) -> None:
    """Extract the AER line file (the one expected member) to *destination*."""
    matches = [member for member in members if member.relative_path == LINE_FILE_MEMBER]
    if len(matches) != 1 or not matches[0].metadata.isfile():
        raise ResourceBuildError(
            f"{archive_path}: expected exactly one regular member {LINE_FILE_MEMBER.as_posix()!r}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            source = archive.extractfile(matches[0].metadata)
            if source is None:
                raise ResourceBuildError(
                    f"{archive_path}: cannot read {LINE_FILE_MEMBER.as_posix()!r}"
                )
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        destination.chmod(matches[0].metadata.mode & 0o666)
    except (OSError, tarfile.TarError) as exc:
        raise ResourceBuildError(
            f"cannot extract AER line file from {archive_path}: {exc}"
        ) from exc


def _require_archive_paths(
    archive_path: Path,
    members: Iterable[ArchiveMember],
    required_paths: Iterable[PurePosixPath],
) -> None:
    """Check that the archive has every required path, each a regular file."""
    available = {member.relative_path: member for member in members}
    required = tuple(required_paths)
    missing = [path for path in required if path not in available]
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        raise ResourceBuildError(
            f"{archive_path}: required source members are missing: {formatted}"
        )
    non_regular = [path for path in required if not available[path].metadata.isfile()]
    if non_regular:
        formatted = ", ".join(path.as_posix() for path in non_regular)
        raise ResourceBuildError(
            f"{archive_path}: required source members are not regular files: {formatted}"
        )


def _safe_remove_staged_path(path: Path, *, staging_root: Path) -> None:
    """Remove *path*, which must lie inside the staging root."""
    try:
        parent = path.parent.resolve(strict=True)
        root = staging_root.resolve(strict=True)
    except OSError as exc:
        raise ResourceBuildError(f"cannot check the build file {path}: {exc}") from exc
    if parent != root and root not in parent.parents:
        raise ResourceBuildError(f"refusing to remove a path outside the build directory: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _clean_archive_build_outputs(
    source_root: Path,
    *,
    executable_name: str,
) -> None:
    """Remove the executable and object files shipped inside an extracted source tree."""
    executable = source_root / executable_name
    if executable.exists() or executable.is_symlink():
        _safe_remove_staged_path(executable, staging_root=source_root)
    build_directory = source_root / "build"
    if not build_directory.is_dir():
        raise ResourceBuildError(f"extracted source tree has no build directory: {source_root}")
    for candidate in sorted(build_directory.iterdir()):
        if candidate.name.endswith(".obj") or candidate.suffix in {".o", ".mod"}:
            _safe_remove_staged_path(candidate, staging_root=source_root)


def _patch_osx_gnu_target(
    makefile_path: Path,
    *,
    make_target: str,
    compiler_flags: str,
) -> None:
    """Set the compiler (``FC=$(FC)``) and the fixed flags of *make_target* in a
    Makefile."""
    try:
        lines = makefile_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        raise ResourceBuildError(f"cannot read build makefile {makefile_path}: {exc}") from exc
    start_indices = [index for index, line in enumerate(lines) if line.strip() == f"{make_target}:"]
    if len(start_indices) != 1:
        raise ResourceBuildError(f"{makefile_path}: expected exactly one {make_target} target")
    start = start_indices[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _MAKE_TARGET_PATTERN.fullmatch(lines[index].strip()):
            end = index
            break

    compiler_indices = [
        index for index in range(start + 1, end) if lines[index].lstrip().startswith("FC=")
    ]
    flag_indices = [
        index for index in range(start + 1, end) if lines[index].lstrip().startswith("FCFLAG=")
    ]
    if len(compiler_indices) != 1 or len(flag_indices) != 1:
        raise ResourceBuildError(
            f"{makefile_path}: {make_target} must define exactly one FC and one FCFLAG assignment"
        )
    lines[compiler_indices[0]] = "\tFC=$(FC) \\\n"
    lines[flag_indices[0]] = f'\tFCFLAG="{compiler_flags}" \\\n'
    try:
        makefile_path.write_text("".join(lines), encoding="utf-8")
    except OSError as exc:
        raise ResourceBuildError(f"cannot patch build makefile {makefile_path}: {exc}") from exc


def apply_source_patch(source_root: Path, patch: SourcePatch) -> None:
    """Apply *patch* to *source_root*; the file must have the recorded SHA-256 before
    and after, and every replaced text must occur exactly once."""
    path = source_root / Path(*patch.relative_path.parts)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ResourceBuildError(f"cannot read {path} to apply {patch.patch_id}: {exc}") from exc
    if sha256(data).hexdigest() != patch.sha256_before:
        raise ResourceBuildError(
            f"{path} is not the expected source (SHA-256 differs); patch {patch.patch_id} "
            "was written for the pinned LBLRTM v12.17 archive"
        )
    for old, new in patch.replacements:
        count = data.count(old)
        if count != 1:
            raise ResourceBuildError(
                f"patch {patch.patch_id}: expected one occurrence in {path}, found {count}"
            )
        data = data.replace(old, new)
    if sha256(data).hexdigest() != patch.sha256_after:
        raise ResourceBuildError(
            f"patch {patch.patch_id}: patched {path} has an unexpected SHA-256"
        )
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise ResourceBuildError(f"cannot write {path} for {patch.patch_id}: {exc}") from exc


def _write_lblrtm_addlibs(
    build_directory: Path,
    toolchain: BuildToolchain,
) -> None:
    """Write LBLRTM's ``addlibs.inc`` pointing at the NetCDF library and headers."""
    contents = (
        "# Generated by gtellcorr's offline resource builder.\n"
        f"ADDLIB = -Wl,-rpath,{toolchain.netcdf_library_directory}\n"
        "NETCDF = yes\n"
        f"NCL = {toolchain.netcdf_library_directory}\n"
        f"NCI = {toolchain.netcdf_include_directory}\n"
        "INCLUDES := -I. -I$(NCI)\n"
    )
    try:
        (build_directory / "addlibs.inc").write_text(contents, encoding="utf-8")
    except OSError as exc:
        raise ResourceBuildError(f"cannot write parameterized LBLRTM addlibs.inc: {exc}") from exc


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    label: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    """Run *command* in *cwd* (printing it), with a minimal environment if
    *environment_overrides* is given; ``ResourceBuildError`` on failure."""
    printable = " ".join(command)
    print(f"[build] {label}: {printable}", flush=True)
    environment: dict[str, str] | None = None
    if environment_overrides is not None:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
        }
        for temporary_variable in ("TMPDIR", "TMP", "TEMP"):
            if temporary_variable in os.environ:
                environment[temporary_variable] = os.environ[temporary_variable]
        environment.update(environment_overrides)
    try:
        subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            shell=False,
            env=environment,
        )
    except OSError as exc:
        raise ResourceBuildError(f"{label} could not start in {cwd}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise ResourceBuildError(
            f"{label} failed with exit status {exc.returncode}; "
            f"the build directory is kept at {cwd}"
        ) from exc


def _build_lnfl(source_root: Path, toolchain: BuildToolchain) -> Path:
    """Compile LNFL (single precision) in *source_root*; return the executable."""
    _clean_archive_build_outputs(
        source_root,
        executable_name=LNFL_EXECUTABLE_NAME,
    )
    build_directory = source_root / "build"
    _patch_osx_gnu_target(
        build_directory / "makefile.common",
        make_target=LNFL_MAKE_TARGET,
        compiler_flags=LNFL_COMPILER_FLAGS,
    )
    _run_checked(
        (
            toolchain.make_command,
            "-f",
            LNFL_MAKEFILE,
            LNFL_MAKE_TARGET,
            f"FC={toolchain.fortran_compiler}",
        ),
        cwd=build_directory,
        label="LNFL source build",
        environment_overrides=({"SDKROOT": str(toolchain.sdk_root)} if toolchain.sdk_root else {}),
    )
    executable = source_root / LNFL_EXECUTABLE_NAME
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ResourceBuildError(f"LNFL build did not create an executable: {executable}")
    return executable


def _build_lblrtm(
    source_root: Path,
    toolchain: BuildToolchain,
    build_spec: LblrtmBuildSpec,
) -> Path:
    """Compile LBLRTM for *build_spec* in *source_root*, with
    :data:`LBLRTM_SOURCE_PATCHES` applied; return the executable."""
    _clean_archive_build_outputs(
        source_root,
        executable_name=build_spec.executable_name,
    )
    for patch in LBLRTM_SOURCE_PATCHES:
        apply_source_patch(source_root, patch)
    build_directory = source_root / "build"
    _patch_osx_gnu_target(
        build_directory / "makefile.common",
        make_target=build_spec.make_target,
        compiler_flags=build_spec.compiler_flags,
    )
    _write_lblrtm_addlibs(build_directory, toolchain)
    _run_checked(
        (
            toolchain.make_command,
            "-f",
            LBLRTM_MAKEFILE,
            build_spec.make_target,
            f"FC={toolchain.fortran_compiler}",
        ),
        cwd=build_directory,
        label="LBLRTM source build",
        environment_overrides=({"SDKROOT": str(toolchain.sdk_root)} if toolchain.sdk_root else {}),
    )
    executable = source_root / build_spec.executable_name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ResourceBuildError(f"LBLRTM build did not create an executable: {executable}")
    return executable


def _write_lnfl_tape5(path: Path) -> str:
    """Write the LNFL TAPE5 (requested range, all molecule slots); return the mask."""
    selected_ids = range(1, LNFL_MOLECULE_SLOT_COUNT + 1)
    mask = build_lnfl_molecule_selection_mask(selected_ids)
    contents = (
        "$ LNFL TAPE5 for aer_v_3.9 -- gtellcorr native resource build\n"
        f"{REQUESTED_VMIN_CM1:10.3f}{REQUESTED_VMAX_CM1:10.3f}\n"
        f"{mask}\n"
    )
    try:
        path.write_text(contents, encoding="ascii")
    except OSError as exc:
        raise ResourceBuildError(f"cannot write LNFL TAPE5: {exc}") from exc
    return mask


def _run_lnfl(
    executable: Path,
    *,
    line_file: Path,
    run_directory: Path,
) -> tuple[Path, Path, Path]:
    """Run LNFL on the line file in a new *run_directory* and check its TAPE3 against
    the TAPE6 summary; return ``(TAPE3, TAPE5, TAPE6)``."""
    run_directory.mkdir(parents=True, exist_ok=False)
    tape1 = run_directory / "TAPE1"
    relative_line_file = os.path.relpath(line_file, run_directory)
    os.symlink(relative_line_file, tape1)
    tape5 = run_directory / "TAPE5"
    expected_mask = _write_lnfl_tape5(tape5)
    _run_checked(
        (str(executable),),
        cwd=run_directory,
        label="LNFL TAPE3 generation",
        environment_overrides={},
    )
    tape3 = run_directory / "TAPE3"
    tape6 = run_directory / "TAPE6"
    for label, path in (("TAPE3", tape3), ("TAPE6", tape6)):
        if not path.is_file() or path.stat().st_size == 0:
            raise ResourceBuildError(f"LNFL did not create a non-empty {label}: {path}")

    tape5_summary = parse_lnfl_tape5(tape5)
    tape6_summary = parse_lnfl_tape6(tape6)
    if tape5_summary.molecule_selection_mask != expected_mask:
        raise ResourceBuildError("generated TAPE5 does not keep the standard 47-slot molecule mask")
    if tape6_summary.molecule_selection_mask != expected_mask:
        raise ResourceBuildError("LNFL TAPE6 did not echo the standard 47-slot molecule mask")
    tape3_header = read_tape3_header(tape3)
    validate_tape3_metadata(
        tape3_header,
        expected_linmol=LNFL_MOLECULE_SLOT_COUNT,
        expected_actual_line_bounds_cm_minus_1=(
            tape6_summary.actual_vmin_cm1,
            tape6_summary.actual_vmax_cm1,
        ),
        bounds_tolerance_cm_minus_1=5e-4,
        expected_total_line_count=tape6_summary.total_line_count,
        expected_molecule_line_counts=tape6_summary.line_count_by_molecule_id,
    )
    return tape3, tape5, tape6


def _validate_tape3_against_lnfl_run(
    shared_tape3: SharedTape3Identity,
    *,
    generated_tape3: Path,
    tape6: Path,
) -> None:
    """Require a shared TAPE3 to match the just-executed LNFL science."""

    tape6_summary = parse_lnfl_tape6(tape6)
    try:
        tape3_header = read_tape3_header(shared_tape3.path)
        validate_tape3_metadata(
            tape3_header,
            expected_linmol=LNFL_MOLECULE_SLOT_COUNT,
            expected_actual_line_bounds_cm_minus_1=(
                tape6_summary.actual_vmin_cm1,
                tape6_summary.actual_vmax_cm1,
            ),
            bounds_tolerance_cm_minus_1=5e-4,
            expected_total_line_count=tape6_summary.total_line_count,
            expected_molecule_line_counts=(tape6_summary.line_count_by_molecule_id),
        )
    except (Tape3FormatError, Tape3ValidationError) as exc:
        raise ResourceBuildError(
            f"shared TAPE3 disagrees with the clean LNFL execution: {exc}"
        ) from exc
    try:
        generated_semantic = compute_tape3_semantic_sha256(generated_tape3)
        generated_scientific = compute_tape3_scientific_sha256(generated_tape3)
    except (
        OSError,
        TypeError,
        ValueError,
        Tape3FormatError,
        Tape3ValidationError,
    ) as exc:
        raise ResourceBuildError(
            f"cannot compare the clean LNFL TAPE3 with the shared TAPE3: {exc}"
        ) from exc
    if generated_semantic != shared_tape3.semantic_sha256:
        raise ResourceBuildError(
            "the TAPE3 just built by LNFL differs from the shipped TAPE3 (SHA-256 "
            "without the LNFL date and time): "
            f"expected {shared_tape3.semantic_sha256}, got "
            f"{generated_semantic}"
        )
    if generated_scientific != shared_tape3.scientific_sha256:
        raise ResourceBuildError(
            "clean LNFL TAPE3 scientific SHA-256 differs from the shared "
            "TAPE3: "
            f"expected {shared_tape3.scientific_sha256}, got "
            f"{generated_scientific}"
        )


def _bundle_path(
    output_root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Path of a manifest *record* inside the bundle (must stay inside it)."""
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str):
        raise ResourceBuildError(
            f"internal error: {label} in the shipped LBLRTM manifest has no relative_path"
        )
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ResourceBuildError(
            f"internal error: {label} in the shipped LBLRTM manifest points outside the bundle"
        )
    return output_root.joinpath(*pure_path.parts)


def _copy_to_bundle(source: Path, destination: Path) -> None:
    """Copy *source* into the bundle (not onto itself)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve(strict=True) == destination.resolve(strict=False):
            raise ResourceBuildError(f"input and bundle destination are the same file: {source}")
        shutil.copy2(source, destination)
    except OSError as exc:
        raise ResourceBuildError(f"cannot stage {source} as {destination}: {exc}") from exc


def _stage_mt_ckd_h2o_runtime_resource(
    lblrtm_source_root: Path,
    output_root: Path,
    product_record: Mapping[str, Any],
) -> Path:
    """Copy the MT_CKD file into the bundle with no-follow, exclusive descriptors."""

    source_relative = MT_CKD_H2O_CONTINUUM_SOURCE_RELATIVE_PATH
    source_parent_fd = _open_directory_chain_without_symlinks(
        lblrtm_source_root,
        source_relative.parts[:-1],
        create=False,
        label="MT_CKD H2O continuum source parent",
    )
    try:
        try:
            source_fd = os.open(
                source_relative.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=source_parent_fd,
            )
        except OSError as exc:
            raise ResourceBuildError(
                "extracted LBLRTM source is missing a safe regular "
                f"{source_relative.as_posix()}: {exc}"
            ) from exc
    finally:
        os.close(source_parent_fd)

    try:
        source_metadata = _validate_pinned_file_descriptor(
            source_fd,
            label="MT_CKD H2O continuum source data",
            record=product_record,
        )
        destination = _bundle_path(
            output_root,
            product_record,
            label=f"products.{MT_CKD_H2O_CONTINUUM_PRODUCT_KEY}",
        )
        destination_relative = PurePosixPath(str(product_record["relative_path"]))
        destination_parent_fd = _open_directory_chain_without_symlinks(
            output_root,
            destination_relative.parts[:-1],
            create=True,
            label="MT_CKD H2O continuum bundle parent",
        )
        try:
            destination_fd: int | None = None
            destination_created = False
            copy_completed = False
            try:
                destination_mode = source_metadata.st_mode & 0o666
                if destination_mode == 0:
                    destination_mode = 0o600
                destination_fd = os.open(
                    destination_relative.name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    destination_mode,
                    dir_fd=destination_parent_fd,
                )
                destination_created = True
                _copy_file_descriptor_bytes(source_fd, destination_fd)
                os.fsync(destination_fd)
                _validate_pinned_file_descriptor(
                    destination_fd,
                    label="staged MT_CKD H2O continuum data",
                    record=product_record,
                )
                copy_completed = True
            except ResourceBuildError:
                raise
            except OSError as exc:
                raise ResourceBuildError(
                    f"cannot safely stage MT_CKD H2O continuum data: {exc}"
                ) from exc
            finally:
                if destination_fd is not None:
                    os.close(destination_fd)
                if destination_created and not copy_completed:
                    with contextlib.suppress(OSError):
                        os.unlink(
                            destination_relative.name,
                            dir_fd=destination_parent_fd,
                        )
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_fd)

    return destination


def _move_to_bundle(source: Path, destination: Path) -> None:
    """Move a generated file into the bundle (never overwriting)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ResourceBuildError(f"refusing to overwrite an existing file: {destination}")
    try:
        source.replace(destination)
    except OSError as exc:
        raise ResourceBuildError(
            f"cannot move generated resource {source} to {destination}: {exc}"
        ) from exc


def _stage_notices(output_root: Path) -> None:
    """Copy the third-party licence notices listed in the manifest into the bundle."""
    manifest = load_lblrtm_resource_manifest()
    licenses = _require_manifest_mapping(
        manifest.data.get("licenses"),
        field="licenses",
    )
    package_directory = LBLRTM_PACKAGE
    for license_name, raw_record in licenses.items():
        record = _require_manifest_mapping(
            raw_record,
            field=f"licenses.{license_name}",
        )
        raw_notice_record = record.get("notice_file")
        if raw_notice_record is None:
            continue
        notice_record = _require_manifest_mapping(
            raw_notice_record,
            field=f"licenses.{license_name}.notice_file",
        )
        relative_notice = notice_record.get("relative_path")
        if not isinstance(relative_notice, str):
            raise ResourceBuildError(f"licenses.{license_name}.notice_file has no relative_path")
        pure_notice = PurePosixPath(relative_notice)
        if pure_notice.is_absolute() or ".." in pure_notice.parts:
            raise ResourceBuildError(f"license notice path escapes bundle: {relative_notice}")
        source = package_directory.joinpath(*pure_notice.parts)
        if not source.is_file():
            raise ResourceBuildError(f"packaged third-party notice is missing: {source}")
        destination = output_root.joinpath(*pure_notice.parts)
        _copy_to_bundle(source, destination)


def _capture_checked_text(
    command: Sequence[str],
    *,
    label: str,
) -> str:
    """Standard output of *command* (``ResourceBuildError`` if it fails)."""
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            shell=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ResourceBuildError(f"{label} could not start: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ResourceBuildError(
            f"{label} failed with exit status {exc.returncode}: {detail}"
        ) from exc
    return completed.stdout


def _inspect_elf_linkage(
    executable: Path,
    *,
    readelf_command: str,
) -> dict[str, Any]:
    """ELF counterpart (Linux) of the Mach-O linkage check."""

    dynamic = _capture_checked_text(
        (readelf_command, "-d", str(executable)),
        label=f"ELF dynamic-section inspection for {executable.name}",
    ).splitlines()
    linked_libraries: list[str] = []
    rpaths: list[str] = []
    for line in dynamic:
        match = re.search(r"\((NEEDED|RUNPATH|RPATH)\)\s+.*\[(.+)\]", line)
        if match is None:
            continue
        if match.group(1) == "NEEDED":
            linked_libraries.append(match.group(2))
        else:
            rpaths.extend(p for p in match.group(2).split(":") if p)
    non_system_absolute_libraries = [
        library
        for library in linked_libraries
        if library.startswith("/")
        and not library.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/"))
    ]
    absolute_rpaths = [rpath for rpath in rpaths if rpath.startswith("/")]
    return {
        "linked_libraries": linked_libraries,
        "lc_rpaths": rpaths,
        "non_system_absolute_libraries": non_system_absolute_libraries,
        "absolute_rpaths": absolute_rpaths,
        "relocatable": not (non_system_absolute_libraries or absolute_rpaths),
    }


def _inspect_macho_linkage(
    executable: Path,
    *,
    otool_command: str,
) -> dict[str, Any]:
    """Linked libraries and runtime paths of an executable (Mach-O on macOS, ELF
    on Linux)."""
    if _HOST_SYSTEM == "Linux":
        return _inspect_elf_linkage(executable, readelf_command=otool_command)
    libraries_output = _capture_checked_text(
        (otool_command, "-L", str(executable)),
        label=f"Mach-O library inspection for {executable.name}",
    )
    linked_libraries = [
        line.strip().split(" (", 1)[0] for line in libraries_output.splitlines()[1:] if line.strip()
    ]
    load_commands = _capture_checked_text(
        (otool_command, "-l", str(executable)),
        label=f"Mach-O load-command inspection for {executable.name}",
    ).splitlines()
    rpaths: list[str] = []
    for index, line in enumerate(load_commands):
        if line.strip() != "cmd LC_RPATH":
            continue
        for candidate in load_commands[index + 1 : index + 5]:
            match = re.match(r"^\s*path\s+(.+?)\s+\(offset\s+\d+\)\s*$", candidate)
            if match is not None:
                rpaths.append(match.group(1))
                break

    non_system_absolute_libraries = [
        library
        for library in linked_libraries
        if library.startswith("/") and not library.startswith(("/usr/lib/", "/System/Library/"))
    ]
    absolute_rpaths = [rpath for rpath in rpaths if rpath.startswith("/")]
    return {
        "linked_libraries": linked_libraries,
        "lc_rpaths": rpaths,
        "non_system_absolute_libraries": non_system_absolute_libraries,
        "absolute_rpaths": absolute_rpaths,
        "relocatable": not (non_system_absolute_libraries or absolute_rpaths),
    }


def _collect_netcdf_provenance(
    toolchain: BuildToolchain,
    *,
    linked_libraries: Sequence[str],
) -> dict[str, Any]:
    """Where the NetCDF installation the executables link to came from
    (configuration tools, versions, libraries)."""
    configuration_tools: dict[str, Any] = {}
    missing_configuration_tools: list[str] = []
    for tool_name in ("nc-config", "nf-config"):
        tool_path = toolchain.netcdf_prefix / "bin" / tool_name
        if not tool_path.is_file() or not os.access(tool_path, os.X_OK):
            missing_configuration_tools.append(tool_name)
            continue
        configuration_tools[tool_name] = {
            "sha256": compute_file_sha256(tool_path),
            "version": _capture_checked_text(
                (str(tool_path), "--version"),
                label=f"{tool_name} version inspection",
            ).strip(),
        }

    libraries: dict[str, Any] = {}
    for install_name in linked_libraries:
        library_name = Path(install_name).name
        if not library_name.startswith(("libnetcdf.", "libnetcdff.")):
            continue
        requested_path = toolchain.netcdf_library_directory / library_name
        try:
            resolved_path = requested_path.resolve(strict=True)
        except OSError as exc:
            raise ResourceBuildError(
                f"linked NetCDF library is missing from the supplied prefix: {requested_path}"
            ) from exc
        libraries[install_name] = {
            "requested_basename": library_name,
            "resolved_basename": resolved_path.name,
            "size_bytes": resolved_path.stat().st_size,
            "sha256": compute_file_sha256(resolved_path),
        }
    if not libraries:
        raise ResourceBuildError("LBLRTM linkage inspection found no NetCDF libraries")
    return {
        "configuration_tools": configuration_tools,
        "missing_configuration_tools": missing_configuration_tools,
        "linked_library_files": libraries,
    }


def _write_provenance(
    output_root: Path,
    *,
    source_records: Mapping[str, Mapping[str, Any]],
    product_records: Mapping[str, Mapping[str, Any]],
    toolchain: BuildToolchain,
    lnfl_executable: Path,
    lblrtm_executable: Path,
    lblrtm_build_spec: LblrtmBuildSpec,
    recipe_schema_version: int,
    builder_sha256: str,
    tape3_product: Path,
    tape3_source_policy: str,
) -> None:
    """Write the record of the build (sources, products, toolchain, recipe) into the
    bundle."""
    runtime_linkage = {
        "lnfl_executable": _inspect_macho_linkage(
            lnfl_executable,
            otool_command=toolchain.otool_command,
        ),
        "lblrtm_executable": _inspect_macho_linkage(
            lblrtm_executable,
            otool_command=toolchain.otool_command,
        ),
    }
    for executable_name, executable_path in (
        ("lnfl_executable", lnfl_executable),
        ("lblrtm_executable", lblrtm_executable),
    ):
        runtime_linkage[executable_name]["size_bytes"] = executable_path.stat().st_size
        runtime_linkage[executable_name]["sha256"] = compute_file_sha256(executable_path)
    all_relocatable = all(record["relocatable"] for record in runtime_linkage.values())
    recipe = {
        "schema_version": recipe_schema_version,
        "builder": BUILDER_RELATIVE_PATH.as_posix(),
        "network_access": "not used",
        "tape3_provenance": {
            "source_policy": tape3_source_policy,
            "size_bytes": tape3_product.stat().st_size,
            "sha256": compute_file_sha256(tape3_product),
            "semantic_sha256": compute_tape3_semantic_sha256(tape3_product),
            "scientific_sha256": compute_tape3_scientific_sha256(tape3_product),
        },
        "make_command": {
            "basename": Path(toolchain.make_command).name,
            "sha256": compute_file_sha256(toolchain.make_command),
            "version_banner": _capture_checked_text(
                (toolchain.make_command, "--version"),
                label="make version inspection",
            ).splitlines()[0],
        },
        "fortran_compiler": {
            "basename": Path(toolchain.fortran_compiler).name,
            "sha256": compute_file_sha256(toolchain.fortran_compiler),
            "version_banner": _capture_checked_text(
                (toolchain.fortran_compiler, "--version"),
                label="Fortran compiler version inspection",
            ).splitlines()[0],
        },
        "netcdf_prefix_policy": (
            "caller supplied; exact build path and resulting Mach-O linkage "
            "are recorded because this candidate executable is host-bound"
        ),
        "netcdf_prefix": str(toolchain.netcdf_prefix),
        "sdk_root_policy": ("caller supplied; exact build path is recorded for provenance"),
        "sdk_root": str(toolchain.sdk_root) if toolchain.sdk_root else "not used (Linux)",
        "sdk_name": toolchain.sdk_root.name if toolchain.sdk_root else "none",
        "lnfl_compiler_flags": LNFL_COMPILER_FLAGS.split(),
        "lblrtm_compiler_flags": lblrtm_build_spec.compiler_flags.split(),
        "lblrtm_source_patches": [patch.record() for patch in LBLRTM_SOURCE_PATCHES],
        "requested_bounds_cm1": [
            REQUESTED_VMIN_CM1,
            REQUESTED_VMAX_CM1,
        ],
        "molecule_selection_mask": build_lnfl_molecule_selection_mask(),
        "source_sha256": {key: record["sha256"] for key, record in source_records.items()},
        "netcdf_provenance": _collect_netcdf_provenance(
            toolchain,
            linked_libraries=runtime_linkage["lblrtm_executable"]["linked_libraries"],
        ),
        "runtime_linkage": runtime_linkage,
        "runtime_distribution_status": ("relocatable" if all_relocatable else "host-bound"),
        "runtime_distribution_note": (
            "all linked libraries and RPATH entries are relocatable"
            if all_relocatable
            else (
                "rebuild from the pinned sources or create a separately "
                "licensed relocatable dependency bundle before distribution"
            )
        ),
    }
    if recipe_schema_version == 1:
        recipe["make_target"] = LNFL_MAKE_TARGET
    else:
        recipe["builder_sha256"] = builder_sha256
        recipe.update(_build_recipe_precision_contract(lblrtm_build_spec))
    recipe_path = _bundle_path(
        output_root,
        product_records["build_recipe"],
        label="products.build_recipe",
    )
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    addlibs_template = (
        "# Substitute one caller-supplied NETCDF_PREFIX at build time.\n"
        "# This creates a host-bound candidate, not a relocatable binary.\n"
        "ADDLIB = -Wl,-rpath,<NETCDF_PREFIX>/lib\n"
        "NETCDF = yes\n"
        "NCL = <NETCDF_PREFIX>/lib\n"
        "NCI = <NETCDF_PREFIX>/include\n"
        "INCLUDES := -I. -I$(NCI)\n"
    )
    addlibs_template_path = _bundle_path(
        output_root,
        product_records["lblrtm_addlibs_template"],
        label="products.lblrtm_addlibs_template",
    )
    addlibs_template_path.parent.mkdir(parents=True, exist_ok=True)
    addlibs_template_path.write_text(
        addlibs_template,
        encoding="utf-8",
    )
    if not all_relocatable:
        print(
            "[warning] the built executables load libraries by absolute path from "
            "outside the bundle; the bundle cannot be copied to another computer",
            flush=True,
        )


def _remove_successful_build_tree(
    build_root: Path,
    *,
    output_root: Path,
) -> None:
    """Remove ``<output_root>/.build`` after a successful build."""
    if build_root.name != ".build" or build_root.parent != output_root:
        raise ResourceBuildError(f"refusing to remove unexpected build tree: {build_root}")
    _safe_remove_staged_path(build_root, staging_root=output_root)


def _print_dry_run_plan(
    *,
    inputs: BuildInputs,
    output_root: Path,
    toolchain: BuildToolchain,
    lblrtm_build_spec: LblrtmBuildSpec,
    recipe_schema_version: int,
    builder_sha256: str,
    tape3_source_policy: str,
    keep_build_tree: bool,
) -> None:
    """Print what a build would do, as JSON, without doing it."""
    plan = {
        "dry_run": True,
        "network_access": False,
        "inputs": {
            "lnfl_archive": str(inputs.lnfl_archive),
            "lblrtm_archive": str(inputs.lblrtm_archive),
            "line_file_archive": str(inputs.line_file_archive),
        },
        "output_root": str(output_root),
        "toolchain": {
            "make_command": toolchain.make_command,
            "fortran_compiler": toolchain.fortran_compiler,
            "otool_command": toolchain.otool_command,
            "netcdf_prefix": str(toolchain.netcdf_prefix),
            "sdk_root": str(toolchain.sdk_root) if toolchain.sdk_root else None,
        },
        "build_contract": _build_recipe_precision_contract(lblrtm_build_spec),
        "build_recipe_schema_version": recipe_schema_version,
        "builder": {
            "relative_path": BUILDER_RELATIVE_PATH.as_posix(),
            "sha256": builder_sha256,
        },
        "scientific_contract": {
            "requested_bounds_cm1": [
                REQUESTED_VMIN_CM1,
                REQUESTED_VMAX_CM1,
            ],
            "selection_field_width": LNFL_MOLECULE_SLOT_COUNT,
            "selection_mask": build_lnfl_molecule_selection_mask(),
            "tape3_source_policy": tape3_source_policy,
        },
        "keep_build_tree": keep_build_tree,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def build_resource_bundle(args: argparse.Namespace) -> Path:
    """Execute one validated offline build and return the bundle root."""

    builder_path = Path(__file__).resolve(strict=True)
    builder_size_bytes = builder_path.stat().st_size
    builder_sha256 = compute_file_sha256(builder_path)
    lblrtm_build_spec = _lblrtm_build_spec(getattr(args, "lblrtm_precision", "single"))
    recipe_schema_version = _resolve_recipe_schema_version(
        getattr(args, "recipe_schema_version", None),
        lblrtm_build_spec=lblrtm_build_spec,
    )
    source_records = _source_records()
    product_records = _product_records(
        lblrtm_build_spec=lblrtm_build_spec,
    )
    raw_tape3_source = getattr(args, "tape3_source", None)
    shared_tape3 = (
        _validate_shared_tape3(
            raw_tape3_source,
            product_record=product_records["tape3"],
        )
        if raw_tape3_source is not None
        else None
    )
    tape3_source_policy = (
        "caller-supplied shared TAPE3 validated against the pinned semantic "
        "and scientific identities; source path intentionally omitted"
        if shared_tape3 is not None
        else "generated by the clean-built LNFL executable in this invocation"
    )
    output_root = _canonical_output_root(args.output_root)
    inputs = _validate_inputs(args, source_records)
    toolchain = _validate_toolchain(args)

    lnfl_members = _validate_archive_members(inputs.lnfl_archive)
    lblrtm_members = _validate_archive_members(inputs.lblrtm_archive)
    line_file_members = _validate_archive_members(inputs.line_file_archive)
    _require_archive_paths(
        inputs.lnfl_archive,
        lnfl_members,
        (
            PurePosixPath("lnfl/build/make_lnfl"),
            PurePosixPath("lnfl/build/makefile.common"),
            PurePosixPath("lnfl/src/lnfl.f"),
        ),
    )
    _require_archive_paths(
        inputs.lblrtm_archive,
        lblrtm_members,
        (
            PurePosixPath("lblrtm/build/make_lblrtm"),
            PurePosixPath("lblrtm/build/makefile.common"),
            PurePosixPath("lblrtm/src/lblrtm.f90"),
            LBLRTM_MT_CKD_SOURCE_MEMBER,
        ),
    )
    _require_archive_paths(
        inputs.line_file_archive,
        line_file_members,
        (LINE_FILE_MEMBER,),
    )

    if args.dry_run:
        _print_dry_run_plan(
            inputs=inputs,
            output_root=output_root,
            toolchain=toolchain,
            lblrtm_build_spec=lblrtm_build_spec,
            recipe_schema_version=recipe_schema_version,
            builder_sha256=builder_sha256,
            tape3_source_policy=tape3_source_policy,
            keep_build_tree=args.keep_build_tree,
        )
        return output_root

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ResourceBuildError(f"cannot create output root {output_root}: {exc}") from exc
    build_root = output_root / ".build"
    build_root.mkdir(exist_ok=False)

    line_file_destination = _bundle_path(
        output_root,
        source_records["aer_line_file"],
        label="sources.aer_line_file",
    )
    _extract_line_file(
        inputs.line_file_archive,
        line_file_destination,
        line_file_members,
    )

    lnfl_extraction_root = build_root / "lnfl-source"
    lblrtm_extraction_root = build_root / "lblrtm-source"
    _extract_source_archive(
        inputs.lnfl_archive,
        lnfl_extraction_root,
        lnfl_members,
    )
    _extract_source_archive(
        inputs.lblrtm_archive,
        lblrtm_extraction_root,
        lblrtm_members,
    )
    lnfl_source_root = lnfl_extraction_root / LNFL_TOP_DIRECTORY
    lblrtm_source_root = lblrtm_extraction_root / LBLRTM_TOP_DIRECTORY
    if not lnfl_source_root.is_dir():
        raise ResourceBuildError(f"LNFL archive did not produce {lnfl_source_root}")
    if not lblrtm_source_root.is_dir():
        raise ResourceBuildError(f"LBLRTM archive did not produce {lblrtm_source_root}")

    _stage_mt_ckd_h2o_runtime_resource(
        lblrtm_source_root,
        output_root,
        product_records[MT_CKD_H2O_CONTINUUM_PRODUCT_KEY],
    )
    lnfl_executable = _build_lnfl(lnfl_source_root, toolchain)
    lblrtm_executable = _build_lblrtm(
        lblrtm_source_root,
        toolchain,
        lblrtm_build_spec,
    )
    generated_tape3, tape5, tape6 = _run_lnfl(
        lnfl_executable,
        line_file=line_file_destination,
        run_directory=build_root / "lnfl-run",
    )
    if shared_tape3 is not None:
        _validate_tape3_against_lnfl_run(
            shared_tape3,
            generated_tape3=generated_tape3,
            tape6=tape6,
        )

    for key, source in (
        (LNFL_SOURCE_KEY, inputs.lnfl_archive),
        (LBLRTM_SOURCE_KEY, inputs.lblrtm_archive),
        (LINE_FILE_SOURCE_KEY, inputs.line_file_archive),
    ):
        _copy_to_bundle(
            source,
            _bundle_path(
                output_root,
                source_records[key],
                label=f"sources.{key}",
            ),
        )
    _copy_to_bundle(
        lnfl_executable,
        _bundle_path(
            output_root,
            product_records["lnfl_executable"],
            label="products.lnfl_executable",
        ),
    )
    _copy_to_bundle(
        lblrtm_executable,
        _bundle_path(
            output_root,
            product_records["lblrtm_executable"],
            label="products.lblrtm_executable",
        ),
    )
    _move_to_bundle(
        tape5,
        _bundle_path(
            output_root,
            product_records["lnfl_tape5"],
            label="products.lnfl_tape5",
        ),
    )
    _move_to_bundle(
        tape6,
        _bundle_path(
            output_root,
            product_records["lnfl_tape6"],
            label="products.lnfl_tape6",
        ),
    )
    tape3_product = _bundle_path(
        output_root,
        product_records["tape3"],
        label="products.tape3",
    )
    if shared_tape3 is None:
        _move_to_bundle(generated_tape3, tape3_product)
    else:
        _copy_to_bundle(shared_tape3.path, tape3_product)
        _validate_exact_file_identity(
            tape3_product,
            label="copied shared TAPE3",
            size_bytes=shared_tape3.size_bytes,
            sha256_digest=shared_tape3.sha256,
        )
        copied_semantic = compute_tape3_semantic_sha256(tape3_product)
        copied_scientific = compute_tape3_scientific_sha256(tape3_product)
        if copied_semantic != shared_tape3.semantic_sha256:
            raise ResourceBuildError(
                "the TAPE3 line file changed while it was copied into the bundle"
            )
        if copied_scientific != shared_tape3.scientific_sha256:
            raise ResourceBuildError(
                "the TAPE3 line data changed while the file was copied into the bundle"
            )
    _stage_notices(output_root)
    _write_provenance(
        output_root,
        source_records=source_records,
        product_records=product_records,
        toolchain=toolchain,
        lnfl_executable=lnfl_executable,
        lblrtm_executable=lblrtm_executable,
        lblrtm_build_spec=lblrtm_build_spec,
        recipe_schema_version=recipe_schema_version,
        builder_sha256=builder_sha256,
        tape3_product=tape3_product,
        tape3_source_policy=tape3_source_policy,
    )
    _validate_exact_file_identity(
        builder_path,
        label="builder source after resource build",
        size_bytes=builder_size_bytes,
        sha256_digest=builder_sha256,
    )

    if not args.keep_build_tree:
        _remove_successful_build_tree(
            build_root,
            output_root=output_root,
        )

    print(f"[done] LBLRTM bundle built in {output_root}")
    return output_root


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without performing any filesystem work."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the hash-pinned gtellcorr LNFL v3.2 / LBLRTM v12.17 "
            "resource bundle offline. No network access is used."
        ),
    )
    parser.add_argument(
        "--lnfl-archive",
        required=True,
        type=Path,
        help="path to the pinned aerlnfl_v3.2.tar source archive",
    )
    parser.add_argument(
        "--lblrtm-archive",
        required=True,
        type=Path,
        help="path to the pinned aerlbl_v12.17.tar source archive",
    )
    parser.add_argument(
        "--line-file-archive",
        required=True,
        type=Path,
        help="path to the pinned aer_v_3.9.tgz line-file archive",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="new or empty directory in which to stage the resource bundle",
    )
    parser.add_argument(
        "--netcdf-prefix",
        required=True,
        type=Path,
        help="NetCDF installation prefix containing lib/ and include/",
    )
    parser.add_argument(
        "--sdk-root",
        required=True,
        type=Path,
        help=(
            "explicit macOS SDK root used for Fortran linking "
            "(for example, the result of xcrun --show-sdk-path)"
        ),
    )
    parser.add_argument(
        "--make-command",
        default="make",
        help="GNU make executable name or path (default: make)",
    )
    parser.add_argument(
        "--fortran-compiler",
        default="gfortran",
        help="GNU Fortran executable name or path (default: gfortran)",
    )
    parser.add_argument(
        "--otool-command",
        default=("readelf" if _HOST_SYSTEM == "Linux" else "otool"),
        help="Mach-O inspection executable name or path (default: otool)",
    )
    parser.add_argument(
        "--lblrtm-precision",
        choices=tuple(sorted(_LBLRTM_BUILD_SPECS)),
        default="single",
        help=("LBLRTM arithmetic and TAPE12 precision; LNFL remains single (default: single)"),
    )
    parser.add_argument(
        "--recipe-schema-version",
        choices=(1, 2),
        type=int,
        help=("build recipe schema; defaults to 1 for single precision and 2 for double precision"),
    )
    parser.add_argument(
        "--tape3-source",
        type=Path,
        help=(
            "optional shared TAPE3 whose semantic and scientific identities "
            "must match the pinned product; LNFL is still clean-built and run"
        ),
    )
    parser.add_argument(
        "--keep-build-tree",
        action="store_true",
        help="retain output-root/.build after a successful build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate hashes, archive safety, host, and toolchain; print the "
            "plan without creating files"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate controlled failures to a concise exit status."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        build_resource_bundle(args)
    except ResourceBuildError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
