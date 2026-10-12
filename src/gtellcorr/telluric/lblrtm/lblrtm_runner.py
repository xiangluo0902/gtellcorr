"""Execute one isolated, auditable LBLRTM v12.17 direct-RT run.

The runner is intentionally limited to process isolation and byte-level
provenance.  It does not parse TAPE12, construct a wavelength grid or clip a
transmission spectrum.

Execution-critical resources are accepted only from a
``validate_lblrtm_resource_bundle`` result.  Each resource is opened without
following symlinks, copied and hashed from that same source descriptor into a
new private-workspace inode, and verified again after LBLRTM exits.
"""

from __future__ import annotations

import logging
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

from .lblrtm_data_models import LblrtmRunResult
from .lblrtm_errors import (
    LblrtmExecutionError,
    LblrtmInputError,
    LblrtmOutputError,
    LblrtmResourceIntegrityError,
)
from .lblrtm_resources import ResolvedLblrtmResourceFiles

log = logging.getLogger(__name__)


def _level(verbose: bool) -> int:
    """Log level of the run messages: INFO if *verbose*, else DEBUG."""
    return logging.INFO if verbose else logging.DEBUG


__all__ = ["execute_lblrtm"]


# These are project safety budgets, not AER format maxima.
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 300.0
_MAXIMUM_TIMEOUT_SECONDS: Final[float] = 3_600.0
_MAX_TAPE5_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_TAPE6_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_TAPE12_BYTES: Final[int] = 512 * 1024 * 1024
_MAX_STDOUT_BYTES: Final[int] = 1024 * 1024
_MAX_STDERR_BYTES: Final[int] = 1024 * 1024
_MAX_EXCEPTION_STREAM_BYTES: Final[int] = 8 * 1024
_MAX_EXCEPTION_TAPE6_BYTES: Final[int] = 8 * 1024
_COPY_CHUNK_BYTES: Final[int] = 1024 * 1024
_WORKSPACE_CREATION_ATTEMPTS: Final[int] = 128

_EXECUTABLE_FILENAME: Final[str] = "lblrtm"
_TAPE3_FILENAME: Final[str] = "TAPE3"
_MT_CKD_FILENAME: Final[str] = "absco-ref_wv-mt-ckd.nc"
_TAPE5_FILENAME: Final[str] = "TAPE5"
_TAPE6_FILENAME: Final[str] = "TAPE6"
_TAPE12_FILENAME: Final[str] = "TAPE12"
_STDOUT_FILENAME: Final[str] = "stdout.log"
_STDERR_FILENAME: Final[str] = "stderr.log"
_PRIVATE_TEMP_DIRECTORY_NAME: Final[str] = "tmp"

_LBLRTM_TIME_FOOTER_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    rb"^0 TIME  LEAVING LBLRTM (?P<run_time>.{15})"
    rb" TOTAL(?P<total_time>.{15})$"
)
_LBLRTM_F15_4_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"^ *[+-]?\d+\.\d{4}$")
_LBLRTM_VERSION_FOOTER_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    rb"^0 Modules and versions used in this calculation:$"
)
_LBLRTM_TAPE6_FATAL_LINES: Final[frozenset[bytes]] = frozenset(
    {
        # AER LBLRTM v12.17 src/lblrtm.f90:2648.
        b"FATAL ERROR ON NSPCRT SPECIFICATION",
    }
)

_O_CLOEXEC: Final[int] = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY: Final[int] = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK: Final[int] = getattr(os, "O_NONBLOCK", 0)


@dataclass(frozen=True, slots=True)
class _RuntimeResourceIdentity:
    """Frozen evidence and staging policy for one runtime resource."""

    logical_name: str
    staged_filename: str
    expected_size_bytes: int
    expected_sha256: str
    staged_mode: int


@dataclass(frozen=True, slots=True)
class _StagedRuntimeFile:
    """Identity of one verified private-workspace resource snapshot."""

    contract: _RuntimeResourceIdentity
    device: int
    inode: int


@dataclass(slots=True)
class _IsolatedWorkspace:
    """One private child directory retained by descriptor during a run."""

    path: Path
    name: str
    parent_fd: int
    directory_fd: int
    device: int
    inode: int
    keep: bool


@dataclass(frozen=True, slots=True)
class _LblrtmDiagnostics:
    """Bounded process diagnostics collected from workspace log files."""

    stdout: str
    stderr: str
    tape6_excerpt: str
    stdout_problem: str | None
    stderr_problem: str | None


def execute_lblrtm(
    tape5_text: str,
    resources: ResolvedLblrtmResourceFiles,
    *,
    work_root: str | os.PathLike[str] | None = None,
    timeout_s: float | None = None,
    keep_work_directory: bool = False,
    verbose: bool = False,
) -> LblrtmRunResult:
    """Run LBLRTM once in a fresh workspace and return raw audited outputs.

    Parameters
    ----------
    tape5_text
        Complete caller-rendered TAPE5 text.  Its ASCII encoding is written
        byte-for-byte without newline or whitespace normalization.
    resources
        A ``ResolvedLblrtmResourceFiles`` returned by
        ``validate_lblrtm_resource_bundle``.  Resolve-only resources are
        rejected.
    work_root
        Optional existing directory under which the private run directory is
        created.  The default is the per-user system temporary directory.
    timeout_s
        Positive finite execution timeout.  ``None`` selects the project
        default; it never means an unlimited run.
    keep_work_directory
        Preserve the unique workspace for audit/debugging when true.
    verbose
        Log the workspace and the run time at INFO (otherwise at DEBUG); full
        diagnostics remain in the returned result or raised exception.
    """

    (
        tape5_bytes,
        normalized_timeout_seconds,
        normalized_work_root,
    ) = _validate_lblrtm_execution_request(
        tape5_text,
        work_root=work_root,
        timeout_s=timeout_s,
        keep_work_directory=keep_work_directory,
        verbose=verbose,
    )
    validated_resources = _require_validated_resources(resources)
    resource_identities = _runtime_resource_identities_from_evidence(validated_resources)
    resource_fingerprint = validated_resources.manifest_fingerprint
    if resource_fingerprint is None:  # pragma: no cover - token invariant
        raise LblrtmResourceIntegrityError(
            "internal error: the checked LBLRTM bundle has no manifest checksum"
        )

    with _isolated_lblrtm_work_directory(
        normalized_work_root,
        keep=keep_work_directory,
    ) as workspace:
        log.log(_level(verbose), "LBLRTM workspace: %s", workspace.path)

        staged_resources = _stage_lblrtm_runtime_resources(
            validated_resources,
            resource_identities,
            workspace,
        )
        staged_tape5 = _write_tape5_to_work_directory(
            tape5_bytes,
            workspace,
        )
        tape5_sha256 = staged_tape5.contract.expected_sha256
        _require_fresh_lblrtm_output_paths(workspace)

        completed_process: subprocess.CompletedProcess[bytes] | None = None
        elapsed_seconds = 0.0
        run_failure: BaseException | None = None
        try:
            completed_process, elapsed_seconds = _run_lblrtm_subprocess(
                workspace,
                timeout_seconds=normalized_timeout_seconds,
            )
        except (
            LblrtmExecutionError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            run_failure = exc

        diagnostics = _collect_lblrtm_diagnostics(workspace)
        execution_failure: LblrtmExecutionError | None = None
        execution_cause: BaseException | None = None
        if run_failure is not None:
            if isinstance(run_failure, subprocess.TimeoutExpired):
                message = (
                    f"LBLRTM exceeded the finite timeout of {normalized_timeout_seconds:g} seconds"
                )
            elif isinstance(run_failure, LblrtmExecutionError):
                message = str(run_failure)
            else:
                message = f"cannot execute staged LBLRTM: {run_failure}"
            execution_failure = LblrtmExecutionError(
                f"{message}\n{_format_bounded_diagnostics(diagnostics)}"
            )
            execution_cause = run_failure
        elif completed_process is None:  # pragma: no cover - control invariant
            raise AssertionError("LBLRTM process result was not recorded")
        elif diagnostics.stdout_problem is not None:
            if diagnostics.stdout_problem.startswith(f"{_STDOUT_FILENAME} exceeds "):
                stdout_message = "LBLRTM stdout exceeds the post-run acceptance limit"
            else:
                stdout_message = "LBLRTM stdout diagnostic artifact was rejected"
            execution_failure = LblrtmExecutionError(
                f"{stdout_message}: {diagnostics.stdout_problem}\n"
                f"{_format_bounded_diagnostics(diagnostics)}"
            )
        elif diagnostics.stderr_problem is not None:
            if diagnostics.stderr_problem.startswith(f"{_STDERR_FILENAME} exceeds "):
                stderr_message = "LBLRTM stderr exceeds the post-run acceptance limit"
            else:
                stderr_message = "LBLRTM stderr diagnostic artifact was rejected"
            execution_failure = LblrtmExecutionError(
                f"{stderr_message}: {diagnostics.stderr_problem}\n"
                f"{_format_bounded_diagnostics(diagnostics)}"
            )
        elif completed_process.returncode != 0:
            execution_failure = LblrtmExecutionError(
                "LBLRTM exited with non-zero return code "
                f"{completed_process.returncode}\n"
                f"{_format_bounded_diagnostics(diagnostics)}"
            )

        integrity_failure: LblrtmResourceIntegrityError | None = None
        try:
            _verify_staged_inputs_after_execution(
                (*staged_resources, staged_tape5),
                workspace,
            )
        except LblrtmResourceIntegrityError as exc:
            integrity_failure = exc

        if execution_failure is not None:
            if integrity_failure is not None:
                _add_exception_note(
                    execution_failure,
                    "Post-run staged-input integrity verification also "
                    f"failed: {integrity_failure}",
                )
            if execution_cause is not None:
                raise execution_failure from execution_cause
            raise execution_failure
        if integrity_failure is not None:
            _add_exception_note(
                integrity_failure,
                _format_bounded_diagnostics(diagnostics),
            )
            raise integrity_failure
        if completed_process is None:  # pragma: no cover - control invariant
            raise AssertionError("LBLRTM process result was not recorded")

        try:
            tape6_bytes = _read_bounded_workspace_file(
                workspace,
                _TAPE6_FILENAME,
                maximum_bytes=_MAX_TAPE6_BYTES,
                allow_empty=False,
                artifact_name="TAPE6",
            )
            _validate_tape6_completion(tape6_bytes)
            try:
                tape6_text = tape6_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise LblrtmOutputError("TAPE6 must contain only ASCII bytes") from exc
            tape12_bytes = _read_tape12_output_bytes(workspace)
        except (LblrtmExecutionError, LblrtmOutputError) as exc:
            _add_exception_note(
                exc,
                _format_bounded_diagnostics(diagnostics),
            )
            raise

        result = LblrtmRunResult(
            tape12_bytes=tape12_bytes,
            tape6_text=tape6_text,
            stdout=diagnostics.stdout,
            stderr=diagnostics.stderr,
            return_code=completed_process.returncode,
            elapsed_seconds=elapsed_seconds,
            resource_fingerprint=resource_fingerprint,
            tape5_sha256=tape5_sha256,
            tape6_sha256=sha256(tape6_bytes).hexdigest(),
            tape12_sha256=sha256(tape12_bytes).hexdigest(),
            work_directory=workspace.path if keep_work_directory else None,
        )
        log.log(_level(verbose), "LBLRTM completed in %.3f s", elapsed_seconds)
        return result


def _add_exception_note(exception: BaseException, note: str) -> None:
    """Attach secondary evidence on both Python 3.10 and Python 3.11+."""

    add_note = getattr(exception, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    if exception.args and isinstance(exception.args[0], str):
        exception.args = (
            f"{exception.args[0]}\n{note}",
            *exception.args[1:],
        )
    else:
        exception.args = (*exception.args, note)


def _is_boolean_scalar(value: object) -> bool:
    """Recognize Python and NumPy booleans without importing NumPy."""

    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__ == "numpy" and value_type.__name__ in {"bool", "bool_"}
    )


def _validate_lblrtm_execution_request(
    tape5_text: object,
    *,
    work_root: object,
    timeout_s: object,
    keep_work_directory: object,
    verbose: object,
) -> tuple[bytes, float, Path | None]:
    """Validate public runner controls before creating any filesystem state."""

    if not isinstance(tape5_text, str):
        raise LblrtmInputError("tape5_text must be a string")
    if not tape5_text:
        raise LblrtmInputError("tape5_text must be non-empty")
    if "\0" in tape5_text:
        raise LblrtmInputError("tape5_text must not contain NUL characters")
    if "\r" in tape5_text:
        raise LblrtmInputError("tape5_text must use LF line endings and must not contain CR")
    if not tape5_text.endswith("\n"):
        raise LblrtmInputError("tape5_text must end with a terminal LF")
    final_record = tape5_text.split("\n")[-2]
    if final_record.rstrip(" ") != "%":
        raise LblrtmInputError("the final TAPE5 record must be '%' with optional trailing spaces")
    try:
        tape5_bytes = tape5_text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise LblrtmInputError("tape5_text must contain only ASCII characters") from exc
    if len(tape5_bytes) > _MAX_TAPE5_BYTES:
        raise LblrtmInputError(f"TAPE5 exceeds the {_MAX_TAPE5_BYTES}-byte input limit")

    if timeout_s is None:
        timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
    else:
        if _is_boolean_scalar(timeout_s):
            raise LblrtmInputError("timeout_s must be a real number, not bool")
        try:
            timeout_seconds = float(timeout_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError("timeout_s must be convertible to a finite float") from exc
        if not math.isfinite(timeout_seconds):
            raise LblrtmInputError("timeout_s must be finite")
        if timeout_seconds <= 0.0:
            raise LblrtmInputError("timeout_s must be greater than zero")
        if timeout_seconds > _MAXIMUM_TIMEOUT_SECONDS:
            raise LblrtmInputError(
                f"timeout_s must not exceed {_MAXIMUM_TIMEOUT_SECONDS:g} seconds"
            )

    for field_name, value in (
        ("keep_work_directory", keep_work_directory),
        ("verbose", verbose),
    ):
        if not isinstance(value, bool):
            raise LblrtmInputError(f"{field_name} must be bool")

    normalized_work_root: Path | None
    if work_root is None:
        normalized_work_root = None
    else:
        if isinstance(work_root, (bytes, bytearray, memoryview)):
            raise LblrtmInputError("work_root must be str or os.PathLike[str], not bytes")
        try:
            normalized_work_root = Path(work_root).expanduser()
        except (TypeError, ValueError, OSError) as exc:
            raise LblrtmInputError("work_root must be convertible to a filesystem path") from exc

    return tape5_bytes, timeout_seconds, normalized_work_root


def _require_validated_resources(
    resources: object,
) -> ResolvedLblrtmResourceFiles:
    """Require the unforgeable complete-bundle validation state."""

    if not isinstance(resources, ResolvedLblrtmResourceFiles):
        raise LblrtmInputError("resources must be a ResolvedLblrtmResourceFiles instance")
    return resources.require_fully_validated()


def _runtime_resource_identities_from_evidence(
    resources: ResolvedLblrtmResourceFiles,
) -> tuple[_RuntimeResourceIdentity, ...]:
    """Snapshot execution-critical size/SHA evidence from validator v2."""

    evidence = resources.validation_evidence
    if evidence is None:  # pragma: no cover - require_fully_validated invariant
        raise LblrtmResourceIntegrityError(
            "internal error: the LBLRTM bundle check result is missing"
        )
    specifications = (
        (
            "products.lblrtm_executable",
            _EXECUTABLE_FILENAME,
            0o500,
        ),
        ("products.tape3", _TAPE3_FILENAME, 0o400),
        (
            "products.mt_ckd_h2o_continuum_coefficients",
            _MT_CKD_FILENAME,
            0o400,
        ),
    )
    identities: list[_RuntimeResourceIdentity] = []
    for logical_name, staged_filename, staged_mode in specifications:
        size_key = f"{logical_name}.size_bytes"
        digest_key = f"{logical_name}.sha256"
        try:
            size_text = evidence[size_key]
            digest = evidence[digest_key]
        except KeyError as exc:  # pragma: no cover - validator invariant
            raise LblrtmResourceIntegrityError(
                f"internal error: the LBLRTM bundle check did not record {exc.args[0]!r}"
            ) from exc
        if not size_text.isdigit():
            raise LblrtmResourceIntegrityError(
                f"internal error: the LBLRTM bundle check recorded no size for {size_key}"
            )
        expected_size = int(size_text)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise LblrtmResourceIntegrityError(
                f"internal error: the LBLRTM bundle check recorded no SHA-256 for {digest_key}"
            )
        identities.append(
            _RuntimeResourceIdentity(
                logical_name=logical_name,
                staged_filename=staged_filename,
                expected_size_bytes=expected_size,
                expected_sha256=digest,
                staged_mode=staged_mode,
            )
        )
    return tuple(identities)


@contextmanager
def _isolated_lblrtm_work_directory(
    work_root: Path | None,
    *,
    keep: bool,
) -> Iterator[_IsolatedWorkspace]:
    """Create and own one fresh mode-0700 workspace without global chdir."""

    _require_descriptor_safety_support()
    parent_path = _normalize_work_root(work_root)
    parent_flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
    parent_fd: int | None = None
    workspace_fd: int | None = None
    workspace: _IsolatedWorkspace | None = None
    workspace_name: str | None = None
    directory_created = False
    try:
        parent_fd = os.open(parent_path, parent_flags)
        parent_status = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_status.st_mode):
            raise LblrtmInputError(f"work_root is not a directory: {parent_path}")

        for _ in range(_WORKSPACE_CREATION_ATTEMPTS):
            workspace_name = "gtellcorr-lblrtm-" + secrets.token_hex(16)
            try:
                os.mkdir(workspace_name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            directory_created = True
            break
        else:  # pragma: no cover - cryptographic collision guard
            raise LblrtmExecutionError("cannot allocate a unique LBLRTM workspace")

        if workspace_name is None:  # pragma: no cover - loop invariant
            raise AssertionError("workspace name was not allocated")
        workspace_fd = os.open(
            workspace_name,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=parent_fd,
        )
        os.fchmod(workspace_fd, 0o700)
        workspace_status = os.fstat(workspace_fd)
        if (
            not stat.S_ISDIR(workspace_status.st_mode)
            or stat.S_IMODE(workspace_status.st_mode) != 0o700
        ):
            raise LblrtmExecutionError("LBLRTM workspace is not a private mode-0700 directory")
        workspace = _IsolatedWorkspace(
            path=parent_path / workspace_name,
            name=workspace_name,
            parent_fd=parent_fd,
            directory_fd=workspace_fd,
            device=workspace_status.st_dev,
            inode=workspace_status.st_ino,
            keep=keep,
        )
    except BaseException as setup_exception:
        cleanup_failure: BaseException | None = None
        if directory_created and parent_fd is not None:
            try:
                if workspace is not None:
                    _securely_remove_lblrtm_work_directory(workspace)
                elif workspace_name is not None:
                    # No input has been staged before setup completes.  A
                    # descriptor-relative rmdir therefore fails safely if an
                    # unexpected entry appeared, rather than deleting it.
                    os.rmdir(workspace_name, dir_fd=parent_fd)
            except BaseException as exc:
                cleanup_failure = exc
        if workspace_fd is not None:
            with suppress(OSError):
                os.close(workspace_fd)
        if parent_fd is not None:
            with suppress(OSError):
                os.close(parent_fd)
        if isinstance(
            setup_exception,
            (LblrtmInputError, LblrtmExecutionError),
        ):
            failure = setup_exception
        elif isinstance(setup_exception, OSError):
            failure = LblrtmExecutionError(
                f"cannot create isolated LBLRTM workspace: {setup_exception}"
            )
        else:
            failure = setup_exception
        if cleanup_failure is not None:
            _add_exception_note(
                failure,
                f"Partial LBLRTM workspace cleanup also failed: {cleanup_failure}",
            )
        if failure is setup_exception:
            raise
        raise failure from setup_exception

    if workspace is None:  # pragma: no cover - setup invariant
        raise AssertionError("LBLRTM workspace was not constructed")
    try:
        yield workspace
    finally:
        active_exception = sys.exc_info()[1]
        cleanup_failure: BaseException | None = None
        if workspace.keep and active_exception is not None:
            _add_exception_note(active_exception, f"LBLRTM workspace retained at {workspace.path}")
        if not workspace.keep:
            try:
                _securely_remove_lblrtm_work_directory(workspace)
            except BaseException as exc:  # preserve cleanup evidence
                cleanup_failure = exc
        if workspace_fd is not None:
            with suppress(OSError):
                os.close(workspace_fd)
        if parent_fd is not None:
            with suppress(OSError):
                os.close(parent_fd)
        if cleanup_failure is not None:
            if active_exception is not None:
                _add_exception_note(
                    active_exception, f"LBLRTM workspace cleanup also failed: {cleanup_failure}"
                )
            else:
                raise LblrtmExecutionError(
                    "LBLRTM completed but its private workspace could not be "
                    f"removed: {cleanup_failure}"
                ) from cleanup_failure


def _normalize_work_root(work_root: Path | None) -> Path:
    """Return an existing canonical directory while rejecting final symlinks."""

    candidate = Path(tempfile.gettempdir()) if work_root is None else work_root
    try:
        if candidate.is_symlink():
            raise LblrtmInputError(f"work_root must not be a symlink: {candidate}")
        normalized = candidate.resolve(strict=True)
    except LblrtmInputError:
        raise
    except (OSError, RuntimeError) as exc:
        raise LblrtmInputError(
            f"work_root does not resolve to an existing directory: {candidate}"
        ) from exc
    if not normalized.is_dir():
        raise LblrtmInputError(f"work_root is not a directory: {normalized}")
    return normalized


def _require_descriptor_safety_support() -> None:
    """Fail closed on platforms lacking required descriptor safety flags."""

    if _O_DIRECTORY == 0 or _O_NOFOLLOW == 0:
        raise LblrtmExecutionError(
            "this platform lacks O_DIRECTORY/O_NOFOLLOW, needed to copy LBLRTM inputs safely"
        )
    required_functions = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    if any(function not in os.supports_dir_fd for function in required_functions):
        raise LblrtmExecutionError(
            "this platform lacks dir_fd support, needed to copy LBLRTM inputs safely"
        )


def _stage_lblrtm_runtime_resources(
    resources: ResolvedLblrtmResourceFiles,
    identities: tuple[_RuntimeResourceIdentity, ...],
    workspace: _IsolatedWorkspace,
) -> tuple[_StagedRuntimeFile, ...]:
    """Copy executable, TAPE3, and MT_CKD into distinct verified inodes."""

    root = resources.root
    if not root.is_absolute():
        raise LblrtmResourceIntegrityError("validated resource root must be absolute")
    root_fd: int | None = None
    staged: list[_StagedRuntimeFile] = []
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise LblrtmResourceIntegrityError(
                f"validated resource root is no longer a directory: {root}"
            )
        for identity in identities:
            source_path = resources.require(identity.logical_name)
            staged.append(
                _stage_verified_runtime_file(
                    source_path,
                    resource_root=root,
                    resource_root_fd=root_fd,
                    identity=identity,
                    workspace=workspace,
                )
            )
    except LblrtmResourceIntegrityError:
        raise
    except (KeyError, OSError, ValueError) as exc:
        raise LblrtmResourceIntegrityError(
            f"cannot stage validated LBLRTM runtime resources: {exc}"
        ) from exc
    finally:
        if root_fd is not None:
            with suppress(OSError):
                os.close(root_fd)
    return tuple(staged)


def _stage_verified_runtime_file(
    source_path: Path,
    *,
    resource_root: Path,
    resource_root_fd: int,
    identity: _RuntimeResourceIdentity,
    workspace: _IsolatedWorkspace,
) -> _StagedRuntimeFile:
    """Copy and verify one resource using the same opened source descriptor."""

    try:
        relative_path = source_path.relative_to(resource_root)
    except ValueError as exc:
        raise LblrtmResourceIntegrityError(
            f"{identity.logical_name} escapes the validated resource root"
        ) from exc
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise LblrtmResourceIntegrityError(f"{identity.logical_name} has an unsafe resource path")

    source_fd: int | None = None
    destination_fd: int | None = None
    opened_directory_fds: list[int] = []
    destination_created = False
    try:
        containing_directory_fd = resource_root_fd
        for component in relative_path.parts[:-1]:
            next_directory_fd = os.open(
                component,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=containing_directory_fd,
            )
            opened_directory_fds.append(next_directory_fd)
            containing_directory_fd = next_directory_fd

        source_fd = os.open(
            relative_path.parts[-1],
            os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
            dir_fd=containing_directory_fd,
        )
        source_status_before = os.fstat(source_fd)
        _validate_regular_source_status(
            source_status_before,
            identity=identity,
        )

        destination_fd = os.open(
            identity.staged_filename,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
            identity.staged_mode,
            dir_fd=workspace.directory_fd,
        )
        destination_created = True
        os.fchmod(destination_fd, identity.staged_mode)
        destination_status_before = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_status_before.st_mode)
            or destination_status_before.st_nlink != 1
        ):
            raise LblrtmResourceIntegrityError(
                f"the working copy {identity.staged_filename} is not a single regular file"
            )
        if (
            source_status_before.st_dev == destination_status_before.st_dev
            and source_status_before.st_ino == destination_status_before.st_ino
        ):
            raise LblrtmResourceIntegrityError(
                f"the working copy {identity.staged_filename} is a hard link to the bundle file"
            )

        copied_digest = sha256()
        copied_size = 0
        while True:
            chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied_size += len(chunk)
            if copied_size > identity.expected_size_bytes:
                raise LblrtmResourceIntegrityError(
                    f"{identity.logical_name} grew while being copied for LBLRTM"
                )
            copied_digest.update(chunk)
            _write_all(destination_fd, chunk)

        source_status_after = os.fstat(source_fd)
        if _source_identity_snapshot(source_status_before) != (
            _source_identity_snapshot(source_status_after)
        ):
            raise LblrtmResourceIntegrityError(
                f"{identity.logical_name} changed while being copied for LBLRTM"
            )
        if copied_size != identity.expected_size_bytes:
            raise LblrtmResourceIntegrityError(
                f"the copy of {identity.logical_name} for LBLRTM has the wrong size: "
                f"expected {identity.expected_size_bytes}, got {copied_size}"
            )
        copied_sha256 = copied_digest.hexdigest()
        if copied_sha256 != identity.expected_sha256:
            raise LblrtmResourceIntegrityError(
                f"the copy of {identity.logical_name} for LBLRTM differs from the bundle "
                f"(SHA-256): expected {identity.expected_sha256}, got {copied_sha256}"
            )

        os.fsync(destination_fd)
        destination_status = os.fstat(destination_fd)
        _validate_staged_destination_status(
            destination_status,
            identity=identity,
        )
        destination_sha256 = _hash_open_file_descriptor(
            destination_fd,
            maximum_bytes=identity.expected_size_bytes,
        )
        if destination_sha256 != identity.expected_sha256:
            raise LblrtmResourceIntegrityError(
                f"the working copy {identity.staged_filename} differs from the bundle (SHA-256): "
                f"expected {identity.expected_sha256}, got "
                f"{destination_sha256}"
            )
        return _StagedRuntimeFile(
            contract=identity,
            device=destination_status.st_dev,
            inode=destination_status.st_ino,
        )
    except LblrtmResourceIntegrityError:
        raise
    except OSError as exc:
        raise LblrtmResourceIntegrityError(f"cannot stage {identity.logical_name}: {exc}") from exc
    finally:
        if destination_fd is not None:
            with suppress(OSError):
                os.close(destination_fd)
        # Keep only a completely verified destination.  During exception
        # unwinding the active exception is non-None.
        if destination_created and sys.exc_info()[0] is not None:
            with suppress(OSError):
                os.unlink(
                    identity.staged_filename,
                    dir_fd=workspace.directory_fd,
                )
        if source_fd is not None:
            with suppress(OSError):
                os.close(source_fd)
        for directory_fd in reversed(opened_directory_fds):
            with suppress(OSError):
                os.close(directory_fd)


def _validate_regular_source_status(
    source_status: os.stat_result,
    *,
    identity: _RuntimeResourceIdentity,
) -> None:
    """Validate the source file identity before copying any bytes."""

    if not stat.S_ISREG(source_status.st_mode):
        raise LblrtmResourceIntegrityError(f"{identity.logical_name} is no longer a regular file")
    if source_status.st_size != identity.expected_size_bytes:
        raise LblrtmResourceIntegrityError(
            f"{identity.logical_name} size changed after validation: expected "
            f"{identity.expected_size_bytes}, got {source_status.st_size}"
        )


def _source_identity_snapshot(status: os.stat_result) -> tuple[int, ...]:
    """Return metadata fields that expose concurrent source replacement/write."""

    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _validate_staged_destination_status(
    destination_status: os.stat_result,
    *,
    identity: _RuntimeResourceIdentity,
) -> None:
    """Require exact type, link count, mode, and evidence size."""

    if not stat.S_ISREG(destination_status.st_mode):
        raise LblrtmResourceIntegrityError(
            f"the working copy {identity.staged_filename} is not a regular file"
        )
    if destination_status.st_nlink != 1:
        raise LblrtmResourceIntegrityError(
            f"the working copy {identity.staged_filename} must not be hard-linked"
        )
    if stat.S_IMODE(destination_status.st_mode) != identity.staged_mode:
        raise LblrtmResourceIntegrityError(
            f"the working copy {identity.staged_filename} has permissions other than "
            f"{identity.staged_mode:#o}"
        )
    if destination_status.st_size != identity.expected_size_bytes:
        raise LblrtmResourceIntegrityError(
            f"the working copy {identity.staged_filename} has the wrong size: expected "
            f"{identity.expected_size_bytes}, got "
            f"{destination_status.st_size}"
        )


def _write_all(file_descriptor: int, payload: bytes) -> None:
    """Write a complete chunk or fail on a zero/short-progress write."""

    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("a write returned zero bytes while copying a file for LBLRTM")
        offset += written


def _hash_open_file_descriptor(
    file_descriptor: int,
    *,
    maximum_bytes: int,
) -> str:
    """Hash one open regular-file snapshot without reopening its path."""

    os.lseek(file_descriptor, 0, os.SEEK_SET)
    digest = sha256()
    total_bytes = 0
    while True:
        chunk = os.read(file_descriptor, _COPY_CHUNK_BYTES)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > maximum_bytes:
            raise LblrtmResourceIntegrityError("a file grew while its checksum was being computed")
        digest.update(chunk)
    return digest.hexdigest()


def _write_tape5_to_work_directory(
    tape5_bytes: bytes,
    workspace: _IsolatedWorkspace,
) -> _StagedRuntimeFile:
    """Write the caller's exact ASCII bytes to a new mode-0400 TAPE5."""

    identity = _RuntimeResourceIdentity(
        logical_name="caller TAPE5",
        staged_filename=_TAPE5_FILENAME,
        expected_size_bytes=len(tape5_bytes),
        expected_sha256=sha256(tape5_bytes).hexdigest(),
        staged_mode=0o400,
    )
    file_descriptor: int | None = None
    created = False
    try:
        file_descriptor = os.open(
            _TAPE5_FILENAME,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
            0o400,
            dir_fd=workspace.directory_fd,
        )
        created = True
        os.fchmod(file_descriptor, 0o400)
        _write_all(file_descriptor, tape5_bytes)
        os.fsync(file_descriptor)
        status_result = os.fstat(file_descriptor)
        _validate_staged_destination_status(
            status_result,
            identity=identity,
        )
        actual_digest = _hash_open_file_descriptor(
            file_descriptor,
            maximum_bytes=len(tape5_bytes),
        )
        if actual_digest != identity.expected_sha256:
            raise LblrtmResourceIntegrityError(
                "internal error: the TAPE5 copied for LBLRTM differs from the one written"
            )
        return _StagedRuntimeFile(
            contract=identity,
            device=status_result.st_dev,
            inode=status_result.st_ino,
        )
    except LblrtmResourceIntegrityError:
        raise
    except OSError as exc:
        raise LblrtmExecutionError(f"cannot write exact TAPE5 bytes: {exc}") from exc
    finally:
        if file_descriptor is not None:
            with suppress(OSError):
                os.close(file_descriptor)
        if created and sys.exc_info()[0] is not None:
            with suppress(OSError):
                os.unlink(_TAPE5_FILENAME, dir_fd=workspace.directory_fd)


def _require_fresh_lblrtm_output_paths(
    workspace: _IsolatedWorkspace,
) -> None:
    """Reject impossible pre-existing outputs; never delete stale files."""

    for filename in (_TAPE6_FILENAME, _TAPE12_FILENAME):
        try:
            os.stat(
                filename,
                dir_fd=workspace.directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LblrtmExecutionError(
                f"cannot inspect fresh output path {filename}: {exc}"
            ) from exc
        raise LblrtmExecutionError(f"fresh LBLRTM workspace unexpectedly contains {filename}")


def _run_lblrtm_subprocess(
    workspace: _IsolatedWorkspace,
    *,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[bytes], float]:
    """Execute the staged binary with fixed cwd and file-backed diagnostics."""

    _assert_workspace_identity(workspace)
    try:
        os.mkdir(
            _PRIVATE_TEMP_DIRECTORY_NAME,
            mode=0o700,
            dir_fd=workspace.directory_fd,
        )
    except FileExistsError as exc:
        raise LblrtmExecutionError("fresh LBLRTM workspace unexpectedly contains tmp") from exc
    except OSError as exc:
        raise LblrtmExecutionError(
            f"cannot create private LBLRTM temporary directory: {exc}"
        ) from exc

    stdout_fd = -1
    stderr_fd = -1
    executable_path = workspace.path / _EXECUTABLE_FILENAME
    controlled_environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(workspace.path / _PRIVATE_TEMP_DIRECTORY_NAME),
        "TZ": "UTC",
    }
    started_at = time.monotonic()
    try:
        stdout_fd = _create_workspace_log_file(
            workspace,
            _STDOUT_FILENAME,
        )
        stderr_fd = _create_workspace_log_file(
            workspace,
            _STDERR_FILENAME,
        )
        with (
            os.fdopen(stdout_fd, "wb", buffering=0) as stdout_stream,
            os.fdopen(stderr_fd, "wb", buffering=0) as stderr_stream,
        ):
            stdout_fd = -1
            stderr_fd = -1
            completed_process = subprocess.run(
                [str(executable_path)],
                shell=False,
                cwd=str(workspace.path),
                env=controlled_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout_seconds,
                check=False,
                close_fds=True,
                start_new_session=True,
                restore_signals=True,
                umask=0o077,
            )
    finally:
        elapsed_seconds = time.monotonic() - started_at
        for file_descriptor in (stdout_fd, stderr_fd):
            if file_descriptor >= 0:
                with suppress(OSError):
                    os.close(file_descriptor)
    _assert_workspace_identity(workspace)
    return completed_process, elapsed_seconds


def _create_workspace_log_file(
    workspace: _IsolatedWorkspace,
    filename: str,
) -> int:
    """Create one private, new regular log file and return its descriptor."""

    try:
        file_descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
            0o600,
            dir_fd=workspace.directory_fd,
        )
        os.fchmod(file_descriptor, 0o600)
        status_result = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(status_result.st_mode)
            or status_result.st_nlink != 1
            or stat.S_IMODE(status_result.st_mode) != 0o600
        ):
            raise LblrtmExecutionError(f"{filename} is not a private unique regular file")
        return file_descriptor
    except LblrtmExecutionError:
        if "file_descriptor" in locals():
            with suppress(OSError):
                os.close(file_descriptor)
        raise
    except OSError as exc:
        if "file_descriptor" in locals():
            with suppress(OSError):
                os.close(file_descriptor)
        raise LblrtmExecutionError(
            f"cannot create bounded workspace log {filename}: {exc}"
        ) from exc


def _assert_workspace_identity(workspace: _IsolatedWorkspace) -> None:
    """Ensure the cwd path still names the directory retained by descriptor."""

    try:
        path_status = os.stat(
            workspace.name,
            dir_fd=workspace.parent_fd,
            follow_symlinks=False,
        )
        descriptor_status = os.fstat(workspace.directory_fd)
    except OSError as exc:
        raise LblrtmExecutionError(f"cannot check the LBLRTM working directory: {exc}") from exc
    expected_identity = (workspace.device, workspace.inode)
    if (
        not stat.S_ISDIR(path_status.st_mode)
        or (path_status.st_dev, path_status.st_ino) != expected_identity
        or (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
        )
        != expected_identity
        or stat.S_IMODE(descriptor_status.st_mode) != 0o700
    ):
        raise LblrtmExecutionError("LBLRTM workspace path was replaced or lost private permissions")


def _collect_lblrtm_diagnostics(
    workspace: _IsolatedWorkspace,
) -> _LblrtmDiagnostics:
    """Collect bounded stdout/stderr and an optional bounded TAPE6 excerpt."""

    stdout_bytes, stdout_problem = _read_log_excerpt(
        workspace,
        _STDOUT_FILENAME,
        collection_limit=_MAX_STDOUT_BYTES,
    )
    stderr_bytes, stderr_problem = _read_log_excerpt(
        workspace,
        _STDERR_FILENAME,
        collection_limit=_MAX_STDERR_BYTES,
    )
    tape6_bytes, _ = _read_log_excerpt(
        workspace,
        _TAPE6_FILENAME,
        collection_limit=_MAX_EXCEPTION_TAPE6_BYTES,
        optional=True,
    )
    return _LblrtmDiagnostics(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        tape6_excerpt=tape6_bytes.decode("ascii", errors="replace"),
        stdout_problem=stdout_problem,
        stderr_problem=stderr_problem,
    )


def _read_log_excerpt(
    workspace: _IsolatedWorkspace,
    filename: str,
    *,
    collection_limit: int,
    optional: bool = False,
) -> tuple[bytes, str | None]:
    """Return a bounded log excerpt and any post-run acceptance problem."""

    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(
            filename,
            os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
            dir_fd=workspace.directory_fd,
        )
        status_result = os.fstat(file_descriptor)
        if not stat.S_ISREG(status_result.st_mode) or status_result.st_nlink != 1:
            return (
                f"[{filename} is not a unique regular file]".encode("ascii"),
                f"{filename} is not a unique regular file",
            )
        if status_result.st_size <= collection_limit:
            return (
                _read_open_file_bytes(
                    file_descriptor,
                    maximum_bytes=collection_limit,
                ),
                None,
            )

        excerpt_limit = min(
            collection_limit,
            _MAX_EXCEPTION_STREAM_BYTES,
        )
        head_bytes = excerpt_limit // 2
        tail_bytes = excerpt_limit - head_bytes
        head = os.read(file_descriptor, head_bytes)
        os.lseek(
            file_descriptor,
            max(status_result.st_size - tail_bytes, 0),
            os.SEEK_SET,
        )
        tail = os.read(file_descriptor, tail_bytes)
        marker = (f"\n[{filename} truncated at {status_result.st_size} bytes]\n").encode("ascii")
        return (
            head + marker + tail,
            f"{filename} exceeds the {collection_limit}-byte post-run acceptance limit",
        )
    except FileNotFoundError:
        if optional:
            return b"", None
        return (
            f"[{filename} is missing]".encode("ascii"),
            f"{filename} is missing",
        )
    except (LblrtmOutputError, OSError) as exc:
        return (
            f"[cannot read {filename}: {exc}]".encode(
                "ascii",
                errors="replace",
            ),
            f"cannot read {filename}: {exc}",
        )
    finally:
        if file_descriptor is not None:
            with suppress(OSError):
                os.close(file_descriptor)


def _read_bounded_workspace_file(
    workspace: _IsolatedWorkspace,
    filename: str,
    *,
    maximum_bytes: int,
    allow_empty: bool,
    artifact_name: str,
) -> bytes:
    """Snapshot one no-follow regular workspace file through the same fd."""

    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(
            filename,
            os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
            dir_fd=workspace.directory_fd,
        )
        status_before = os.fstat(file_descriptor)
        if not stat.S_ISREG(status_before.st_mode):
            raise LblrtmOutputError(f"{artifact_name} is not a regular file")
        if status_before.st_nlink != 1:
            raise LblrtmOutputError(f"{artifact_name} must have exactly one hard link")
        if status_before.st_size > maximum_bytes:
            raise LblrtmOutputError(f"{artifact_name} exceeds the {maximum_bytes}-byte limit")
        payload = _read_open_file_bytes(
            file_descriptor,
            maximum_bytes=maximum_bytes,
        )
        status_after = os.fstat(file_descriptor)
        if _source_identity_snapshot(status_before) != (_source_identity_snapshot(status_after)):
            raise LblrtmOutputError(f"{artifact_name} changed while being read")
        if len(payload) != status_before.st_size:
            raise LblrtmOutputError(f"{artifact_name} size changed while being read")
        if not allow_empty and not payload:
            raise LblrtmOutputError(f"{artifact_name} is empty")
        return payload
    except LblrtmOutputError:
        raise
    except FileNotFoundError as exc:
        raise LblrtmOutputError(f"{artifact_name} is missing") from exc
    except (MemoryError, OSError) as exc:
        raise LblrtmOutputError(f"cannot snapshot {artifact_name}: {exc}") from exc
    finally:
        if file_descriptor is not None:
            with suppress(OSError):
                os.close(file_descriptor)


def _read_open_file_bytes(
    file_descriptor: int,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read at most one extra byte so growth cannot evade a byte budget."""

    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = os.read(
            file_descriptor,
            min(_COPY_CHUNK_BYTES, maximum_bytes + 1 - total_bytes),
        )
        if not chunk:
            break
        chunks.append(chunk)
        total_bytes += len(chunk)
        if total_bytes > maximum_bytes:
            raise LblrtmOutputError(f"workspace file exceeds the {maximum_bytes}-byte limit")
        if total_bytes == maximum_bytes:
            extra_byte = os.read(file_descriptor, 1)
            if extra_byte:
                raise LblrtmOutputError(f"workspace file exceeds the {maximum_bytes}-byte limit")
            break
    try:
        return b"".join(chunks)
    except MemoryError as exc:
        raise LblrtmOutputError("cannot allocate a bounded workspace-file snapshot") from exc


def _validate_tape6_completion(tape6_bytes: bytes) -> None:
    """Require official positive completion footers without word blacklists."""

    # NUL is present elsewhere in successful official output.  Match the two
    # source-defined completion records against raw lines so that a NUL cannot
    # splice together an otherwise invalid marker.  The returned text and hash
    # retain every original byte.
    lines = tape6_bytes.splitlines()
    fatal_lines = _LBLRTM_TAPE6_FATAL_LINES.intersection(line.strip(b" \t") for line in lines)
    if fatal_lines:
        fatal_line = sorted(fatal_lines)[0].decode("ascii")
        raise LblrtmExecutionError(f"LBLRTM reported a fatal error in TAPE6: {fatal_line}")
    time_footer_lines: list[int] = []
    for index, line in enumerate(lines):
        time_match = _LBLRTM_TIME_FOOTER_PATTERN.fullmatch(line)
        if time_match is None:
            continue
        if all(
            _LBLRTM_F15_4_PATTERN.fullmatch(time_match.group(field_name)) is not None
            for field_name in ("run_time", "total_time")
        ):
            time_footer_lines.append(index)
    if len(time_footer_lines) != 1:
        raise LblrtmExecutionError(
            "TAPE6 must contain exactly one complete source-format official "
            "'TIME LEAVING LBLRTM ... TOTAL ...' footer"
        )
    version_footer_lines = tuple(
        index
        for index, line in enumerate(lines)
        if _LBLRTM_VERSION_FOOTER_PATTERN.fullmatch(line) is not None
    )
    if len(version_footer_lines) != 1:
        raise LblrtmExecutionError(
            "TAPE6 must contain exactly one complete source-format official modules/version footer"
        )
    if version_footer_lines[0] <= time_footer_lines[0]:
        raise LblrtmExecutionError(
            "TAPE6 official modules/version footer must follow the completion footer"
        )


def _read_tape12_output_bytes(
    workspace: _IsolatedWorkspace,
) -> bytes:
    """Return raw, bounded TAPE12 bytes without parsing or numerical policy."""

    return _read_bounded_workspace_file(
        workspace,
        _TAPE12_FILENAME,
        maximum_bytes=_MAX_TAPE12_BYTES,
        allow_empty=False,
        artifact_name="TAPE12",
    )


def _verify_staged_inputs_after_execution(
    staged_inputs: tuple[_StagedRuntimeFile, ...],
    workspace: _IsolatedWorkspace,
) -> None:
    """Recheck runtime resources and exact TAPE5 after process completion."""

    for staged in staged_inputs:
        identity = staged.contract
        file_descriptor: int | None = None
        try:
            file_descriptor = os.open(
                identity.staged_filename,
                os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
                dir_fd=workspace.directory_fd,
            )
            status_result = os.fstat(file_descriptor)
            if status_result.st_dev != staged.device or status_result.st_ino != staged.inode:
                raise LblrtmResourceIntegrityError(
                    f"the working copy {identity.staged_filename} was replaced while LBLRTM ran"
                )
            _validate_staged_destination_status(
                status_result,
                identity=identity,
            )
            actual_digest = _hash_open_file_descriptor(
                file_descriptor,
                maximum_bytes=identity.expected_size_bytes,
            )
            if actual_digest != identity.expected_sha256:
                raise LblrtmResourceIntegrityError(
                    f"the working copy {identity.staged_filename} changed while LBLRTM ran"
                )
        except LblrtmResourceIntegrityError:
            raise
        except OSError as exc:
            raise LblrtmResourceIntegrityError(
                f"cannot re-check the working copy {identity.staged_filename}: {exc}"
            ) from exc
        finally:
            if file_descriptor is not None:
                with suppress(OSError):
                    os.close(file_descriptor)


def _format_bounded_diagnostics(
    diagnostics: _LblrtmDiagnostics,
) -> str:
    """Format bounded exception evidence without generic fatal-word parsing."""

    sections = (
        ("stdout", diagnostics.stdout),
        ("stderr", diagnostics.stderr),
        ("TAPE6 excerpt", diagnostics.tape6_excerpt),
    )
    rendered: list[str] = []
    for label, value in sections:
        if not value:
            continue
        bounded_value = value
        if len(bounded_value.encode("utf-8", errors="replace")) > (_MAX_EXCEPTION_STREAM_BYTES):
            encoded = bounded_value.encode("utf-8", errors="replace")
            bounded_value = encoded[:_MAX_EXCEPTION_STREAM_BYTES].decode("utf-8", errors="replace")
            bounded_value += "\n[diagnostic excerpt truncated]"
        rendered.append(f"{label}:\n{bounded_value}")
    return "\n".join(rendered) if rendered else "no process diagnostics"


def _securely_remove_lblrtm_work_directory(
    workspace: _IsolatedWorkspace,
) -> None:
    """Delete only this retained workspace using descriptor-relative names."""

    _assert_workspace_identity(workspace)
    _unlink_directory_contents(workspace.directory_fd)
    with os.scandir(workspace.directory_fd) as entries:
        remaining = tuple(entries)
    if remaining:
        raise OSError("workspace gained entries during cleanup")
    path_status = os.stat(
        workspace.name,
        dir_fd=workspace.parent_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(path_status.st_mode)
        or path_status.st_dev != workspace.device
        or path_status.st_ino != workspace.inode
    ):
        raise OSError("the LBLRTM working directory was replaced before it could be removed")
    os.rmdir(workspace.name, dir_fd=workspace.parent_fd)


def _unlink_directory_contents(directory_fd: int) -> None:
    """Recursively unlink names without following symlinks."""

    with os.scandir(directory_fd) as entries:
        entry_names = tuple(entry.name for entry in entries)
    for entry_name in entry_names:
        status_result = os.stat(
            entry_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(status_result.st_mode):
            child_fd = os.open(
                entry_name,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=directory_fd,
            )
            try:
                child_status = os.fstat(child_fd)
                if (
                    child_status.st_dev != status_result.st_dev
                    or child_status.st_ino != status_result.st_ino
                ):
                    raise OSError(f"directory {entry_name!r} changed during cleanup")
                _unlink_directory_contents(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(entry_name, dir_fd=directory_fd)
        else:
            os.unlink(entry_name, dir_fd=directory_fd)
