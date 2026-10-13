"""Locate, verify, freeze and build the LBLRTM resource bundle.

Layout below ``Settings.resource_root`` (all locations can be overridden)::

    <resource_root>/
      lblrtm/gnu_double/            the bundle (Settings.lblrtm_bundle)
        bin/ data/ provenance/ sources/ third_party_notices/
        gtellcorr_manifest.json     local manifest (written by freeze)
      runtime/lblrtm-toolchain/     conda toolchain used to compile the bundle
      atmosphere/                   MIPAS and CelesTrak reference files

The numerical validation itself is
:func:`gtellcorr.telluric.lblrtm.validate_lblrtm_resource_bundle`; this module adds
the check against the LBLRTM setup description shipped with gtellcorr (see
:mod:`.setup`).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtellcorr._logging import get_logger
from gtellcorr.errors import ResourceError
from gtellcorr.settings import Settings
from gtellcorr.telluric.lblrtm import (
    DEFAULT_MANIFEST_FILENAME,
    LblrtmResourceError,
    ResolvedLblrtmResourceFiles,
    load_lblrtm_resource_manifest,
    validate_lblrtm_resource_bundle,
)
from gtellcorr.telluric.lblrtm.lblrtm_errors import LblrtmError
from gtellcorr.telluric.resources.setup import check_setup, load_setup

__all__ = [
    "LOCAL_MANIFEST",
    "VerifiedResources",
    "build",
    "freeze",
    "status",
    "toolchain_packages",
    "verify",
]

log = get_logger(__name__)

LOCAL_MANIFEST = "gtellcorr_manifest.json"
TOOLCHAIN_DIR = "lblrtm-toolchain"
_LBLRTM_PACKAGE = Path(__file__).resolve().parents[1] / "lblrtm"


@dataclass(frozen=True)
class VerifiedResources:
    """A bundle that passed both levels of validation."""

    bundle: Path
    resources: ResolvedLblrtmResourceFiles
    """Resolved files, as consumed by :mod:`gtellcorr.telluric.lblrtm`."""
    local_fingerprint: str
    setup_id: str
    """Name of the LBLRTM setup the bundle matches."""
    description: str
    """Program and data versions, in words."""

    @property
    def contract_id(self) -> str:
        """Name used by gtellcorr 0.9.0 for :attr:`setup_id` (removed in 1.0)."""
        return self.setup_id


def _settings(settings: Settings | None) -> Settings:
    """*settings*, or the loaded ones."""
    return settings if settings is not None else Settings.load()


def _bundle(settings: Settings) -> Path:
    """The configured bundle directory (``Settings.lblrtm_bundle``)."""
    assert settings.lblrtm_bundle is not None  # set by Settings.__post_init__
    return settings.lblrtm_bundle


# ---------------------------------------------------------------------------
# freeze / verify
# ---------------------------------------------------------------------------
def freeze(bundle: str | os.PathLike[str], *, force: bool = False) -> Path:
    """Measure *bundle* and write its local manifest; return the manifest path.

    Uses :mod:`._freeze` with the shipped double-precision manifest as template,
    then checks the result against the LBLRTM setup description.
    """
    from gtellcorr.telluric.resources import _freeze

    root = Path(bundle).expanduser().resolve()
    target = root / LOCAL_MANIFEST
    if target.exists() and not force:
        raise ResourceError(f"{target} exists; pass force=True to re-measure the bundle")
    try:
        data = _freeze.freeze_manifest(root, _LBLRTM_PACKAGE / DEFAULT_MANIFEST_FILENAME)
    except (LblrtmError, LblrtmResourceError, OSError) as exc:
        raise ResourceError(f"cannot freeze {root}: {exc}") from exc
    check_setup(data, load_setup())
    _freeze._write_json_atomically(target, data, force=True)
    return target


def verify(
    settings: Settings | None = None, *, bundle: str | os.PathLike[str] | None = None
) -> VerifiedResources:
    """Check the configured bundle against its local manifest and the LBLRTM setup."""
    root = (
        Path(bundle).expanduser().resolve() if bundle is not None else _bundle(_settings(settings))
    )
    manifest_path = root / LOCAL_MANIFEST
    if not root.is_dir():
        raise ResourceError(
            f"LBLRTM bundle not found: {root}. Build one with 'gtellcorr resources build' "
            "or set lblrtm_bundle in the settings."
        )
    if not manifest_path.is_file():
        raise ResourceError(
            f"{root} has no {LOCAL_MANIFEST} (the record of its file sizes and checksums); "
            "if the bundle was copied here, run 'gtellcorr resources freeze' once"
        )
    setup = load_setup()
    try:
        manifest = load_lblrtm_resource_manifest(manifest_path)
        check_setup(manifest.data, setup)
        resources = validate_lblrtm_resource_bundle(manifest, root)
    except ResourceError:
        raise
    except (LblrtmError, LblrtmResourceError, OSError) as exc:
        raise ResourceError(f"LBLRTM bundle {root} failed validation: {exc}") from exc
    return VerifiedResources(
        bundle=root,
        resources=resources,
        local_fingerprint=manifest.fingerprint,
        setup_id=str(setup["setup_id"]),
        description=str(setup["description"]),
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    from gtellcorr.telluric.lblrtm import compute_file_sha256

    return str(compute_file_sha256(path))


def status(settings: Settings | None = None) -> dict[str, Any]:
    """A plain report of every resource and whether it is usable (never raises)."""
    s = _settings(settings)
    report: dict[str, Any] = {
        "resource_root": str(s.resource_root),
        "cache_dir": str(s.cache_dir),
        "lblrtm_bundle": {"path": str(_bundle(s))},
    }
    try:
        verified = verify(s)
        report["lblrtm_bundle"].update(
            ok=True,
            fingerprint=verified.local_fingerprint,
            setup=verified.setup_id,
            description=verified.description,
        )
    except ResourceError as exc:
        report["lblrtm_bundle"].update(ok=False, problem=str(exc))
    mipas = load_setup()["atmosphere_references"]["rfm_mipas_midlatitude_night"]
    mipas_path = s.resource_root / "atmosphere" / mipas["file_name"]
    report["mipas"] = {
        "path": str(mipas_path),
        "ok": mipas_path.is_file() and _sha256(mipas_path) == mipas["sha256"],
    }
    from gtellcorr.data.atmosphere.reference import latest_celestrak

    try:
        snapshot = latest_celestrak(s.resource_root / "atmosphere")
        report["celestrak"] = {"path": str(snapshot.path), "ok": True}
    except ResourceError as exc:
        report["celestrak"] = {"path": None, "ok": False, "problem": str(exc)}
    return report


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def toolchain_packages(system: str | None = None) -> tuple[str, ...]:
    """conda-forge packages of the isolated compile toolchain for *system*."""
    system = system or platform.system()
    base = ("netcdf-fortran", "libnetcdf", "gfortran", "make")
    if system == "Darwin":
        # conda's gfortran drives clang as assembler/linker on macOS.
        return (*base, "clang", "cctools", "ld64")
    return (*base, "binutils")


def _ensure_toolchain(prefix: Path) -> Path:
    """The toolchain at *prefix*, created with conda / mamba (conda-forge) if it has
    no ``nf-config`` yet."""
    if (prefix / "bin" / "nf-config").exists():
        return prefix
    conda = shutil.which("mamba") or shutil.which("conda")
    if conda is None:
        raise ResourceError(
            "no conda/mamba found to create the LBLRTM toolchain; install one, or pass "
            "netcdf_prefix= pointing to a NetCDF-Fortran installation with gfortran"
        )
    command = [conda, "create", "-y", "-p", str(prefix), "--override-channels", "-c", "conda-forge"]
    log.info("creating LBLRTM toolchain in %s", prefix)
    subprocess.run([*command, *toolchain_packages()], check=True)
    return prefix


def _fortran_compiler(prefix: Path) -> str:
    """The toolchain's gfortran (a target-prefixed one first), else one on PATH."""
    candidates = [*sorted((prefix / "bin").glob("*-gfortran")), prefix / "bin" / "gfortran"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("gfortran")
    if found is None:
        raise ResourceError(f"no gfortran in {prefix / 'bin'} or on PATH")
    return found


def build(
    settings: Settings | None = None,
    *,
    lnfl_archive: str | os.PathLike[str],
    lblrtm_archive: str | os.PathLike[str],
    line_file_archive: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    netcdf_prefix: str | os.PathLike[str] | None = None,
    keep_build_tree: bool = False,
) -> VerifiedResources:
    """Compile LNFL + LBLRTM from the source archives and record the bundle.

    The builder checks the archives against their SHA-256 in the setup description. Without
    *netcdf_prefix*, an isolated conda-forge toolchain is created (once) under
    ``<resource_root>/runtime/lblrtm-toolchain``. The build runs in a subprocess
    with the toolchain first on ``PATH``.
    """
    s = _settings(settings)
    out = Path(output).expanduser().resolve() if output is not None else _bundle(s)
    if out.exists() and any(out.iterdir()):
        raise ResourceError(f"output directory is not empty: {out}")
    prefix = (
        Path(netcdf_prefix).expanduser().resolve()
        if netcdf_prefix is not None
        else _ensure_toolchain(s.resource_root / "runtime" / TOOLCHAIN_DIR)
    )
    args = [
        sys.executable,
        "-m",
        "gtellcorr.telluric.resources._build",
        "--lnfl-archive", str(Path(lnfl_archive).expanduser()),
        "--lblrtm-archive", str(Path(lblrtm_archive).expanduser()),
        "--line-file-archive", str(Path(line_file_archive).expanduser()),
        "--output-root", str(out),
        "--netcdf-prefix", str(prefix),
        "--fortran-compiler", _fortran_compiler(prefix),
        "--lblrtm-precision", "double",
        "--recipe-schema-version", "2",
    ]  # fmt: skip
    make = prefix / "bin" / "make"
    if make.exists():
        args += ["--make-command", str(make)]
    if platform.system() == "Darwin":
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True
        )
        args += ["--sdk-root", sdk.stdout.strip()]
    else:
        args += ["--sdk-root", "/"]
    if keep_build_tree:
        args.append("--keep-build-tree")
    env = {**os.environ, "PATH": f"{prefix / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    log.info("building LBLRTM resources into %s", out)
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
        raise ResourceError(f"LBLRTM resource build failed:\n{tail}")
    freeze(out, force=True)
    return verify(s, bundle=out)


def describe(verified: VerifiedResources) -> str:
    """Bundle directory and, on the next line, its program and data versions (CLI)."""
    return f"{verified.bundle}\n  {verified.description}"
