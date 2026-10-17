"""Command-line interface: ``gtellcorr <command> …``.

The CLI only parses arguments, reads and writes TOML files and calls the public
API; it contains no computation. Commands:

* ``configure`` - write a commented run file for a data directory;
* ``inspect``   - list the exposures gtellcorr would process;
* ``run``       - correct the exposures of a run file;
* ``resources`` - check, freeze, build or download the external resources.

Run ``gtellcorr <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from gtellcorr._version import __version__
from gtellcorr.errors import GtellcorrError

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``gtellcorr`` command and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="gtellcorr",
        description="Forward-model telluric correction for high-resolution spectrographs.",
    )
    parser.add_argument("--version", action="version", version=f"gtellcorr {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="more log output (-v info, -vv debug)"
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    configure = commands.add_parser(
        "configure",
        help="write a commented run-configuration file",
        description=(
            "Write a run configuration (TOML) with every parameter and its documentation, "
            "starting from the defaults for a data directory and instrument."
        ),
    )
    configure.add_argument("-o", "--output", required=True, type=Path, help="file to write")
    configure.add_argument(
        "--directory", required=True, help="directory holding the reduced spectra"
    )
    configure.add_argument("--instrument", help="instrument name (e.g. WIYN/NEID)")
    configure.add_argument("--no-comments", action="store_true", help="omit documentation")
    configure.add_argument("--force", action="store_true", help="overwrite an existing file")
    configure.set_defaults(handler=_cmd_configure)

    inspect = commands.add_parser(
        "inspect",
        help="list the exposures gtellcorr would process in a directory",
        description="Identify the instrument and list the selected exposures with their metadata.",
    )
    inspect.add_argument("directory", type=Path)
    which = inspect.add_mutually_exclusive_group()
    which.add_argument("--instrument", help="instrument name (default: identify from headers)")
    which.add_argument(
        "--instrument-file", type=Path, help="TOML file of your own instrument definition"
    )
    inspect.add_argument("--channel", help="channel (default: the instrument's default channel)")
    inspect.add_argument("--include", action="append", help="keep names containing this token")
    inspect.add_argument("--exclude", action="append", default=[], help="drop names with token")
    inspect.set_defaults(handler=_cmd_inspect)

    runp = commands.add_parser("run", help="correct the exposures of a run file ([data] + [fit])")
    runp.add_argument("config", type=Path, help="run file (TOML) with [data] and [fit] tables")
    runp.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="default: the current directory"
    )
    runp.add_argument(
        "--reuse",
        choices=("matching", "any", "never"),
        default="matching",
        help="reuse existing exposure results: matching provenance (default), any, never",
    )
    runp.add_argument("--no-plot", action="store_true", help="skip the figures")
    runp.add_argument(
        "--no-stellar", action="store_true", help="skip the stellar treatment (strong mask only)"
    )
    runp.set_defaults(handler=_cmd_run)

    res = commands.add_parser(
        "resources", help="check, build or download LBLRTM and the atmosphere reference files"
    )
    res_cmd = res.add_subparsers(dest="resources_command", metavar="<action>", required=True)
    res_cmd.add_parser("status", help="show what is installed and usable").set_defaults(
        handler=_cmd_resources_status
    )
    verify = res_cmd.add_parser(
        "verify", help="check the LBLRTM bundle: file sizes, checksums and versions"
    )
    verify.add_argument("--bundle", type=Path, help="bundle directory (default: settings)")
    verify.set_defaults(handler=_cmd_resources_verify)
    freeze = res_cmd.add_parser(
        "freeze",
        help="record the file sizes and checksums of a bundle copied from elsewhere",
    )
    freeze.add_argument("--bundle", type=Path, help="bundle directory (default: settings)")
    freeze.add_argument("--force", action="store_true", help="replace an existing record")
    freeze.set_defaults(handler=_cmd_resources_freeze)
    build = res_cmd.add_parser("build", help="compile LNFL/LBLRTM from the source archives")
    build.add_argument("--lnfl-archive", type=Path, required=True, help="aerlnfl_v3.2.tar")
    build.add_argument("--lblrtm-archive", type=Path, required=True, help="aerlbl_v12.17.tar")
    build.add_argument("--line-file-archive", type=Path, required=True, help="aer_v_3.9.tgz")
    build.add_argument("--output", type=Path, help="bundle directory (default: settings)")
    build.add_argument(
        "--netcdf-prefix", type=Path, help="existing NetCDF-Fortran + gfortran prefix"
    )
    build.set_defaults(handler=_cmd_resources_build)
    fetch = res_cmd.add_parser("fetch", help="download the MIPAS and CelesTrak reference files")
    fetch.add_argument("what", choices=("mipas", "celestrak", "all"), nargs="?", default="all")
    fetch.set_defaults(handler=_cmd_resources_fetch)
    return parser


def _cmd_resources_status(args: argparse.Namespace) -> int:
    """``resources status``: print what is installed; exit 1 if anything is missing."""
    from gtellcorr.telluric.resources import status

    report = status()
    bundle = report["lblrtm_bundle"]
    print(f"resource root: {report['resource_root']}")
    print(f"cache:         {report['cache_dir']}")
    mark = "ok " if bundle["ok"] else "MISSING/INVALID"
    print(f"LBLRTM bundle: {mark} {bundle['path']}")
    if bundle["ok"]:
        print(f"               {bundle['description']}")
        if args.verbose:
            print(f"               manifest checksum {bundle['fingerprint']}")
    else:
        print(f"               {bundle['problem']}")
    for key, label in (("mipas", "MIPAS"), ("celestrak", "CelesTrak")):
        entry = report[key]
        print(f"{label + ':':<14} {'ok ' if entry['ok'] else 'MISSING'} {entry['path'] or ''}")
        if "problem" in entry:
            print(f"               {entry['problem']}")
    ok = bundle["ok"] and report["mipas"]["ok"] and report["celestrak"]["ok"]
    return 0 if ok else 1


def _cmd_resources_verify(args: argparse.Namespace) -> int:
    """``resources verify``: check the LBLRTM bundle (raises if it fails)."""
    from gtellcorr.telluric.resources import describe, verify

    print(f"LBLRTM bundle ok: {describe(verify(bundle=args.bundle))}")
    return 0


def _cmd_resources_freeze(args: argparse.Namespace) -> int:
    """``resources freeze``: record the file sizes and checksums of a bundle."""
    from gtellcorr.settings import Settings
    from gtellcorr.telluric.resources import freeze

    bundle = args.bundle or Settings.load().lblrtm_bundle
    assert bundle is not None  # Settings always resolves a default bundle path
    print(f"wrote {freeze(bundle, force=args.force)}")
    return 0


def _cmd_resources_build(args: argparse.Namespace) -> int:
    """``resources build``: compile LNFL and LBLRTM, build the line file, verify."""
    from gtellcorr.telluric.resources import build, describe

    verified = build(
        lnfl_archive=args.lnfl_archive,
        lblrtm_archive=args.lblrtm_archive,
        line_file_archive=args.line_file_archive,
        output=args.output,
        netcdf_prefix=args.netcdf_prefix,
    )
    print(f"built and verified: {describe(verified)}")
    return 0


def _cmd_resources_fetch(args: argparse.Namespace) -> int:
    """``resources fetch``: download the MIPAS and/or CelesTrak reference files."""
    from gtellcorr.data.atmosphere.reference import fetch_celestrak, fetch_mipas
    from gtellcorr.settings import Settings
    from gtellcorr.telluric.resources import load_setup

    folder = Settings.load().resource_root / "atmosphere"
    if args.what in ("mipas", "all"):
        ref = load_setup()["atmosphere_references"]["rfm_mipas_midlatitude_night"]
        got = fetch_mipas(
            folder, file_name=ref["file_name"], sha256=ref["sha256"], url=ref["source_url"]
        )
        print(f"MIPAS:     {got.path}")
    if args.what in ("celestrak", "all"):
        got = fetch_celestrak(folder)
        print(f"CelesTrak: {got.path} (sha256 {got.sha256[:12]}…)")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """``inspect``: identify the instrument and print one line per exposure."""
    from gtellcorr.data.observation import Observation, load_instrument_file

    instrument = args.instrument
    if args.instrument_file is not None:
        instrument = load_instrument_file(args.instrument_file)
    obs = Observation.from_directory(
        args.directory,
        instrument=instrument,
        channel=args.channel,
        include=args.include,
        exclude=args.exclude,
    )
    spec = obs.instrument
    print(f"instrument: {spec.name}   channel: {obs.channel}   exposures: {len(obs)}")
    print(f"directory:  {obs.directory}")
    print(
        f"orders:     {len(spec.orders)} ({spec.orders[0]}..{spec.orders[-1]})   "
        f"reader: {spec.spectral.reader}"
    )
    print()
    print(f"{'exposure':<42} {'mid-exposure UTC':<26} {'airmass':>8} {'R':>9}  site  object")
    for e in obs.exposures:
        site = "hdr" if set(e.site_sources.values()) == {"header"} else "mixed"
        flag = "*" if e.airmass_source != "fits_header" else " "
        print(
            f"{e.name:<42} {e.time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]:<26} "
            f"{e.airmass:>7.4f}{flag} {e.resolution:>9.0f}  {site:<5} {e.object}"
        )
    if any(e.airmass_source != "fits_header" for e in obs.exposures):
        print("\n* airmass missing or invalid in the header; 1.5 is used")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """``run``: load the run file and run the pipeline."""
    from gtellcorr.config.run_config import RunConfig
    from gtellcorr.data.observation import Observation
    from gtellcorr.pipeline import Pipeline

    config = RunConfig.load(args.config)
    observation = Observation.from_config(config.data)
    pipeline = Pipeline(
        observation,
        config.fit,
        output_dir=args.output_dir,
        reuse=args.reuse,
        plot=not args.no_plot,
        apply_stellar_treatment=not args.no_stellar,
    )
    result = pipeline.run()
    print(f"results: {result.root}")
    print(f"summary: {result.data_dir / 'summary.csv'}")
    return 0


def _cmd_configure(args: argparse.Namespace) -> int:
    """``configure``: write a run file with default parameters and their documentation."""
    from gtellcorr.config.run_config import RunConfig
    from gtellcorr.config.toml_io import save_run_config

    if args.output.exists() and not args.force:
        print(f"error: {args.output} exists; use --force to overwrite", file=sys.stderr)
        return 1
    data = {"directory": args.directory}
    if args.instrument:
        data["instrument"] = args.instrument
    config = RunConfig.from_dict({"data": data})
    save_run_config(config, args.output, comments=not args.no_comments)
    print(f"wrote {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``gtellcorr`` command; returns the exit status.

    gtellcorr errors are printed as one ``error:`` line (exit status 1) instead of
    a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except GtellcorrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
