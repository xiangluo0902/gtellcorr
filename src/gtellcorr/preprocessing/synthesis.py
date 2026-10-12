"""Synthetic stellar templates (pySME / iSpec).

* :func:`stellar_template` synthesises a continuum-normalised spectrum at rest (air
  nm) over a window and Doppler-shifts it to the working frame.
* :func:`generate_template` builds the full-band rest-frame template used by the
  ``template`` treatment: for pySME the VALD line list is sliced per chunk; the
  chunks are synthesised in parallel worker processes (resumable chunk cache),
  broadened to the instrument resolution (:func:`broaden_template`) and stitched.
* :func:`load_template` and :func:`stellar_model_for_order` read a template and
  place it on an order's wavelengths at an exposure's RV.

All parameters are explicit (:class:`Synthesis`); worker processes receive them
with the job instead of reading global configuration.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.errors import ConfigError
from gtellcorr.preprocessing._ispec import import_ispec
from gtellcorr.preprocessing.config import BROAD_LINE_WINDOWS_OPTICAL
from gtellcorr.preprocessing.windows import C_KMS

if TYPE_CHECKING:
    from gtellcorr.data.observation.target import Star
    from gtellcorr.preprocessing.config import StellarModelConfig
    from gtellcorr.settings import Settings

__all__ = [
    "StellarParameters",
    "Synthesis",
    "broaden_template",
    "generate_template",
    "load_template",
    "stellar_model_for_order",
    "stellar_template",
    "synthesis_resolution",
    "template_filename",
]

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM / sigma of a Gaussian

#: VALD lines within this distance of a chunk enter its PySME synthesis [nm].
SLICE_MARGIN_NM = 3.0
#: Lines with far-reaching wings (every H I line, and the lines inside
#: ``BROAD_LINE_WINDOWS_OPTICAL``) enter every chunk within this distance [nm]. Their
#: wings, and their effect on the formation of nearby lines, reach well beyond
#: ``SLICE_MARGIN_NM``: without them, the chunks next to H-alpha disagree with it by
#: up to 1e-3.
BROAD_LINE_MARGIN_NM = 50.0
#: Each chunk is synthesised with this margin on both sides [nm].
CHUNK_MARGIN_NM = 2.0
#: Version of the chunk cache layout; bumped when chunks are built differently.
_CHUNK_CACHE_VERSION = "v2"


@dataclass(frozen=True)
class StellarParameters:
    """Parameters of the star that a synthesis uses.

    Where a value is ``None``, iSpec estimates it (``vmic`` and ``vmac`` from Teff,
    log g and [M/H]; ``alpha`` from [M/H]); pySME takes ``vmic`` = 1 km/s and ``vmac`` = 0
    and uses neither ``alpha`` nor ``limb_darkening_coeff``.
    """

    name: str
    """Star name; the first part of the template file name."""
    teff: float
    """Effective temperature [K]."""
    logg: float
    """Surface gravity, log10(g / cm s^-2)."""
    feh: float
    """Metallicity [M/H]."""
    vsini_kms: float
    """Projected rotational velocity [km/s]."""
    vsys_kms: float = 0.0
    """Systemic velocity [km/s]; :func:`stellar_template`'s default shift."""
    vmic_kms: float | None = None
    """Microturbulence [km/s]."""
    vmac_kms: float | None = None
    """Macroturbulence [km/s]."""
    alpha: float | None = None
    """Alpha-element enhancement [alpha/Fe] (iSpec only; ignored by pySME without a
    warning)."""
    limb_darkening_coeff: float = 0.6
    """Linear limb-darkening coefficient of the rotational broadening (iSpec only)."""

    @classmethod
    def from_star(cls, star: Star) -> StellarParameters:
        """The target's parameters; raises :class:`~gtellcorr.errors.ConfigError` if
        Teff, log g, [M/H] or vsini is missing (Vsys defaults to 0)."""
        missing = [k for k in ("teff", "logg", "feh", "vsini_kms") if getattr(star, k) is None]
        if missing:
            raise ConfigError(
                f"target {star.name!r}: {', '.join(missing)} needed for a stellar template"
            )
        return cls(
            name=star.name,
            teff=float(star.teff),  # type: ignore[arg-type]
            logg=float(star.logg),  # type: ignore[arg-type]
            feh=float(star.feh),  # type: ignore[arg-type]
            vsini_kms=float(star.vsini_kms),  # type: ignore[arg-type]
            vsys_kms=float(star.vsys_kms or 0.0),
            vmic_kms=star.vmic_kms,
            vmac_kms=star.vmac_kms,
            alpha=star.alpha,
            limb_darkening_coeff=float(star.limb_darkening_coeff),
        )


@dataclass(frozen=True)
class Synthesis:
    """Everything a synthesis needs; picklable, so it can be sent to worker processes."""

    star: StellarParameters
    backend: str
    """``"pysme"`` or ``"ispec"``."""
    linelist: str | None
    """VALD line list (pySME)."""
    ispec_dir: str | None
    """iSpec installation directory (iSpec backend)."""
    pysme: dict[str, Any] = field(default_factory=dict)
    """``stellar_model.pysme`` settings."""
    ispec: dict[str, Any] = field(default_factory=dict)
    """``stellar_model.ispec`` settings."""

    @classmethod
    def from_config(cls, star: Star, config: StellarModelConfig, settings: Settings) -> Synthesis:
        """From the target, the ``stellar_model`` configuration and the settings
        (``stellar_linelist``, ``ispec_dir``)."""
        return cls(
            star=StellarParameters.from_star(star),
            backend=config.backend,
            linelist=None if settings.stellar_linelist is None else str(settings.stellar_linelist),
            ispec_dir=None if settings.ispec_dir is None else str(settings.ispec_dir),
            pysme=config.pysme.to_dict(),
            ispec=config.ispec.to_dict(),
        )


# ---------------------------------------------------------------------------
# backends: rest-frame synthesis -> (wave_nm_air, flux_normalised)
# ---------------------------------------------------------------------------
def _ispec_synth(syn: Synthesis, lo: float, hi: float, linelist: str | None) -> tuple[Array, Array]:
    """iSpec synthesis over *lo*-*hi* nm on a ``wave_step_nm`` grid, with iSpec's own
    line list (lines shallower than ``min_depth`` dropped; *linelist* is ignored),
    rotation and macroturbulence, and its instrumental broadening ``resolution``."""
    st, s = syn.star, syn.ispec
    ispec = import_ispec(syn.ispec_dir, fork=True)
    d = str(Path(str(syn.ispec_dir)).expanduser())
    teff, logg, feh = st.teff, st.logg, st.feh
    alpha = st.alpha if st.alpha is not None else ispec.determine_abundance_enchancements(feh)
    vmic = st.vmic_kms if st.vmic_kms is not None else ispec.estimate_vmic(teff, logg, feh)
    vmac = st.vmac_kms if st.vmac_kms is not None else ispec.estimate_vmac(teff, logg, feh)
    atomic = ispec.read_atomic_linelist(
        f"{d}/input/linelists/transitions/{s['linelist']}/atomic_lines.tsv",
        wave_base=lo,
        wave_top=hi,
    )
    atomic = atomic[atomic["theoretical_depth"] >= s["min_depth"]]
    isotopes = ispec.read_isotope_data(f"{d}/input/isotopes/{s['isotopes']}")
    abund = ispec.read_solar_abundances(f"{d}/input/abundances/{s['abundances']}/stdatom.dat")
    layers = ispec.load_modeled_layers_pack(f"{d}/input/atmospheres/{s['atmosphere']}/")
    target = {"teff": teff, "logg": logg, "MH": feh, "alpha": alpha}
    atm = ispec.interpolate_atmosphere_layers(layers, target, code=s["code"])
    spec = ispec.create_spectrum_structure(np.arange(lo, hi, s["wave_step_nm"]))
    spec["flux"] = ispec.generate_spectrum(
        spec["waveobs"], atm, teff, logg, feh, alpha, atomic, isotopes, abund, None,
        microturbulence_vel=vmic, macroturbulence=vmac, vsini=st.vsini_kms,
        limb_darkening_coeff=st.limb_darkening_coeff, R=s.get("resolution", 500000),
        code=s["code"],
    )  # fmt: skip
    return np.asarray(spec["waveobs"], float), np.asarray(spec["flux"], float)


def _pysme_synth(syn: Synthesis, lo: float, hi: float, linelist: str | None) -> tuple[Array, Array]:
    """pySME synthesis over *lo*-*hi* nm from the VALD list *linelist* (default
    ``syn.linelist``), optionally cut at ``line_strength_cut`` in
    ``log gf - theta * chi`` (theta = 5040 / Teff, chi in eV). Atmosphere grid
    ``atmosphere`` (default ``marcs2012``), abundance pattern ``abundances`` (default
    ``grevesse2007``) scaled by [M/H], optional NLTE grids ``nlte``. No instrumental
    broadening; the grid is ``wave_step_nm`` or PySME's adaptive one."""
    from pysme.abund import Abund
    from pysme.linelist.vald import ValdFile
    from pysme.sme import SME_Structure
    from pysme.synthesize import synthesize_spectrum

    st, s = syn.star, syn.pysme
    path = linelist or syn.linelist
    if path is None:
        raise ConfigError(
            "pySME synthesis needs a VALD line list; set stellar_linelist in the settings"
        )
    sme = SME_Structure()
    # the abundance pattern first: [M/H] (``sme.monh``) scales it
    sme.abund = Abund(monh=st.feh, pattern=s.get("abundances", "grevesse2007"))
    sme.teff, sme.logg, sme.monh = st.teff, st.logg, st.feh
    sme.vmic = 1.0 if st.vmic_kms is None else st.vmic_kms
    sme.vmac = 0.0 if st.vmac_kms is None else st.vmac_kms
    sme.vsini = st.vsini_kms
    ll = ValdFile(path)
    cut = s.get("line_strength_cut")
    if cut is not None:
        strength = np.asarray(ll["gflog"], float) - (5040.0 / st.teff) * np.asarray(
            ll["excit"], float
        )
        ll = ll[np.nonzero(strength >= float(cut))[0]]
    sme.linelist = ll
    sme.wran = [lo * 10.0, hi * 10.0]
    step = s.get("wave_step_nm")
    if step is not None:  # a fixed output grid; otherwise PySME's adaptive one
        sme.wave = [np.arange(lo * 10.0, hi * 10.0, float(step) * 10.0)]
    atmo = s.get("atmosphere", "marcs2012")
    sme.atmo.source = atmo if atmo.endswith(".sav") else atmo + ".sav"
    sme.atmo.method = "grid"
    for elem, grid in (s.get("nlte") or {}).items():
        sme.nlte.set_nlte(elem, grid)
    sme = synthesize_spectrum(sme)
    return np.asarray(sme.wave[0], float) / 10.0, np.asarray(sme.synth[0], float)


_BACKENDS = {"ispec": _ispec_synth, "pysme": _pysme_synth}


def stellar_template(
    syn: Synthesis,
    wave_range_nm: tuple[float, float],
    *,
    rv_kms: float | None = None,
    backend: str | None = None,
    wave_margin_nm: float = 2.0,
    linelist: str | None = None,
) -> tuple[Array, Array]:
    """Continuum-normalised synthesis shifted by ``rv_kms`` (default: Vsys).

    Returns ``(wave_nm, flux)`` (air nm) over ``wave_range_nm`` widened by
    ``wave_margin_nm`` on both sides. The rest-frame synthesis is padded by the shift
    plus 0.2 nm, so the returned range is covered whatever the RV. *backend*
    overrides ``syn.backend``; *linelist* replaces ``syn.linelist`` (pySME only).
    """
    rv = syn.star.vsys_kms if rv_kms is None else rv_kms
    name = syn.backend if backend is None else backend
    lo = float(wave_range_nm[0]) - wave_margin_nm
    hi = float(wave_range_nm[1]) + wave_margin_nm
    pad = (abs(rv) / C_KMS) * hi + 0.2
    if name not in _BACKENDS:
        raise ConfigError(f"unknown stellar-model backend {name!r}; use 'ispec' | 'pysme'")
    w_rest, f = _BACKENDS[name](syn, lo - pad, hi + pad, linelist)
    w = w_rest * (1.0 + rv / C_KMS)
    win = (w >= lo) & (w <= hi)
    return w[win], f[win]


# ---------------------------------------------------------------------------
# full-band template
# ---------------------------------------------------------------------------
def _backend_label(syn: Synthesis) -> str:
    """Backend tag of a template file name: ``pySME-VALD`` or ``iSpec-<code>``."""
    if str(syn.backend).lower() == "pysme":
        return "pySME-VALD"
    return f"iSpec-{syn.ispec.get('code', 'spectrum')}"


def template_filename(
    syn: Synthesis, *, wave_range_nm: tuple[float, float], resolution: float
) -> str:
    """Template file name from the star, backend, *resolution* and *wave_range_nm*, e.g.
    ``TOI-2076_pySME-VALD_Teff5180_logg4.608_..._rest.dat``.

    The name is also the key of the template's chunk cache (:func:`generate_template`);
    inputs not in it do not cause a rebuild.
    """
    st = syn.star
    lo, hi = wave_range_nm
    return (
        f"{st.name}_{_backend_label(syn)}"
        f"_Teff{float(st.teff):.0f}"
        f"_logg{st.logg}"
        f"_MH{float(st.feh):+0.1f}"
        f"_vsini{st.vsini_kms}kms"
        f"_R{round(float(resolution))}"
        f"_{round(float(lo))}-{round(float(hi))}nm"
        "_rest.dat"
    )


def synthesis_resolution(syn: Synthesis) -> float:
    """Resolving power of the raw synthesis: iSpec applies ``ispec.resolution`` itself
    (0 = none); PySME applies no instrumental broadening (infinite)."""
    if str(syn.backend).lower() == "ispec":
        r = float(syn.ispec.get("resolution", 500000) or 0.0)
        return r if r > 0 else np.inf
    return np.inf


def _is_broad_line(species: str, wave_nm: float) -> bool:
    """Whether a VALD line gets ``BROAD_LINE_MARGIN_NM`` (H I, or a strong broad line)."""
    return species == "H 1" or any(lo <= wave_nm <= hi for lo, hi in BROAD_LINE_WINDOWS_OPTICAL)


def _slice_vald(
    full_path: str, chunks: list[tuple[int, int]], slice_dir: Path
) -> dict[tuple[int, int], str]:
    """One pass over the full VALD list -> a sub-list per chunk.

    A chunk gets the lines within ``SLICE_MARGIN_NM`` of it, and the broad lines
    (:func:`_is_broad_line`) within ``BROAD_LINE_MARGIN_NM``. Each sub-list keeps the
    VALD header (after any leading ``WARNING`` lines) and footer, with the
    four-line records of the "long" VALD format in between. If every sub-list already
    exists in *slice_dir*, they are reused as they are.
    """
    slice_dir.mkdir(parents=True, exist_ok=True)
    paths = {c: str(slice_dir / f"ll_{c[0]}_{c[1]}.vald") for c in chunks}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    with open(full_path, errors="ignore") as fh:
        lines = fh.readlines()
    while lines and lines[0].lstrip().startswith("WARNING"):
        lines = lines[1:]
    header = lines[:2]
    term = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("* oscillator") or line.lstrip().startswith("References:")
        ),
        len(lines),
    )
    data, footer = lines[2:term], lines[term:]
    handles = {c: open(p, "w") for c, p in paths.items()}  # noqa: SIM115 - closed below
    try:
        for fh in handles.values():
            fh.writelines(header)
        for i in range(0, len(data) - 3, 4):
            first = data[i]
            if not first.startswith("'"):
                continue
            fields = first.split(",")
            try:
                wl_nm = float(fields[1]) / 10.0
            except (IndexError, ValueError):
                continue
            species = fields[0].strip().strip("'")
            reach = BROAD_LINE_MARGIN_NM if _is_broad_line(species, wl_nm) else SLICE_MARGIN_NM
            block = data[i : i + 4]
            for (lo, hi), fh in handles.items():
                if lo - reach <= wl_nm <= hi + reach:
                    fh.writelines(block)
        for fh in handles.values():
            fh.writelines(footer)
    finally:
        for fh in handles.values():
            fh.close()
    return paths


def _synth_chunk(
    args: tuple[Synthesis, int, int, str | None, float, str],
) -> tuple[int, int, int, float]:
    """Worker: synthesise one chunk at rest, broaden it to *resolution* and write it.

    pySME gets the chunk's slice of the VALD list; iSpec reads its own line list. The
    broadening (:func:`broaden_template`) starts from :func:`synthesis_resolution`.
    """
    syn, lo, hi, llpath, resolution, outpath = args
    w, f = stellar_template(
        syn, (lo, hi), rv_kms=0.0, wave_margin_nm=CHUNK_MARGIN_NM, linelist=llpath
    )
    f_r = broaden_template(
        w, f, from_resolution=synthesis_resolution(syn), to_resolution=resolution
    )
    np.savetxt(outpath, np.column_stack([w, f_r]), fmt="%.6f %.6f")
    return (lo, hi, len(w), float(np.min(f_r)))


def generate_template(
    syn: Synthesis,
    out_path: str | os.PathLike[str],
    *,
    wave_range_nm: tuple[float, float],
    resolution: float,
    chunk_nm: float = 25.0,
    workers: int = 12,
) -> Path:
    """Generate (or reuse) the full-band rest-frame template at ``out_path``.

    The band is synthesised in ``chunk_nm`` chunks by parallel worker processes, with
    the backend chosen by ``syn.backend``: pySME with a per-chunk slice of the VALD
    list (``stellar_linelist`` setting), or iSpec with its own line list and code
    (``ispec_dir`` setting, ``stellar_model.ispec``). Each chunk is broadened to
    *resolution* with :func:`broaden_template`, and the chunks are joined with
    :func:`_stitch`. Chunk edges are whole nanometres: the range is truncated to
    integers and split every ``int(chunk_nm)`` nm.

    Chunks (and the VALD slices) are cached under
    ``<template dir>/_fullband_chunks/<template stem>/<cache version>``, so an
    interrupted run resumes and templates of different names never share chunks. An
    existing file at ``out_path`` is returned without checks.

    Reuse goes by the file name alone (:func:`template_filename`), which records the
    star name, backend (and iSpec code), Teff, log g, [M/H], vsini, R and wavelength
    range. A change of any other input (vmic, vmac, alpha, the VALD list, the
    ``stellar_model.pysme`` / ``stellar_model.ispec`` settings) reuses the existing
    template and chunks: delete the template and its ``_fullband_chunks/<template
    stem>`` directory to rebuild it.
    """
    target = Path(out_path)
    if target.exists():
        return target
    backend = str(syn.backend).lower()
    if backend not in _BACKENDS:
        raise ConfigError(f"unknown stellar-model backend {syn.backend!r}; use 'ispec' | 'pysme'")
    if backend == "pysme" and syn.linelist is None:
        raise ConfigError("pySME templates need a VALD line list (stellar_linelist setting)")
    if backend == "ispec" and syn.ispec_dir is None:
        raise ConfigError("iSpec templates need the iSpec installation (ispec_dir setting)")
    lo0, hi0 = wave_range_nm
    chunks = [
        (lo, min(lo + int(chunk_nm), int(hi0))) for lo in range(int(lo0), int(hi0), int(chunk_nm))
    ]
    chunk_dir = target.parent.resolve() / "_fullband_chunks" / target.stem / _CHUNK_CACHE_VERSION
    chunk_dir.mkdir(parents=True, exist_ok=True)
    slices: dict[tuple[int, int], str | None] = (
        dict(_slice_vald(str(syn.linelist), chunks, chunk_dir / "slices"))
        if backend == "pysme"
        else dict.fromkeys(chunks)
    )
    todo = [
        (syn, lo, hi, slices[(lo, hi)], float(resolution), str(chunk_dir / f"chunk_{lo}_{hi}.dat"))
        for (lo, hi) in chunks
        if not (chunk_dir / f"chunk_{lo}_{hi}.dat").exists()
    ]
    log.info(
        "stellar template: %d chunks cached, %d to synthesise", len(chunks) - len(todo), len(todo)
    )
    if todo:
        with cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1
        ) as ex:
            for lo, hi, n, mn in ex.map(_synth_chunk, todo):
                log.info("chunk %s-%s nm done (%d points, min flux %.3f)", lo, hi, n, mn)
    wave, flux = _stitch(
        [np.loadtxt(chunk_dir / f"chunk_{lo}_{hi}.dat") for lo, hi in chunks],
        chunks,
        reach_kms=5.0 * _kernel_sigma_kms(synthesis_resolution(syn), float(resolution)),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    st = syn.star
    title = (
        "pySME-VALD (full line list, sliced+parallel) rest-frame stellar template  "
        if backend == "pysme"
        else f"{_backend_label(syn)} rest-frame stellar template (chunked, parallel)  "
    )
    np.savetxt(
        target,
        np.column_stack([wave, flux]),
        fmt="%.6f %.6f",
        header=(
            f"{title}Teff={st.teff} logg={st.logg} [M/H]={st.feh} vsini={st.vsini_kms} "
            f"R={float(resolution)}\ncolumns: wave_nm(air)  flux(norm)"
        ),
    )
    return target


def _kernel_sigma_kms(r_from: float, r_to: float) -> float:
    """Sigma [km/s] of the Gaussian that broadens *r_from* to *r_to*."""
    return float(C_KMS * np.sqrt(1.0 / r_to**2 - 1.0 / r_from**2) / _GAUSS_FWHM)


def _stitch(
    parts: list[Array], chunks: list[tuple[int, int]], *, reach_kms: float
) -> tuple[Array, Array]:
    """Join broadened chunks: each keeps its own range ``[lo, hi)``.

    The joins lie ``CHUNK_MARGIN_NM`` inside both neighbours, far from their array
    edges. The outer margins of the band are kept, less *reach_kms* (the broadening
    kernel's reach) at the two array ends.
    """
    waves, fluxes = [], []
    last = len(chunks) - 1
    for k, (part, (lo, hi)) in enumerate(zip(parts, chunks, strict=True)):
        w, f = part[:, 0], part[:, 1]
        start = lo if k > 0 else w[0] * (1.0 + reach_kms / C_KMS)
        if k < last:
            sel = (w >= start) & (w < hi)
        else:
            sel = (w >= start) & (w <= w[-1] * (1.0 - reach_kms / C_KMS))
        waves.append(w[sel])
        fluxes.append(f[sel])
    return np.concatenate(waves), np.concatenate(fluxes)


# ---------------------------------------------------------------------------
# per-exposure stellar model
# ---------------------------------------------------------------------------
def load_template(path: str | os.PathLike[str]) -> tuple[Array, Array]:
    """Two-column ``wave_nm flux`` rest-frame template."""
    d = np.loadtxt(path)
    return d[:, 0].astype(float), d[:, 1].astype(float)


def broaden_template(
    wave_nm: Any, flux: Any, *, from_resolution: float, to_resolution: float
) -> Array:
    """Broaden a spectrum from *from_resolution* to *to_resolution* (``to < from``).

    Gaussian kernel of FWHM ``c * sqrt(1/to^2 - 1/from^2)`` in velocity space
    (``from_resolution = inf`` for an unbroadened synthesis). The spectrum is
    interpolated with a cubic spline onto a uniform ``ln lambda`` grid five times
    finer than its median sampling (and at most a twentieth of the kernel FWHM),
    convolved there (FFT, edge padding) and interpolated back to *wave_nm*. Points
    closer than 1 % of the median step, as at the joins of stitched chunks, are
    averaged first, so the spline cannot ring between them.

    On sampled Gaussian lines the effective resolving power is within 0.03 % of
    *to_resolution* for steps up to 0.5 pm at 630 nm (0.15 % at 1 pm), with no line
    shift above 0.1 m/s.
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import fftconvolve

    w = np.asarray(wave_nm, float)
    f = np.asarray(flux, float)
    if not to_resolution < from_resolution:
        raise ConfigError(
            f"template resolution {from_resolution:g} must exceed the target resolution "
            f"{to_resolution:g} to be broadened to it"
        )
    if w.size < 2 or np.any(np.diff(w) <= 0):
        raise ConfigError("template wavelengths must be strictly increasing")
    ln_w = np.log(w)
    step = np.diff(ln_w)
    first = np.concatenate([[True], step >= 0.01 * np.median(step)])
    group = np.cumsum(first) - 1
    count = np.bincount(group)
    ln_m = np.bincount(group, ln_w) / count
    f_m = np.bincount(group, f) / count
    sigma = C_KMS * np.sqrt(1.0 / to_resolution**2 - 1.0 / from_resolution**2) / _GAUSS_FWHM
    dv = min(sigma * _GAUSS_FWHM / 20.0, float(np.median(np.diff(ln_m))) * C_KMS / 5.0)
    n = int(np.ceil((ln_m[-1] - ln_m[0]) * C_KMS / dv)) + 1
    grid = np.linspace(ln_m[0], ln_m[-1], n)
    step_v = float(grid[1] - grid[0]) * C_KMS
    half = max(int(np.ceil(5.0 * sigma / step_v)), 1)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step_v / sigma) ** 2)
    kernel /= kernel.sum()
    fine = CubicSpline(ln_m, f_m)(grid) if ln_m.size > 2 else np.interp(grid, ln_m, f_m)
    broad = fftconvolve(np.pad(fine, half, mode="edge"), kernel, mode="valid")
    return np.asarray(np.interp(ln_w, grid, broad), float)


def stellar_model_for_order(wave_nm: Any, rv_kms: float, template: tuple[Array, Array]) -> Array:
    """Rest template shifted by ``rv_kms`` and resampled to ``wave_nm`` (1.0 outside)."""
    tw, tf = template
    return np.asarray(
        np.interp(np.asarray(wave_nm, float), tw * (1.0 + rv_kms / C_KMS), tf, left=1.0, right=1.0),
        float,
    )
