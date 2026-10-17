"""Strong telluric line-core mask, from a transmission model only.

A full-band transmission (all twelve engine molecules at their reference columns,
H2O scaled by ``h2o_relcol``, Gaussian LSF at the exposure's resolving power) is
computed in chunks for the first exposure's profile, airmass and resolution; where
it falls below ``threshold`` the wavelengths become exclusion windows for every
exposure. The chunks and the product (transmission ``.npz`` and mask ``.txt``) are
cached under ``<run>/mask_spectral_line/data``. Each carries a key (SHA-256) of
everything it depends on (optical-depth cache, columns, LSF, grid, mask settings) and
SHA-256 sums of its arrays; a cached file is reused only if all of them match.

The telluric windows of the RV measurement (:func:`ensure_rv_telluric_windows`) come
from the same model, with the LBLRTM continua divided out and a threshold near 1.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.preprocessing.windows import (
    Window,
    boolean_mask_windows,
    pad_window,
    read_windows,
)
from gtellcorr.telluric.molecules import ENGINE_MOLECULES

if TYPE_CHECKING:
    from gtellcorr.data.observation.instrument import InstrumentSpec
    from gtellcorr.preprocessing.config import StrongTelluricMask
    from gtellcorr.telluric.geometry import Geometry
    from gtellcorr.telluric.model import TelluricModel

__all__ = [
    "LineTransmission",
    "StrongMaskResult",
    "component_molecules",
    "ensure_rv_telluric_windows",
    "ensure_strong_telluric_mask",
    "generate_strong_telluric_mask",
    "line_transmission",
    "line_transmission_windows",
    "run_line_transmission",
    "rv_telluric_windows",
    "strong_telluric_windows",
    "windows_of_line_transmission",
]

Array = NDArray[np.float64]

_REFERENCE_CACHE_SCHEMA = "gtellcorr_strong_reference_cache_v4"
_REFERENCE_IDENTITY_DOMAIN = b"gtellcorr_strong_reference_forward_v1\x00"
_REFERENCE_REQUEST_IDENTITY_DOMAIN = b"gtellcorr_strong_reference_request_v1\x00"
_MASK_IDENTITY_DOMAIN = b"gtellcorr_strong_mask_identity_v1\x00"
_MASK_IDENTITY_SCHEMA = "gtellcorr_strong_mask_identity_v2"


# ---------------------------------------------------------------------------
# keys and atomic writes
# ---------------------------------------------------------------------------
def _identity_digest(domain: bytes, record: Mapping[str, Any]) -> str:
    """SHA-256 of *domain* and the sorted-key JSON of *record* (length-prefixed)."""
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    digest = sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(array: Any, *, dtype: Any) -> str:
    """SHA-256 of an array's dtype, shape and C-ordered bytes after casting to *dtype*."""
    normalized = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest = sha256()
    encoded = normalized.dtype.str.encode("ascii")
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)
    digest.update(normalized.ndim.to_bytes(4, "big"))
    for extent in normalized.shape:
        digest.update(int(extent).to_bytes(8, "big"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _reference_hashes(wave: Any, trans: Any, names: Any, by_molecule: Any) -> dict[str, str]:
    """Array checksums stored with a transmission chunk."""
    return {
        "wave_nm": _array_sha256(wave, dtype="<f8"),
        "transmission": _array_sha256(trans, dtype="<f8"),
        "molecule_names": _array_sha256(names, dtype="<U16"),
        "transmission_by_molecule": _array_sha256(by_molecule, dtype="<f8"),
    }


def _mask_hashes(
    wave: Any, trans: Any, mask: Any, windows: Any, names: Any, by_molecule: Any
) -> dict[str, str]:
    """Array checksums stored with the mask product."""
    return {
        **_reference_hashes(wave, trans, names, by_molecule),
        "mask": _array_sha256(mask, dtype=np.uint8),
        "windows": _array_sha256(windows, dtype="<f8"),
    }


def _atomic_savez(path: str | os.PathLike[str], **arrays: Any) -> None:
    """Write a compressed ``.npz`` through a temporary file, so readers never see a
    partial file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp.npz", dir=out.parent)
    tmp: Path | None = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write a text file through a temporary file (see :func:`_atomic_savez`)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    tmp: Path | None = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# windows, names, chunks
# ---------------------------------------------------------------------------
def strong_telluric_windows(
    wave_nm: Any,
    transmission: Any,
    *,
    threshold: float = 0.05,
    pad_nm: float = 0.0,
    merge_gap_nm: float = 0.0,
    min_width_nm: float = 0.0,
) -> list[Window]:
    """Windows where the (air, topocentric) model transmission is below ``threshold``."""
    wave = np.asarray(wave_nm, float)
    trans = np.asarray(transmission, float)
    if wave.shape != trans.shape:
        raise ValueError("wave_nm and transmission must have the same shape")
    mask = np.isfinite(wave) & np.isfinite(trans) & (trans < float(threshold))
    return boolean_mask_windows(
        wave, mask, pad_nm=pad_nm, merge_gap_nm=merge_gap_nm, min_width_nm=min_width_nm
    )


def line_transmission_windows(
    wave_nm: Any,
    line_transmission: Any,
    *,
    threshold: float,
    airmass_ratio: float = 1.0,
    pad_kms: float = 0.0,
) -> list[Window]:
    """Windows where the line transmission, scaled to another airmass, is below
    ``threshold``, each padded by *pad_kms*.

    The line optical depth is taken as proportional to the airmass:
    ``T ** airmass_ratio`` (a broadened transmission, so only approximately; exact for
    the weak lines that set a threshold near 1).
    """
    wave = np.asarray(wave_nm, float)
    trans = np.asarray(line_transmission, float)
    if wave.shape != trans.shape:
        raise ValueError("wave_nm and line_transmission must have the same shape")
    if not float(airmass_ratio) > 0.0:
        raise ValueError(f"airmass_ratio must be positive, got {airmass_ratio!r}")
    scaled = np.clip(trans, 1e-300, None) ** float(airmass_ratio)
    mask = np.isfinite(wave) & np.isfinite(trans) & (scaled < float(threshold))
    windows = boolean_mask_windows(wave, mask)
    return [pad_window(lo, hi, pad_kms) for lo, hi in windows]


@dataclass(frozen=True, eq=False)
class LineTransmission:
    """Model line transmission of a run (air, topocentric wavelengths).

    LBLRTM continua divided out, every engine molecule at its reference column,
    Gaussian LSF at the run's resolving power, at the geometry's ``airmass``; one
    ``(wave_nm, transmission)`` pair per wavelength chunk.
    """

    chunks: tuple[tuple[Array, Array], ...]
    airmass: float

    def at(self, wave_nm: Any, airmass: float | None = None) -> Array:
        """Transmission at *wave_nm* (1 outside the chunks), scaled to *airmass* as
        :func:`line_transmission_windows` does (``T ** (airmass / self.airmass)``)."""
        wave = np.asarray(wave_nm, float)
        out = np.ones(wave.shape)
        for w, t in self.chunks:
            inside = (wave >= w[0]) & (wave <= w[-1])
            out[inside] = np.interp(wave[inside], w, t)
        if airmass is not None:
            out = np.clip(out, 1e-300, None) ** (float(airmass) / self.airmass)
        return out

    def sha256(self) -> str:
        """SHA-256 of the airmass and the arrays (records what a product was built with)."""
        digest = sha256(np.float64(self.airmass).tobytes())
        for w, t in self.chunks:
            digest.update(np.ascontiguousarray(w, np.float64).tobytes())
            digest.update(np.ascontiguousarray(t, np.float64).tobytes())
        return digest.hexdigest()


def line_transmission(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    *,
    wavelength_range_nm: tuple[float, float],
    resolution: float,
    chunk_nm: float = 50.0,
) -> LineTransmission:
    """The line transmission (LBLRTM continua divided out, every engine molecule at
    its reference column, Gaussian LSF of *resolution*) at the geometry's airmass.

    The optical-depth caches are those of the strong-telluric mask (same chunks,
    molecules and geometry), so no LBLRTM run is added when that mask exists.
    """
    from gtellcorr.fitting.lsf import apply_lsf
    from gtellcorr.telluric.order_cache import continuum_transmittance, transmittance

    chunks = []
    lo_nm, hi_nm = (float(w) for w in wavelength_range_nm)
    for lo, hi in _chunk_ranges(lo_nm, hi_nm, chunk_nm):
        cache = model.build_order_cache(
            dict(profile), (lo, hi), geometry=geometry, fit_molecules=ENGINE_MOLECULES
        )
        scales = dict.fromkeys(ENGINE_MOLECULES, 1.0)
        wave, total = transmittance(cache, scales)
        continuum = continuum_transmittance(cache, scales)
        line = total if continuum is None else total / np.clip(continuum, 1e-300, None)
        line = apply_lsf(wave, line, float(resolution), kind="gauss", shape=2.0)
        keep = np.isfinite(wave) & np.isfinite(line) & (wave >= lo) & (wave <= hi)
        chunks.append((np.asarray(wave[keep], float), np.asarray(line[keep], float)))
    return LineTransmission(chunks=tuple(chunks), airmass=float(geometry.airmass))


def rv_telluric_windows(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    *,
    wavelength_range_nm: tuple[float, float],
    resolution: float,
    threshold: float,
    airmass_ratio: float = 1.0,
    pad_kms: float = 0.0,
    chunk_nm: float = 50.0,
) -> list[Window]:
    """Telluric windows of the RV measurement: where the :func:`line_transmission`
    at *airmass_ratio* times the geometry's airmass is below *threshold*
    (:func:`line_transmission_windows`)."""
    transmission = line_transmission(
        model, profile, geometry, wavelength_range_nm=wavelength_range_nm,
        resolution=resolution, chunk_nm=chunk_nm,
    )  # fmt: skip
    return windows_of_line_transmission(
        transmission, threshold=threshold, airmass_ratio=airmass_ratio, pad_kms=pad_kms
    )


def windows_of_line_transmission(
    transmission: LineTransmission, *, threshold: float, airmass_ratio: float, pad_kms: float
) -> list[Window]:
    """:func:`line_transmission_windows` of each chunk of *transmission*."""
    windows: list[Window] = []
    for wave, line in transmission.chunks:
        windows.extend(
            line_transmission_windows(
                wave, line, threshold=threshold, airmass_ratio=airmass_ratio, pad_kms=pad_kms
            )
        )
    return windows


def _token(value: float, fmt: str) -> str:
    """A number as a file-name token: trailing zeros dropped, ``-`` -> ``m``, ``.`` -> ``p``."""
    return format(value, fmt).rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def filename_token(
    *,
    wavelength_range_nm: tuple[float, float],
    resolution: float,
    backend: str,
    upper_source: str,
    time: datetime | None,
    airmass: float,
    threshold: float,
) -> str:
    """Readable summary of the mask's inputs in its file names: wavelength range, R,
    atmosphere sources and epoch, airmass and threshold (the keys decide reuse)."""
    lo, hi = map(float, wavelength_range_nm)
    dt_token = time.strftime("%Y%m%dT%H%M%S") if time is not None else "unknown_time"
    backend_token = str(backend).strip().replace(".", "_").replace("/", "_") or "unknown"
    upper_token = str(upper_source).strip().replace(".", "_").replace("/", "_") or "unknown"
    return (
        f"w{_token(lo, '.2f')}-{_token(hi, '.2f')}nm_"
        f"R{round(float(resolution))}_"
        f"{backend_token}-{upper_token}-{dt_token}_"
        f"am{_token(float(airmass), '.3f')}_"
        f"thr{_token(float(threshold), '.4f')}"
    )


def _chunk_ranges(lo_nm: float, hi_nm: float, chunk_nm: float) -> list[tuple[float, float]]:
    """Consecutive ``(lo, hi)`` ranges of at most *chunk_nm* covering *lo_nm*-*hi_nm*."""
    lo, hi, chunk = float(lo_nm), float(hi_nm), float(chunk_nm)
    if chunk <= 0:
        raise ValueError("chunk_nm must be positive")
    out = []
    start = lo
    while start < hi:
        end = min(start + chunk, hi)
        out.append((start, end))
        start = end
    return out


def component_molecules(instrument: InstrumentSpec) -> tuple[str, ...]:
    """Molecules whose individual transmissions are stored with the mask for
    diagnostics: the engine molecules among the instrument's anchor-order molecules."""
    engine = set(ENGINE_MOLECULES)
    return tuple(
        sorted({str(m).lower() for m in instrument.anchor_orders if str(m).lower() in engine})
    )


def _reference_abundances(
    ref_column: Mapping[str, float], *, relcol: float = 1.0, molecule: str | None = None
) -> dict[str, float]:
    """Abundances of the mask model: every molecule at its reference column with H2O
    scaled by *relcol*; or, with *molecule*, that molecule alone (the others at zero),
    scaled by *relcol* whatever the molecule (a diagnostic component only)."""
    if molecule is None:
        abundances = {m: float(ref_column[m]) for m in ENGINE_MOLECULES}
        abundances["h2o"] = float(relcol)
        return abundances
    molecule = str(molecule).lower()
    if molecule not in ENGINE_MOLECULES:
        raise ValueError(f"unsupported telluric component molecule {molecule!r}")
    abundances = dict.fromkeys(ENGINE_MOLECULES, 0.0)
    abundances[molecule] = (
        float(relcol) if molecule == "h2o" else float(ref_column[molecule]) * float(relcol)
    )
    return abundances


# ---------------------------------------------------------------------------
# reference transmission in chunks
# ---------------------------------------------------------------------------
def _request_identity(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    chunk: tuple[float, float],
    *,
    relcol: float,
    resolution: float,
    components: Sequence[str],
    ref_column: Mapping[str, float],
) -> str:
    """Key of a chunk request (optical-depth cache, columns, grid range, LSF);
    computed before the chunk and used as its cache file name."""
    tau_key, _ = model.order_cache_identity(
        profile,
        (float(chunk[0]), float(chunk[1])),
        geometry=geometry,
        fit_molecules=ENGINE_MOLECULES,
    )
    requests = []
    for molecule in (None, *components):
        values = _reference_abundances(ref_column, relcol=relcol, molecule=molecule)
        scales = {
            m: (float(values[m]) if m == "h2o" else float(values[m]) / float(ref_column[m]))
            for m in ENGINE_MOLECULES
        }
        requests.append(
            {
                "role": "total" if molecule is None else "component",
                "molecule": None if molecule is None else str(molecule),
                "relative_column_scales": scales,
            }
        )
    return _identity_digest(
        _REFERENCE_REQUEST_IDENTITY_DOMAIN,
        {
            "schema": _REFERENCE_CACHE_SCHEMA,
            "tau_cache_key": tau_key,
            "models": requests,
            "requested_output": {
                "wavelength_medium": "air",
                "wavelength_unit": "nm",
                "wave_start_nm": float(chunk[0]),
                "wave_end_nm": float(chunk[1]),
                "broadening": {"kind": "gauss", "shape": 2.0, "resolution": float(resolution)},
            },
            "component_interpolation": "numpy_linear_no_extrapolation_to_total_output_grid",
        },
    )


def _model_range(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    lo: float,
    hi: float,
    *,
    resolution: float,
    abundances: Mapping[str, float],
) -> tuple[Array, Array, dict[str, Any]]:
    """Broadened transmission of the given abundances over *lo*-*hi* nm (finite samples
    only) and the information its key is made from."""
    from gtellcorr.telluric.transmission import reference_spectrum

    wave, trans, info = reference_spectrum(
        model, profile, lo, hi, geometry=geometry, resolution=resolution, abundances=abundances
    )
    keep = np.isfinite(wave) & np.isfinite(trans) & (wave >= float(lo)) & (wave <= float(hi))
    return wave[keep], trans[keep], info


def _compute_chunk(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    chunk: tuple[float, float],
    *,
    relcol: float,
    resolution: float,
    components: Sequence[str],
    ref_column: Mapping[str, float],
) -> tuple[Array, Array, Array, str]:
    """Total and per-molecule transmission of one chunk on the total's grid, and the
    key of the result."""
    lo, hi = chunk
    wave, total, info = _model_range(
        model, profile, geometry, lo, hi, resolution=resolution,
        abundances=_reference_abundances(ref_column, relcol=relcol),
    )  # fmt: skip
    rows = []
    identities: list[dict[str, Any]] = [
        {"role": "total", "forward_identity_digest": info["forward_identity_digest"]}
    ]
    for molecule in components:
        w_m, t_m, m_info = _model_range(
            model, profile, geometry, lo, hi, resolution=resolution,
            abundances=_reference_abundances(ref_column, relcol=relcol, molecule=molecule),
        )  # fmt: skip
        rows.append(np.interp(wave, w_m, t_m, left=np.nan, right=np.nan))
        identities.append(
            {
                "role": "component",
                "molecule": str(molecule),
                "forward_identity_digest": m_info["forward_identity_digest"],
            }
        )
    by_molecule = np.vstack(rows) if rows else np.empty((0, wave.size), float)
    identity = _identity_digest(
        _REFERENCE_IDENTITY_DOMAIN,
        {
            "schema": _REFERENCE_CACHE_SCHEMA,
            "models": identities,
            "output_grid": _grid_record(wave),
            "component_interpolation": "numpy_linear_no_extrapolation_to_total_output_grid",
        },
    )
    return wave, total, by_molecule, identity


def _grid_record(wave: Array) -> dict[str, Any]:
    """Record of a wavelength grid for keys (size and checksum)."""
    return {
        "medium": "air",
        "unit": "nm",
        "point_count": int(wave.size),
        "float_dtype": "<f8",
        "sha256": sha256(np.ascontiguousarray(wave, dtype="<f8").tobytes(order="C")).hexdigest(),
    }


def _load_chunk(
    path: str, request_identity: str, components: tuple[str, ...]
) -> tuple[Array, Array, Array, str] | None:
    """A cached chunk, or ``None`` if it is unreadable, was built for another request or
    fails its checks (schema, checksums, shapes, finite increasing grid)."""
    try:
        with np.load(path, allow_pickle=False) as data:
            schema = str(np.asarray(data["cache_schema"]).item())
            identity = str(np.asarray(data["forward_identity_digest"]).item())
            request = str(np.asarray(data["request_identity_digest"]).item())
            names = tuple(str(v) for v in data["molecule_names"])
            w = np.asarray(data["wave_nm"], float)
            t = np.asarray(data["transmission"], float)
            by = np.asarray(data["transmission_by_molecule"], float)
            recorded = json.loads(str(np.asarray(data["array_sha256_json"]).item()))
            actual = _reference_hashes(w, t, np.asarray(data["molecule_names"]), by)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    ok = (
        schema == _REFERENCE_CACHE_SCHEMA
        and request == request_identity
        and len(identity) == 64
        and all(c in "0123456789abcdef" for c in identity)
        and names == components
        and recorded == actual
        and w.ndim == 1
        and t.shape == w.shape
        and by.shape == (len(components), w.size)
        and bool(np.all(np.isfinite(w)))
        and bool(np.all(np.diff(w) > 0.0))
        and bool(np.all(np.isfinite(t)))
        and bool(np.all(np.isfinite(by)))
    )
    return (w, t, by, identity) if ok else None


def reference_transmission_bundle(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    *,
    wavelength_range_nm: tuple[float, float],
    resolution: float,
    components: Sequence[str],
    relcol: float = 1.0,
    chunk_nm: float = 30.0,
    cache_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> tuple[Array, Array, tuple[str, ...], Array, str]:
    """Total and per-molecule transmission over *wavelength_range_nm*, joined from
    *chunk_nm* chunks, plus the key of the whole.

    Returns ``(wave_nm, total, component names, per-component transmissions,
    key)``. With *cache_dir*, each chunk is stored there and reused unless
    *overwrite*. Joined chunks keep only the points beyond the previous chunk's end.
    """
    from gtellcorr.telluric.lblrtm.tau_cache import REF_COLUMN

    names = tuple(str(m).lower() for m in components)
    complete = model.lblrtm_profile(profile) if "CO2_ppmv" not in profile else dict(profile)
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a finite positive number")
    lo_nm, hi_nm = map(float, wavelength_range_nm)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    waves: list[Array] = []
    totals: list[Array] = []
    parts: list[Array] = []
    chunk_ids = []
    for chunk in _chunk_ranges(lo_nm, hi_nm, chunk_nm):
        request = _request_identity(
            model, complete, geometry, chunk, relcol=relcol, resolution=resolution,
            components=names, ref_column=REF_COLUMN,
        )  # fmt: skip
        path = (
            os.path.join(str(cache_dir), f"chunk_{chunk[0]:.3f}_{chunk[1]:.3f}_{request}.npz")
            if cache_dir
            else None
        )
        cached = (
            _load_chunk(path, request, names)
            if path and os.path.exists(path) and not overwrite
            else None
        )
        if cached is None:
            w, t, by, identity = _compute_chunk(
                model, complete, geometry, chunk, relcol=relcol, resolution=resolution,
                components=names, ref_column=REF_COLUMN,
            )  # fmt: skip
            if path:
                stored = np.asarray(names, dtype="U16")
                hashes = _reference_hashes(w, t, stored, by)
                _atomic_savez(
                    path,
                    cache_schema=np.asarray(_REFERENCE_CACHE_SCHEMA),
                    request_identity_digest=np.asarray(request),
                    forward_identity_digest=np.asarray(identity),
                    array_sha256_json=np.asarray(
                        json.dumps(hashes, sort_keys=True, separators=(",", ":"))
                    ),
                    wave_nm=w,
                    transmission=t,
                    molecule_names=stored,
                    transmission_by_molecule=by,
                )
        else:
            w, t, by, identity = cached
        chunk_ids.append(
            {
                "wave_start_nm": float(chunk[0]),
                "wave_end_nm": float(chunk[1]),
                "request_identity_digest": request,
                "forward_identity_digest": identity,
            }
        )
        keep = np.ones(w.shape, bool)
        if waves:
            keep &= w > waves[-1][-1]
        if np.any(keep):
            waves.append(w[keep])
            totals.append(t[keep])
            parts.append(by[:, keep])
    if not waves:
        raise RuntimeError("reference telluric model produced no finite samples")
    wave = np.concatenate(waves)
    total = np.concatenate(totals)
    by_molecule = np.concatenate(parts, axis=1) if parts else np.empty((0, wave.size), float)
    bundle = _identity_digest(
        _REFERENCE_IDENTITY_DOMAIN,
        {"schema": _REFERENCE_CACHE_SCHEMA, "chunks": chunk_ids, "output_grid": _grid_record(wave)},
    )
    return wave, total, names, by_molecule, bundle


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------
def _product_paths(out_dir: Path, stem: str, token: str | None) -> dict[str, Path]:
    """Paths of the transmission (``.npz``) and mask (``.txt``) products."""
    suffix = f"_{token}" if token else ""
    return {
        "spectrum": out_dir / f"{stem}_strong_telluric_transmission{suffix}.npz",
        "mask": out_dir / f"{stem}_strong_telluric_mask{suffix}.txt",
    }


def _write_product(
    paths: Mapping[str, Path],
    *,
    wave: Array,
    trans: Array,
    windows: list[Window],
    names: tuple[str, ...],
    by_molecule: Array,
    metadata: Mapping[str, Any],
) -> None:
    """Write the mask (``lo_nm hi_nm`` lines under a commented header of *metadata*) and
    the transmission ``.npz`` with the pixel mask, keys and checksums."""
    mask = np.zeros(wave.shape, bool)
    for lo, hi in windows:
        mask |= (wave >= lo) & (wave <= hi)
    stored = np.asarray(names, dtype="U16")
    windows_array = (
        np.asarray(windows, float).reshape(-1, 2) if windows else np.empty((0, 2), float)
    )
    hashes = _mask_hashes(wave, trans, mask, windows_array, stored, by_molecule)
    lines = ["# strong telluric line-core mask; columns: lo_nm hi_nm\n"]
    for key in sorted(metadata):
        lines.append(f"# {key}: {metadata[key]}\n")
    for lo, hi in windows_array:
        lines.append(f"{lo:.8f} {hi:.8f}\n")
    text = "".join(lines)
    _atomic_savez(
        paths["spectrum"],
        cache_schema=np.asarray(str(metadata.get("cache_schema", ""))),
        forward_identity_digest=np.asarray(str(metadata.get("forward_identity_digest", ""))),
        mask_identity_digest=np.asarray(str(metadata.get("mask_identity_digest", ""))),
        array_sha256_json=np.asarray(json.dumps(hashes, sort_keys=True, separators=(",", ":"))),
        mask_text_sha256=np.asarray(sha256(text.encode("utf-8")).hexdigest()),
        wave_nm=wave,
        transmission=np.asarray(trans, float),
        mask=mask,
        windows=windows_array,
        molecule_names=stored,
        transmission_by_molecule=by_molecule,
        metadata=str(dict(metadata)),
    )
    _atomic_write_text(paths["mask"], text)


def _product_matches(paths: Mapping[str, Path], forward: str, mask_identity: str) -> bool:
    """Whether the stored product has these keys and passes its checks (array and
    mask-text checksums, shapes, finite increasing grid)."""
    try:
        with np.load(paths["spectrum"], allow_pickle=False) as p:
            wave = np.asarray(p["wave_nm"], np.float64)
            trans = np.asarray(p["transmission"], np.float64)
            mask = np.asarray(p["mask"], bool)
            windows = np.asarray(p["windows"], np.float64)
            names = np.asarray(p["molecule_names"])
            parts = np.asarray(p["transmission_by_molecule"], np.float64)
            recorded = json.loads(str(np.asarray(p["array_sha256_json"]).item()))
            recorded_text = str(np.asarray(p["mask_text_sha256"]).item())
            schema = str(np.asarray(p["cache_schema"]).item())
            fwd = str(np.asarray(p["forward_identity_digest"]).item())
            mid = str(np.asarray(p["mask_identity_digest"]).item())
        actual_text = sha256(Path(paths["mask"]).read_bytes()).hexdigest()
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return False
    return (
        schema == _REFERENCE_CACHE_SCHEMA
        and fwd == forward
        and mid == mask_identity
        and recorded == _mask_hashes(wave, trans, mask, windows, names, parts)
        and recorded_text == actual_text
        and wave.ndim == 1
        and trans.shape == wave.shape
        and mask.shape == wave.shape
        and windows.ndim == 2
        and windows.shape[1:] == (2,)
        and parts.shape == (names.size, wave.size)
        and bool(np.all(np.isfinite(wave)))
        and bool(np.all(np.diff(wave) > 0.0))
        and bool(np.all(np.isfinite(trans)))
        and bool(np.all(np.isfinite(windows)))
        and bool(np.all(np.isfinite(parts)))
    )


@dataclass(frozen=True)
class StrongMaskResult:
    """The run's strong-telluric mask and where it is stored."""

    windows: list[Window]
    """Exclusion windows (air nm, topocentric)."""
    paths: dict[str, Path]
    """``spectrum`` (transmission ``.npz``) and ``mask`` (``.txt``)."""
    status: str
    """``exists`` (reused) or ``generated``."""
    forward_identity_digest: str
    """Key of the transmission model."""
    mask_identity_digest: str
    """Key of the mask (model key and mask settings)."""
    cache_token: str
    """File-name token of the product."""


def generate_strong_telluric_mask(
    model: TelluricModel,
    profile: Mapping[str, Any],
    geometry: Geometry,
    *,
    instrument: InstrumentSpec,
    config: StrongTelluricMask,
    out_dir: Path,
    stem: str,
    resolution: float,
    time: datetime | None,
    lower_backend: str,
    upper_source: str,
    overwrite: bool = False,
) -> StrongMaskResult:
    """Build (or reuse) the run's strong-telluric mask from one transmission model over
    the instrument's wavelength range; *profile*, *geometry*, *resolution* and *time*
    are those of the reference exposure."""
    wr = instrument.spectral.wavelength_range_nm
    if wr is None:
        raise ValueError(f"{instrument.name}: spectral.wavelength_range_nm is not configured")
    wavelength_range = (min(map(float, wr)), max(map(float, wr)))
    base_token = filename_token(
        wavelength_range_nm=wavelength_range,
        resolution=resolution,
        backend=lower_backend,
        upper_source=upper_source,
        time=time,
        airmass=geometry.airmass,
        threshold=config.threshold,
    )
    cache_dir = out_dir / f"{stem}_strong_telluric_chunks_{base_token}_forward-v3"
    wave, trans, names, parts, forward = reference_transmission_bundle(
        model,
        profile,
        geometry,
        wavelength_range_nm=wavelength_range,
        resolution=resolution,
        components=component_molecules(instrument),
        relcol=config.h2o_relcol,
        chunk_nm=config.chunk_nm,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )
    mask_identity = _identity_digest(
        _MASK_IDENTITY_DOMAIN,
        {
            "schema": _MASK_IDENTITY_SCHEMA,
            "forward_identity_digest": forward,
            "threshold": float(config.threshold),
            "pad_nm": float(config.pad_nm),
            "merge_gap_nm": float(config.merge_gap_nm),
            "min_width_nm": float(config.min_width_nm),
        },
    )
    token = f"{base_token}_mask-{mask_identity}"
    paths = _product_paths(out_dir, stem, token)
    if (
        not overwrite
        and paths["spectrum"].exists()
        and paths["mask"].exists()
        and _product_matches(paths, forward, mask_identity)
    ):
        return StrongMaskResult(
            windows=read_windows(paths["mask"]),
            paths=paths,
            status="exists",
            forward_identity_digest=forward,
            mask_identity_digest=mask_identity,
            cache_token=token,
        )
    windows = strong_telluric_windows(
        wave, trans, threshold=config.threshold, pad_nm=config.pad_nm,
        merge_gap_nm=config.merge_gap_nm, min_width_nm=config.min_width_nm,
    )  # fmt: skip
    _write_product(
        paths,
        wave=wave,
        trans=trans,
        windows=windows,
        names=names,
        by_molecule=parts,
        metadata={
            "cache_schema": _REFERENCE_CACHE_SCHEMA,
            "forward_identity_digest": forward,
            "mask_identity_digest": mask_identity,
            "instrument": instrument.name.split("/", 1)[-1],
            "cache_token": token,
            "lower_backend": lower_backend,
            "upper_source": upper_source,
            "profile_time": time.isoformat() if time is not None else None,
            "wavelength_range_nm": wavelength_range,
            "resolution": float(resolution),
            "airmass": float(geometry.airmass),
            "threshold": float(config.threshold),
            # recorded as "relcol", as in earlier masks, so cached masks stay valid
            "relcol": float(config.h2o_relcol),
            "chunk_nm": float(config.chunk_nm),
            "pad_nm": float(config.pad_nm),
            "merge_gap_nm": float(config.merge_gap_nm),
            "min_width_nm": float(config.min_width_nm),
            "component_molecules": ",".join(names),
            "source": "generate_telluric_spectrum",
        },
    )
    return StrongMaskResult(
        windows=read_windows(paths["mask"]),
        paths=paths,
        status="generated",
        forward_identity_digest=forward,
        mask_identity_digest=mask_identity,
        cache_token=token,
    )


def run_line_transmission(
    model: TelluricModel,
    exposures: Sequence[Any],
    *,
    instrument: InstrumentSpec,
    fit: Any,
    settings: Any,
    celestrak: Any = None,
) -> LineTransmission:
    """The run's :func:`line_transmission`: profile and airmass of ``exposures[0]``, the
    start R of the fit (:func:`~gtellcorr.data.observation.instrument.start_resolution`),
    over the instrument's wavelength range, in the strong-telluric mask's chunks."""
    from gtellcorr.data.atmosphere import full_profile_for_exposure
    from gtellcorr.data.observation.instrument import start_resolution
    from gtellcorr.telluric.geometry import Geometry

    if not exposures:
        raise ValueError("no exposures available for the run's telluric model")
    wr = instrument.spectral.wavelength_range_nm
    if wr is None:
        raise ValueError(f"{instrument.name}: spectral.wavelength_range_nm is not configured")
    if model.config.continuum_mode == "scaled":
        model = model.with_continuum_mode("full")  # the strong mask's caches
    first = exposures[0]
    full = full_profile_for_exposure(
        first, instrument, config=fit.atmosphere, settings=settings, celestrak=celestrak
    )
    profile = model.lblrtm_profile({k: full[k] for k in ("z_km", "T_K", "P_hPa", "H2O_ppmv")})
    site = first.site
    geometry = Geometry(
        observatory=(site.lat_deg, site.lon_deg, site.alt_m / 1000.0),
        airmass=float(first.airmass or 1.5),
    )
    return line_transmission(
        model,
        profile,
        geometry,
        wavelength_range_nm=(min(map(float, wr)), max(map(float, wr))),
        resolution=start_resolution(exposures, instrument, fit.lsf.resolution),
        chunk_nm=float(fit.masks.strong_telluric.chunk_nm),
    )


def ensure_rv_telluric_windows(
    model: TelluricModel,
    exposures: Sequence[Any],
    *,
    instrument: InstrumentSpec,
    fit: Any,
    settings: Any,
    celestrak: Any = None,
    transmission: LineTransmission | None = None,
) -> list[Window]:
    """Telluric windows of the run's RV measurement (``rv.telluric_mask = "model"``).

    The model is that of the strong-telluric mask (:func:`run_line_transmission`, or
    *transmission* if given), with the line optical depth scaled to the largest
    airmass of *exposures* (:func:`line_transmission_windows`).
    """
    if transmission is None:
        transmission = run_line_transmission(
            model, exposures, instrument=instrument, fit=fit, settings=settings,
            celestrak=celestrak,
        )  # fmt: skip
    largest = max(float(e.airmass or 1.5) for e in exposures)
    rv = fit.rv
    return windows_of_line_transmission(
        transmission,
        threshold=float(rv.telluric_threshold),
        airmass_ratio=largest / transmission.airmass,
        pad_kms=float(rv.telluric_pad_kms),
    )


def ensure_strong_telluric_mask(
    model: TelluricModel,
    exposures: Sequence[Any],
    *,
    instrument: InstrumentSpec,
    fit: Any,
    settings: Any,
    out_dir: Path,
    stem: str,
    celestrak: Any = None,
    overwrite: bool = False,
) -> StrongMaskResult | None:
    """The run's strong-telluric mask from its first exposure (``None`` if disabled).

    ``fit`` is the :class:`~gtellcorr.config.FitConfig`; the atmospheric profile and
    airmass of ``exposures[0]`` and the start R of the fit
    (:func:`~gtellcorr.data.observation.instrument.start_resolution`) define the
    transmission model.
    """
    from gtellcorr.data.atmosphere import full_profile_for_exposure
    from gtellcorr.data.observation.instrument import start_resolution
    from gtellcorr.telluric.geometry import Geometry

    config = fit.masks.strong_telluric
    if not config.enabled:
        return None
    if not exposures:
        raise ValueError("no exposures available for the strong telluric mask")
    if model.config.continuum_mode == "scaled":
        # The mask uses the transmission at the reference columns, where a scaled
        # cache equals a full one; with all twelve molecules fitted a full cache needs
        # 15 LBLRTM runs instead of 63, and it is shared with full-mode runs.
        model = model.with_continuum_mode("full")
    first = exposures[0]
    full = full_profile_for_exposure(
        first, instrument, config=fit.atmosphere, settings=settings, celestrak=celestrak
    )
    profile = model.lblrtm_profile({k: full[k] for k in ("z_km", "T_K", "P_hPa", "H2O_ppmv")})
    site = first.site
    geometry = Geometry(
        observatory=(site.lat_deg, site.lon_deg, site.alt_m / 1000.0),
        airmass=float(first.airmass or 1.5),
    )
    resolution = start_resolution(exposures, instrument, fit.lsf.resolution)
    return generate_strong_telluric_mask(
        model,
        profile,
        geometry,
        instrument=instrument,
        config=config,
        out_dir=out_dir,
        stem=stem,
        resolution=resolution,
        time=first.time_utc,
        lower_backend=fit.atmosphere.lower_backend,
        upper_source=fit.atmosphere.upper_source,
        overwrite=overwrite,
    )
