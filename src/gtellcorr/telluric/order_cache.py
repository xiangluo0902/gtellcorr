"""Forward model for one order using a persistent optical-depth cache.

Surface state, the complete atmospheric profile, geometry, and wavelength window
are fixed when the cache is built.  Only molecular relative-column scales
remain free during optimization, so every fit-loop evaluation is pure NumPy.

The expensive LBLRTM resources are validated once per run
(:class:`~gtellcorr.telluric.model.TelluricModel`) and bound through a
run-scoped :class:`contextvars.ContextVar`.  This module
never resolves a resource path on its own, and cache construction fails closed
outside that explicit run scope.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gtellcorr.telluric.geometry import Geometry
from gtellcorr.telluric.lblrtm.lblrtm_errors import OpacityBasisCacheError
from gtellcorr.telluric.lblrtm.lblrtm_resources import ResolvedLblrtmResourceFiles
from gtellcorr.telluric.lblrtm.tau_cache import (
    _SUPPORTED_WAVELENGTH_FORMULAS,
    REF_COLUMN,
    TauCache,
)

_CACHE_PUBLICATION_WAIT_S = 30.0
_CACHE_PUBLICATION_POLL_S = 0.05


def _required(value, name):
    """*value*, which must be given: there are no hidden configuration defaults (the
    caller, :class:`~gtellcorr.telluric.model.TelluricModel`, passes every value)."""
    if value is None:
        raise ValueError(f"{name} must be given explicitly")
    return value


@dataclass(frozen=True)
class _LblrtmForwardContext:
    """Validated resources and cache location owned by one run."""

    resources: ResolvedLblrtmResourceFiles
    cache_directory: Path
    timeout_s: float
    wave_margin_nm: float
    lblrtm_verbose: bool
    reference_atmosphere: Mapping[str, object] | None
    vacuum_air_formula: str
    """Vacuum-to-air formula of the cache grid (that of the observations)."""


_LBLRTM_FORWARD_CONTEXT: ContextVar[_LblrtmForwardContext | None] = ContextVar(
    "gtellcorr_lblrtm_forward_context",
    default=None,
)


@contextmanager
def use_lblrtm_forward_context(
    *,
    resources,
    cache_directory,
    timeout_s=None,
    wave_margin_nm=None,
    lblrtm_verbose=None,
    reference_atmosphere=None,
    vacuum_air_formula=None,
):
    """Bind validated LBLRTM resources, the cache directory and the run settings for
    the duration of the ``with`` block."""

    if not isinstance(resources, ResolvedLblrtmResourceFiles):
        raise TypeError("resources must be a ResolvedLblrtmResourceFiles instance")
    resources.require_fully_validated()
    directory = Path(cache_directory).expanduser()
    if not directory.is_absolute():
        raise ValueError("cache_directory must be an absolute path")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"TauCache path is not a directory: {directory}")
    resolved_timeout_s = float(_required(timeout_s, "timeout_s"))
    if not np.isfinite(resolved_timeout_s) or resolved_timeout_s <= 0.0:
        raise ValueError("timeout_s must be a finite positive number")
    resolved_wave_margin_nm = float(_required(wave_margin_nm, "wave_margin_nm"))
    if not np.isfinite(resolved_wave_margin_nm) or resolved_wave_margin_nm < 0.0:
        raise ValueError("wave_margin_nm must be finite and non-negative")
    resolved_lblrtm_verbose = bool(_required(lblrtm_verbose, "lblrtm_verbose"))
    resolved_formula = str(_required(vacuum_air_formula, "vacuum_air_formula"))
    if resolved_formula not in _SUPPORTED_WAVELENGTH_FORMULAS:
        raise ValueError(
            f"vacuum_air_formula must be one of {sorted(_SUPPORTED_WAVELENGTH_FORMULAS)}, "
            f"got {resolved_formula!r}"
        )
    if reference_atmosphere is None:
        reference_snapshot = None
    else:
        if not isinstance(reference_atmosphere, Mapping):
            raise TypeError("reference_atmosphere must be a mapping or None")
        provenance = reference_atmosphere.get("reference_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("reference_atmosphere has no record of its source file")
        source_sha256 = provenance.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("reference_atmosphere has no valid source_sha256")
        reference_values = {}
        for field_name, value in reference_atmosphere.items():
            if isinstance(value, np.ndarray):
                captured = np.array(value, copy=True)
                captured.setflags(write=False)
                reference_values[field_name] = captured
            elif field_name == "reference_provenance":
                reference_values[field_name] = MappingProxyType(dict(value))
            else:
                reference_values[field_name] = value
        reference_snapshot = MappingProxyType(reference_values)

    token = _LBLRTM_FORWARD_CONTEXT.set(
        _LblrtmForwardContext(
            resources=resources,
            cache_directory=directory,
            timeout_s=resolved_timeout_s,
            wave_margin_nm=resolved_wave_margin_nm,
            lblrtm_verbose=resolved_lblrtm_verbose,
            reference_atmosphere=reference_snapshot,
            vacuum_air_formula=resolved_formula,
        )
    )
    try:
        yield
    finally:
        _LBLRTM_FORWARD_CONTEXT.reset(token)


def _require_lblrtm_forward_context() -> _LblrtmForwardContext:
    """The active run context (``RuntimeError`` outside one)."""
    context = _LBLRTM_FORWARD_CONTEXT.get()
    if context is None:
        raise RuntimeError(
            "build_order_cache must run inside TelluricModel.activate() (a checked LBLRTM bundle)"
        )
    context.resources.require_fully_validated()
    return context


def current_lblrtm_reference_atmosphere() -> Mapping[str, object] | None:
    """Return the checked RFM atmosphere of the active run, read-only.

    ``None`` means no LBLRTM run context is active, so a stand-alone
    atmospheric-profile caller must load the reference file itself.  An active
    context without a reference is a programming error and raises.
    """

    context = _LBLRTM_FORWARD_CONTEXT.get()
    if context is None:
        return None
    context.resources.require_fully_validated()
    if context.reference_atmosphere is None:
        raise RuntimeError(
            "internal error: the active LBLRTM setup has no MIPAS reference atmosphere"
        )
    return context.reference_atmosphere


def _load_tau_cache(
    path: Path,
    *,
    cache_key: str,
    cache_metadata,
    resources: ResolvedLblrtmResourceFiles,
) -> TauCache:
    """Load the cache with exactly this key; damage or a mismatch raises."""

    return TauCache.load(
        path,
        expected_cache_key=cache_key,
        expected_metadata=cache_metadata,
        resources=resources,
    )


def _load_published_cache_winner(
    path: Path,
    *,
    cache_key: str,
    cache_metadata,
    resources: ResolvedLblrtmResourceFiles,
) -> TauCache:
    """Wait briefly for an atomic concurrent publication, then load it."""

    deadline = time.monotonic() + _CACHE_PUBLICATION_WAIT_S
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "a concurrent TauCache publisher did not publish the "
                f"expected cache within {_CACHE_PUBLICATION_WAIT_S:.0f} s: "
                f"{path}"
            )
        time.sleep(_CACHE_PUBLICATION_POLL_S)
    return _load_tau_cache(
        path,
        cache_key=cache_key,
        cache_metadata=cache_metadata,
        resources=resources,
    )


def _build_or_load_tau_cache(
    constructor_arguments: dict,
    *,
    fit_molecules: tuple[str, ...],
    context: _LblrtmForwardContext,
) -> TauCache:
    """Load by the full key, or build the cache and write it atomically once."""

    cache_candidate = TauCache(**constructor_arguments)
    cache_key, cache_metadata = cache_candidate.expected_cache_identity(fit_molecules=fit_molecules)
    cache_path = context.cache_directory / f"{cache_key}.npz"

    if cache_path.exists():
        return _load_tau_cache(
            cache_path,
            cache_key=cache_key,
            cache_metadata=cache_metadata,
            resources=context.resources,
        )

    cache_candidate.build(fit_molecules=fit_molecules)
    if cache_candidate.cache_key != cache_key or cache_candidate.cache_metadata != cache_metadata:
        raise RuntimeError(
            "internal error: the built optical-depth cache has a different key than planned"
        )
    try:
        cache_candidate.save(cache_path)
    except FileExistsError:
        return _load_published_cache_winner(
            cache_path,
            cache_key=cache_key,
            cache_metadata=cache_metadata,
            resources=context.resources,
        )
    return cache_candidate


@dataclass
class OrderCache:
    """A built optical-depth cache for one order and wavelength window."""

    tau_cache: object
    wave_nm: np.ndarray
    fit_molecules: tuple
    ref_column: dict


def _prepare_order_cache_request(
    profile: dict,
    wave_range_nm: tuple[float, float],
    *,
    geometry: Geometry,
    fit_molecules: list[str],
    wave_margin_nm: float | None,
    lblrtm_verbose: bool | None,
):
    """Return the exact TauCache constructor state without running LBLRTM."""
    context = _require_lblrtm_forward_context()
    if wave_margin_nm is None:
        wave_margin_nm = context.wave_margin_nm
    if lblrtm_verbose is None:
        lblrtm_verbose = context.lblrtm_verbose

    margin_nm = float(wave_margin_nm)
    if not np.isfinite(margin_nm) or margin_nm < 0.0:
        raise ValueError("wave_margin_nm must be finite and non-negative")
    lo_nm = float(wave_range_nm[0]) - margin_nm
    hi_nm = float(wave_range_nm[1]) + margin_nm
    lowfreq = 1.0e7 / hi_nm
    highfreq = 1.0e7 / lo_nm

    lat, _lon, alt_km = geometry.observatory
    altitude = np.asarray(profile["z_km"], dtype=np.float64)
    surface_pressure = float(
        np.interp(
            alt_km,
            altitude,
            np.asarray(profile["P_hPa"], dtype=np.float64),
        )
    )
    surface_temperature = float(
        np.interp(
            alt_km,
            altitude,
            np.asarray(profile["T_K"], dtype=np.float64),
        )
    )
    normalized_fit_molecules = tuple(fit_molecules)
    effective_airmass = max(float(geometry.airmass), 1.001)
    constructor_arguments = {
        "lowfreq": lowfreq,
        "highfreq": highfreq,
        "pressure": surface_pressure,
        "temperature": surface_temperature,
        "airmass": effective_airmass,
        "lat": float(lat),
        "alt": float(alt_km),
        "vac2air": True,
        "atmosphere_profile": profile,
        "resources": context.resources,
        "wavelength_conversion_formula": context.vacuum_air_formula,
        "wave_margin_nm": 0.0,
        "continuum_mode": "none",
        "timeout_s": context.timeout_s,
        "verbose": bool(lblrtm_verbose),
    }
    return constructor_arguments, normalized_fit_molecules, context


def expected_order_cache_identity(
    profile: dict,
    wave_range_nm: tuple[float, float],
    *,
    geometry: Geometry,
    fit_molecules: list[str],
    wave_margin_nm: float | None = None,
    lblrtm_verbose: bool | None = None,
):
    """Return the key of the optical-depth cache without computing it.

    Higher cache layers use this to decide whether their already-broadened
    products are reusable before any LBLRTM calculation.  The key comes
    from the same ``TauCache.expected_cache_identity`` code
    used by :func:`build_order_cache`.
    """

    constructor_arguments, normalized_fit_molecules, _context = _prepare_order_cache_request(
        profile,
        wave_range_nm,
        geometry=geometry,
        fit_molecules=fit_molecules,
        wave_margin_nm=wave_margin_nm,
        lblrtm_verbose=lblrtm_verbose,
    )
    cache_candidate = TauCache(**constructor_arguments)
    return cache_candidate.expected_cache_identity(fit_molecules=normalized_fit_molecules)


def build_order_cache(
    profile: dict,
    wave_range_nm: tuple[float, float],
    *,
    geometry: Geometry,
    fit_molecules: list[str],
    wave_margin_nm: float | None = None,
    lblrtm_verbose: bool | None = None,
) -> OrderCache:
    """Build or load the per-molecule optical-depth cache for one order.

    The requested air-wavelength interval is padded by ``wave_margin_nm`` on each
    side and turned into wavenumbers (``1e7 / nm``); the
    :class:`~gtellcorr.telluric.lblrtm.tau_cache.TauCache` itself gets no further
    margin. The surface pressure and temperature are the profile's at the site
    altitude; the airmass is used as given, floored at 1.001. The cache grid is
    converted to air with the run's ``frames.vacuum_air_formula`` (the formula is
    part of the cache key); no continuum, no quality mask.
    """

    constructor_arguments, normalized_fit_molecules, context = _prepare_order_cache_request(
        profile,
        wave_range_nm,
        geometry=geometry,
        fit_molecules=fit_molecules,
        wave_margin_nm=wave_margin_nm,
        lblrtm_verbose=lblrtm_verbose,
    )

    try:
        tau_cache = _build_or_load_tau_cache(
            constructor_arguments,
            fit_molecules=normalized_fit_molecules,
            context=context,
        )
    except OpacityBasisCacheError:
        raise
    except Exception as exc:
        raise OpacityBasisCacheError(
            "failed to build or load the LBLRTM optical-depth cache of the order"
        ) from exc
    return OrderCache(
        tau_cache=tau_cache,
        wave_nm=np.asarray(tau_cache.wavegrid, dtype=np.float64),
        fit_molecules=normalized_fit_molecules,
        ref_column={molecule: float(REF_COLUMN[molecule]) for molecule in normalized_fit_molecules},
    )


def transmittance(cache: OrderCache, scales: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return transmission for relative-column multipliers on the cache grid."""

    columns = {
        molecule: float(scales[molecule]) * cache.ref_column[molecule] for molecule in scales
    }
    transmission = cache.tau_cache.transmittance(**columns)
    return cache.wave_nm, np.asarray(transmission, dtype=np.float64)
