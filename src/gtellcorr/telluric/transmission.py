"""Telluric transmission spectra.

:func:`transmittance` evaluates an order's optical-depth cache for given relative
columns. :func:`transmission` is
the user-facing forward model: a complete profile and a line of sight in, the
(optionally LSF-convolved) transmission on your wavelength grid out.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gtellcorr.errors import ConfigError, did_you_mean
from gtellcorr.telluric.geometry import Geometry
from gtellcorr.telluric.model import LBLRTM_PROFILE_FIELDS, TelluricModel
from gtellcorr.telluric.molecules import ENGINE_MOLECULES
from gtellcorr.telluric.order_cache import OrderCache, transmittance

__all__ = ["OrderCache", "reference_spectrum", "transmission", "transmittance"]

DEFAULT_RELCOL: Mapping[str, float] = {"h2o": 1.0, "o2": 1.0}


def transmission(
    model: TelluricModel,
    profile: Mapping[str, Any],
    wave_nm: ArrayLike,
    *,
    geometry: Geometry,
    relcol: Mapping[str, float] | None = None,
    resolution: float | None = None,
    lsf_kind: str = "gauss",
    lsf_shape: float = 2.0,
) -> NDArray[np.float64]:
    """Telluric transmission on *wave_nm* (air, topocentric, nm, increasing).

    Parameters
    ----------
    profile
        A full profile (``z_km``, ``T_K``, ``P_hPa``, ``H2O_ppmv``; the fixed molecules
        are added from the reference atmosphere) or an already complete LBLRTM profile.
    relcol
        Relative columns of the scaled molecules (default h2o = o2 = 1); every other
        molecule keeps its reference column.
    resolution
        Resolving power of the LSF; ``None`` returns the unconvolved transmission.
        The convolution grid is set by ``resolution``; a *fit* sets its grid from the
        R bounds, so its model agrees with this one to interpolation accuracy, not
        bit for bit.
    """
    from gtellcorr.fitting.lsf_grid import apply_lsf_fixed_grid

    wave = np.asarray(wave_nm, float)
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.diff(wave) > 0):
        raise ValueError("wave_nm must be a strictly increasing 1-D grid")
    columns = dict(DEFAULT_RELCOL if relcol is None else relcol)
    if not columns:
        raise ValueError("relcol must name at least one molecule")
    for name in columns:
        if name not in ENGINE_MOLECULES:
            raise ConfigError(
                f"relcol: {name!r} cannot be scaled; engine molecules are "
                f"{', '.join(ENGINE_MOLECULES)}.{did_you_mean(name, ENGINE_MOLECULES)}"
            )
    # LBLRTM ID order, as in a fit spec, so equal requests share one cache entry
    columns = {m: float(columns[m]) for m in ENGINE_MOLECULES if m in columns}
    complete = (
        dict(profile)
        if set(profile) == set(LBLRTM_PROFILE_FIELDS)
        else model.lblrtm_profile(profile)
    )
    cache = model.build_order_cache(
        complete,
        (float(wave[0]), float(wave[-1])),
        geometry=geometry,
        fit_molecules=tuple(columns),
    )
    grid, trans = transmittance(cache, columns)
    if resolution is not None:
        trans = apply_lsf_fixed_grid(grid, trans, float(resolution), kind=lsf_kind, shape=lsf_shape)
    return np.asarray(np.interp(wave, grid, trans), float)


# ---------------------------------------------------------------------------
# reference spectrum (strong-telluric mask)
# ---------------------------------------------------------------------------
_FORWARD_IDENTITY_SCHEMA = "gtellcorr_broadened_telluric_reference_v1"
_FORWARD_IDENTITY_DOMAIN = b"gtellcorr_broadened_telluric_reference_v1\x00"


def reference_spectrum(
    model: TelluricModel,
    profile: Mapping[str, Any],
    wave_start_nm: float,
    wave_end_nm: float,
    *,
    geometry: Geometry,
    resolution: float,
    abundances: Mapping[str, float],
    broaden: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Transmission of all twelve engine molecules on the cache grid, with its key.

    Used by the strong-telluric mask. ``abundances`` gives, per engine molecule, the
    h2o multiplier or the column in ``REF_COLUMN`` units. With *broaden*, the
    transmission is convolved with a Gaussian of resolving power *resolution*
    (:func:`~gtellcorr.fitting.lsf.apply_lsf`, R fixed).
    ``info["forward_identity_digest"]`` is the key of the result (cache key of the
    strong-mask chunks).
    """
    import hashlib
    import json

    from gtellcorr.fitting.lsf import apply_lsf

    complete = (
        dict(profile)
        if set(profile) == set(LBLRTM_PROFILE_FIELDS)
        else model.lblrtm_profile(profile)
    )
    missing = [m for m in ENGINE_MOLECULES if m not in abundances]
    if missing:
        raise ValueError(f"abundances missing for {missing}")
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a finite positive number")
    wave_start_nm, wave_end_nm = float(wave_start_nm), float(wave_end_nm)
    if not (0.0 < wave_start_nm < wave_end_nm and np.isfinite(wave_end_nm)):
        raise ValueError("wave_start_nm and wave_end_nm must be positive and increasing")
    order_cache = model.build_order_cache(
        complete, (wave_start_nm, wave_end_nm), geometry=geometry, fit_molecules=ENGINE_MOLECULES
    )
    scales = {}
    for molecule in ENGINE_MOLECULES:
        value = float(abundances[molecule])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"abundance for {molecule!r} must be finite and non-negative")
        scales[molecule] = (
            value if molecule == "h2o" else value / float(order_cache.ref_column[molecule])
        )
    wave, trans = transmittance(order_cache, scales)
    wave = np.asarray(wave, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    if broaden:
        trans = apply_lsf(wave, trans, resolution, kind="gauss", shape=2.0)
    if (
        wave.ndim != 1
        or trans.shape != wave.shape
        or wave.size < 2
        or not np.all(np.isfinite(wave))
        or not np.all(np.isfinite(trans))
        or np.any(wave <= 0.0)
        or np.any(np.diff(wave) <= 0.0)
    ):
        raise RuntimeError("the LBLRTM forward model returned an invalid air-wavelength spectrum")
    key = getattr(order_cache.tau_cache, "cache_key", None)
    if not isinstance(key, str) or len(key) != 64:
        raise RuntimeError("internal error: the optical-depth cache has no valid key")
    record = {
        "schema": _FORWARD_IDENTITY_SCHEMA,
        "tau_cache_key": key,
        "relative_column_scales": {m: scales[m] for m in ENGINE_MOLECULES},
        "broadening": {
            "applied": bool(broaden),
            "kind": "gauss" if broaden else "none",
            "shape": 2.0 if broaden else None,
            "resolution": resolution if broaden else None,
        },
        "output_grid": {
            "medium": "air",
            "unit": "nm",
            "point_count": int(wave.size),
            "float_dtype": "<f8",
            "sha256": hashlib.sha256(
                np.ascontiguousarray(wave, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
        },
    }
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(_FORWARD_IDENTITY_DOMAIN)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    info = {
        "forward_identity_schema": _FORWARD_IDENTITY_SCHEMA,
        "forward_identity_digest": digest.hexdigest(),
        "tau_cache_key": key,
    }
    return wave, trans, info
