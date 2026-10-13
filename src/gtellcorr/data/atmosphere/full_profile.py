"""Build per-exposure full atmospheric profiles (0--120 km) by merging a
lower-atmosphere profile (typically ERA5) with an upper-atmosphere
climatology (default NRLMSISE-00; MIPAS available as a fallback), cached
under ``<atmosphere_cache>/full_profile/`` (``Settings.atmosphere_cache``) for
the optical-depth caches and the fits.

Each full profile carries ``z_km``, ``T_K``, ``P_hPa``, ``H2O_ppmv`` on
gtellcorr's fixed 0--120 km grid, with a record of how it was made (lower/upper
source, reference atmosphere, blend zone, PWV).

Why a separate step?
--------------------
* Building the profile is network-bound (reanalysis download), runs once per
  exposure and always gives the same result for the same inputs.
* The optical-depth caches and the fits then read the cached ``.dat``: no
  network and no stitching at fit time.

The ``.dat`` is the record of "what atmosphere did this fit actually use".

The functions here read their parameters from the context set by the public
entry points; call them through :mod:`gtellcorr.data.atmosphere`::

    from gtellcorr.data import atmosphere
    full = atmosphere.get_full_profile(time_utc, site, config=..., settings=...)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import BinaryIO

import numpy as np

from gtellcorr.data.atmosphere._context import (
    BACKEND_SOURCE,
    atm,
    atmos_profile_basename,
    atmos_profile_dir,
    full_profile_dir,
)
from gtellcorr.data.atmosphere.retriever import (
    AtmosphereProfile,
    backend_cycle,
    check_site_coverage,
)
from gtellcorr.data.atmosphere.rfm_reference import (
    CANONICAL_ALTITUDE_GRID_KM,
    load_rfm_mipas_reference,
)
from gtellcorr.data.atmosphere.space_weather import (
    canonical_msis_space_weather_identity,
    configured_msis_space_weather_inputs,
)

logger = logging.getLogger(__name__)


def _report(verbose: bool, message: str) -> None:
    """Log a progress *message*: at INFO if *verbose*, else at DEBUG."""
    logger.log(logging.INFO if verbose else logging.DEBUG, message)


_FULL_PROFILE_CACHE_SCHEMA = "gtellcorr_full_profile_cache_v3"
_FULL_PROFILE_KEY_DOMAIN = b"gtellcorr/full-profile-request/v3\0"
_LOWER_PROFILE_CACHE_SCHEMA = "gtellcorr_lower_profile_cache_v2"
_LOWER_PROFILE_KEY_DOMAIN = b"gtellcorr/lower-profile-request/v2\0"
_FULL_PROFILE_RESULT_SCHEMA = "gtellcorr_full_profile_result_v1"
_FULL_PROFILE_RESULT_DOMAIN = b"gtellcorr/full-profile-result/v1\0"
_PYMSIS_VERSION_CONTRACT = "0.12.0"
_FULL_PROFILE_LOWER_ARRAY_FIELDS = (
    "height_km",
    "pressure_hpa",
    "temperature_K",
    "h2o_vmr_ppmv",
)


def _msis00_on_grid(
    z_km: np.ndarray,
    when_utc: datetime,
    lat_deg: float,
    lon_deg: float,
    *,
    space_weather_inputs: Mapping | None = None,
) -> dict:
    """NRLMSISE-00 T(z), P(z) on a given altitude grid.  H2O is not
    provided by MSIS-00; the caller pairs the returned (T, P) with a
    MIPAS H2O profile.

    Lazy import: pymsis is only loaded when upper_source='msis00'.
    """
    if space_weather_inputs is None:
        space_weather_inputs = configured_msis_space_weather_inputs(when_utc)
    space_weather_identity = canonical_msis_space_weather_identity(space_weather_inputs)
    import pymsis

    pymsis_version = getattr(pymsis, "__version__", None)
    if pymsis_version != _PYMSIS_VERSION_CONTRACT:
        raise RuntimeError(
            'NRLMSISE-00 (upper_source = "msis00") requires pymsis '
            f"{_PYMSIS_VERSION_CONTRACT}; found {pymsis_version!r}"
        )
    calculation_time_utc = (
        when_utc.replace(tzinfo=UTC) if when_utc.tzinfo is None else when_utc.astimezone(UTC)
    )
    # pymsis interprets datetime64 inputs as UTC.  Strip the already-normalized
    # timezone before NumPy conversion to avoid timezone-aware datetime
    # parsing, which NumPy has deprecated.
    calculation_time_for_pymsis = calculation_time_utc.replace(tzinfo=None)
    out = pymsis.calculate(
        dates=np.array([calculation_time_for_pymsis]),
        lons=np.array([lon_deg]),
        lats=np.array([lat_deg]),
        alts=np.asarray(z_km, float),
        f107s=np.array(
            [space_weather_identity["f107_previous_day"]],
            dtype=np.float64,
        ),
        f107as=np.array(
            [space_weather_identity["f107a_center81"]],
            dtype=np.float64,
        ),
        aps=np.array(
            [space_weather_identity["ap_storm_vector"]],
            dtype=np.float64,
        ),
        version="00",
        # Keep the numerics used so far.  The complete
        # seven-element vector is parsed and recorded, but daily-Ap mode uses
        # only aps[:, 0].  Switching to storm-time mode (-1) is a separate
        # scientific change and must not be hidden inside an offline-data fix.
        geomagnetic_activity=1,
    )
    arr = np.squeeze(out)  # (n_alts, 11)
    if arr.shape != (np.asarray(z_km).size, 11):
        raise ValueError(
            "pymsis returned an unexpected output shape: "
            f"{arr.shape}; expected ({np.asarray(z_km).size}, 11)"
        )
    T = np.asarray(
        arr[:, pymsis.Variable.TEMPERATURE],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("pymsis returned invalid temperature values")

    # MSIS explicitly returns number densities for the neutral constituents.
    # Summing them and applying p = n k_B T remains valid as composition
    # changes near the homopause; rho * R_dry * T would silently assume a
    # fixed dry-air molar mass and is biased at the top of this 120-km grid.
    density_variables = (
        pymsis.Variable.N2,
        pymsis.Variable.O2,
        pymsis.Variable.O,
        pymsis.Variable.HE,
        pymsis.Variable.H,
        pymsis.Variable.AR,
        pymsis.Variable.N,
        pymsis.Variable.ANOMALOUS_O,
        pymsis.Variable.NO,
    )
    number_densities = np.asarray(
        arr[:, [int(variable) for variable in density_variables]],
        dtype=np.float64,
    )
    if np.any(np.isinf(number_densities)) or np.any(
        number_densities[np.isfinite(number_densities)] < 0.0
    ):
        raise ValueError("pymsis returned invalid constituent number densities")
    # NRLMSISE-00 marks species outside their defined altitude range as NaN.
    # Such unavailable constituents contribute zero; at least one modelled
    # constituent must remain at every level.
    total_number_density_m3 = np.sum(
        np.where(np.isnan(number_densities), 0.0, number_densities),
        axis=1,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(total_number_density_m3)) or np.any(total_number_density_m3 <= 0.0):
        raise ValueError("pymsis total particle number density must be finite and positive")
    boltzmann_joule_per_kelvin = 1.380649e-23  # exact SI definition
    P = total_number_density_m3 * boltzmann_joule_per_kelvin * T / 100.0
    return dict(
        T_K=T,
        P_hPa=P,
        pressure_method="pymsis_total_number_density_times_kB_T",
        space_weather_inputs=space_weather_identity,
        geomagnetic_activity_mode="daily_ap_1",
        pymsis_package_version=pymsis_version,
    )


def extract_base_atmosphere(
    reference_source: (
        str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO | None
    ) = None,
    *,
    expected_sha256: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Return the project grid and, when supplied, an official RFM reference.

    With no ``reference_source`` this returns only the independently defined
    float64 0--120 km grid.  Building a complete profile requires a caller to
    load the reference explicitly by passing the official RFM
    mid-latitude-night ``ngt.atm`` file, its SHA-256 and source URL.
    """
    if reference_source is None:
        if expected_sha256 is not None or source_url is not None:
            raise ValueError("expected_sha256/source_url require reference_source")
        return {
            "z_km": np.array(
                CANONICAL_ALTITUDE_GRID_KM,
                dtype=np.float64,
                copy=True,
            )
        }
    if expected_sha256 is None or source_url is None:
        raise ValueError("explicit RFM reference requires expected_sha256 and source_url")
    return load_rfm_mipas_reference(
        reference_source,
        expected_sha256=expected_sha256,
        source_url=source_url,
    )


def _build_full_profile_resolved(
    lower: AtmosphereProfile,
    upper_source: str = "msis00",
    blend_zone_km: tuple | None = None,
    base: dict | None = None,
    *,
    space_weather_inputs: Mapping | None = None,
) -> tuple:
    """Merge a lower-atmosphere profile with an upper-atmosphere
    climatology onto the gtellcorr altitude grid (0--120 km) via
    :func:`edit_profile`.

    Parameters
    ----------
    lower : AtmosphereProfile
        Lower-atmosphere profile (e.g. ERA5 0.19--47.6 km).
    upper_source : {'msis00', 'mipas'}, default 'msis00'
        - 'msis00': use NRLMSISE-00 (Picone+2002) evaluated at the
          exposure date/site for T(z)/P(z); H2O uses MIPAS (MSIS does
          not give water).
        - 'mipas': use the explicit official RFM MIPAS reference for
          T/P/H2O.
    blend_zone_km : tuple of (z_start, z_end), optional
        Forwarded to :func:`edit_profile`; defaults to the top 6 km of
        the lower profile (molecfit's standard blend width).
    base : dict
        Reference returned by ``extract_base_atmosphere(explicit_source, ...)``.
        It supplies only the upper H2O/reference-molecule climatology; the
        output grid is always created by gtellcorr.

    Returns
    -------
    full : dict
        Keys ``z_km``, ``T_K``, ``P_hPa``, ``H2O_ppmv`` on the exact
        project-owned 0--120 km, 121-level float64 grid.
    metadata : dict
        Fields describing how the profile was made, for the .dat ``#``-header (see
        :func:`save_full_profile`).
    """
    if upper_source not in ("msis00", "mipas"):
        raise ValueError(f'upper_source must be "msis00" or "mipas", got {upper_source!r}')

    if base is None:
        base = extract_base_atmosphere(
            atm.RFM_MIPAS_REFERENCE_PATH,
            expected_sha256=atm.RFM_MIPAS_REFERENCE_SHA256,
            source_url=atm.RFM_MIPAS_REFERENCE_SOURCE_URL,
        )
    if not isinstance(base, Mapping):
        raise TypeError("base must be a reference-atmosphere mapping")
    try:
        reference_z = np.asarray(base["z_km"], dtype=np.float64)
        reference_h2o = np.asarray(base["H2O_ppmv"], dtype=np.float64)
        reference_provenance = base["reference_provenance"]
    except KeyError as exc:
        raise ValueError(
            "base must come from extract_base_atmosphere with an explicit "
            f"RFM reference; missing {exc.args[0]!r}"
        ) from exc
    if not isinstance(reference_provenance, Mapping):
        raise ValueError("the reference atmosphere's source record must be a table")
    reference_sha256 = reference_provenance.get("source_sha256")
    if not isinstance(reference_sha256, str) or len(reference_sha256) != 64:
        raise ValueError("the reference atmosphere has no valid source_sha256")
    if not np.array_equal(reference_z, CANONICAL_ALTITUDE_GRID_KM):
        raise ValueError("base reference grid does not match the 0--120 km grid")
    if (
        reference_h2o.shape != CANONICAL_ALTITUDE_GRID_KM.shape
        or not np.all(np.isfinite(reference_h2o))
        or np.any(reference_h2o < 0.0)
    ):
        raise ValueError("base H2O reference must be finite and non-negative")
    # Always allocate the grid from the project constant.  The third-party HGT
    # column above is only checked against it, never used as the grid.
    z_base = np.array(
        CANONICAL_ALTITUDE_GRID_KM,
        dtype=np.float64,
        copy=True,
    )

    if upper_source == "msis00":
        when = datetime.fromisoformat(lower.obs_datetime_utc.replace("Z", "+00:00"))
        if space_weather_inputs is None:
            space_weather_inputs = configured_msis_space_weather_inputs(when)
        msis = _msis00_on_grid(
            z_base,
            when,
            lat_deg=lower.obs_lat_deg,
            lon_deg=lower.obs_lon_deg,
            space_weather_inputs=space_weather_inputs,
        )
        upper_T = msis["T_K"]
        upper_P = msis["P_hPa"]
        upper_H2O = reference_h2o
        upper_label = (
            f"msis00 @ ({lower.obs_lat_deg:+.3f},{lower.obs_lon_deg:+.3f}) {when.isoformat()}"
        )
    else:  # mipas
        try:
            upper_T = np.asarray(base["T_K"], dtype=np.float64)
            upper_P = np.asarray(base["P_hPa"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "mipas upper source requires T_K and P_hPa in the explicit RFM reference"
            ) from exc
        upper_H2O = reference_h2o
        upper_label = "mipas mid-latitude night (Remedios v3 2001, official RFM source)"

    T = edit_profile(
        z_base, upper_T, lower.height_km, lower.temperature_K, blend_zone_km=blend_zone_km
    )
    P = edit_profile(
        z_base, upper_P, lower.height_km, lower.pressure_hpa, blend_zone_km=blend_zone_km
    )
    H2O = edit_profile(
        z_base, upper_H2O, lower.height_km, lower.h2o_vmr_ppmv, blend_zone_km=blend_zone_km
    )

    full = dict(z_km=z_base, T_K=T, P_hPa=P, H2O_ppmv=H2O)

    if blend_zone_km is None:
        z_top = float(lower.height_km.max())
        bz = (z_top - atm.BLEND_ZONE_WIDTH_KM, z_top)
    else:
        bz = (float(blend_zone_km[0]), float(blend_zone_km[1]))

    metadata = dict(
        lower_source=lower.backend,
        lower_fetched_utc=lower.fetched_utc,
        lower_obs_datetime_utc=lower.obs_datetime_utc,
        lower_z_km_range=[float(lower.height_km.min()), float(lower.height_km.max())],
        lower_n_levels=int(lower.height_km.size),
        upper_source=upper_label,
        obs_lat_deg=lower.obs_lat_deg,
        obs_lon_deg=lower.obs_lon_deg,
        obs_alt_km=lower.obs_alt_km,
        blend_zone_km=list(bz),
        blend_method="molecfit-style 6 km linear weighted",
        pwv_mm=lower.pwv_mm,
        n_levels=int(z_base.size),
        z_km_range=[float(z_base[0]), float(z_base[-1])],
        reference_atmosphere=dict(reference_provenance),
        upper_pressure_method=(
            msis.get("pressure_method")
            if upper_source == "msis00"
            else "official_rfm_reference_pressure"
        ),
        space_weather_inputs=(
            msis.get("space_weather_inputs") if upper_source == "msis00" else None
        ),
        geomagnetic_activity_mode=(
            msis.get("geomagnetic_activity_mode") if upper_source == "msis00" else None
        ),
        pymsis_package_version=(
            msis.get("pymsis_package_version") if upper_source == "msis00" else None
        ),
    )
    return full, metadata


def build_full_profile(
    lower: AtmosphereProfile,
    upper_source: str = "msis00",
    blend_zone_km: tuple | None = None,
    base: dict | None = None,
) -> tuple:
    """Build one complete profile while resolving MSIS drivers exactly once.

    The public signature remains stable.  ``get_full_profile`` can pass its
    already-resolved inputs to the private construction function so the cache key
    and the calculation use the same objects.
    """

    return _build_full_profile_resolved(
        lower,
        upper_source=upper_source,
        blend_zone_km=blend_zone_km,
        base=base,
        space_weather_inputs=None,
    )


def _full_profile_request_identity(
    lower: AtmosphereProfile,
    *,
    backend: str,
    interpolate: bool,
    merge_onsite: bool,
    surface_T_K,
    surface_RH_pct,
    surface_P_hpa,
    upper_source: str,
    blend_zone_km,
    reference_provenance: Mapping,
    space_weather_inputs: Mapping | None = None,
) -> dict:
    """Return the cache key of one stitched profile.

    The key covers the numerical lower atmosphere and every option or versioned
    construction rule that can select or transform it.  It is separate from the
    hash of the output profile: the key tells whether a cached result was built
    for the same *request*, including inputs that can change the MSIS upper
    atmosphere or the record of an otherwise equal numerical result.
    """
    lower_arrays = {}
    for field_name in _FULL_PROFILE_LOWER_ARRAY_FIELDS:
        values = np.asarray(getattr(lower, field_name), dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"lower profile {field_name} must be a finite one-dimensional "
                "array with at least two levels"
            )
        canonical_values = np.ascontiguousarray(values, dtype="<f8")
        lower_arrays[field_name] = {
            "dtype": "float64_le",
            "shape": [int(size) for size in canonical_values.shape],
            "sha256": sha256(canonical_values.tobytes(order="C")).hexdigest(),
        }

    lower_size = int(np.asarray(lower.height_km).size)
    if any(
        int(np.asarray(getattr(lower, field_name)).size) != lower_size
        for field_name in _FULL_PROFILE_LOWER_ARRAY_FIELDS
    ):
        raise ValueError("lower profile arrays must have the same number of levels")

    reference_sha256 = reference_provenance.get("source_sha256")
    if (
        not isinstance(reference_sha256, str)
        or len(reference_sha256) != 64
        or any(character not in "0123456789abcdef" for character in reference_sha256)
    ):
        raise ValueError("the reference atmosphere has no valid source_sha256")

    surface_inputs = {}
    for field_name, value in (
        ("temperature_K", surface_T_K),
        ("relative_humidity_pct", surface_RH_pct),
        ("pressure_hpa", surface_P_hpa),
    ):
        if value is None:
            surface_inputs[field_name] = None
            continue
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"surface {field_name} must be finite or None")
        surface_inputs[field_name] = numeric_value

    lower_top_km = float(np.max(np.asarray(lower.height_km, dtype=np.float64)))
    if blend_zone_km is None:
        default_width_km = float(atm.BLEND_ZONE_WIDTH_KM)
        resolved_blend_zone_km = [
            lower_top_km - default_width_km,
            lower_top_km,
        ]
        blend_zone_source = "configured_default_width"
    else:
        if len(blend_zone_km) != 2:
            raise ValueError("blend_zone_km must contain exactly two values")
        resolved_blend_zone_km = [
            float(blend_zone_km[0]),
            float(blend_zone_km[1]),
        ]
        if (
            not np.all(np.isfinite(resolved_blend_zone_km))
            or resolved_blend_zone_km[1] <= resolved_blend_zone_km[0]
        ):
            raise ValueError("blend_zone_km must contain finite values with end > start")
        default_width_km = None
        blend_zone_source = "explicit"

    canonical_grid = np.ascontiguousarray(
        CANONICAL_ALTITUDE_GRID_KM,
        dtype="<f8",
    )
    if upper_source == "msis00":
        if space_weather_inputs is None:
            raise ValueError("an msis00 upper atmosphere needs the space-weather inputs")
        canonical_space_weather = canonical_msis_space_weather_identity(space_weather_inputs)
    else:
        if space_weather_inputs is not None:
            raise ValueError("space-weather inputs are only valid for upper_source='msis00'")
        canonical_space_weather = None

    request = {
        "schema": _FULL_PROFILE_CACHE_SCHEMA,
        "lower_profile": {
            "backend_requested": str(backend),
            "backend_returned": str(lower.backend),
            "obs_datetime_utc": str(lower.obs_datetime_utc),
            "obs_lat_deg": float(lower.obs_lat_deg),
            "obs_lon_deg": float(lower.obs_lon_deg),
            "obs_alt_km": float(lower.obs_alt_km),
            "grid_datetime_utc": str(lower.grid_datetime_utc),
            "grid_lat_deg": float(lower.grid_lat_deg),
            "grid_lon_deg": float(lower.grid_lon_deg),
            "arrays": lower_arrays,
        },
        "lower_retrieval": {
            "horizontal_interpolation_requested": bool(interpolate),
            "horizontal_interpolation_policy": ("backend_bilinear_or_nearest_v1"),
        },
        "onsite_merge": {
            "requested": bool(merge_onsite),
            "applied": bool(lower.merged_with_asm),
            "inputs": surface_inputs,
            "mixing_height_km": float(atm.ASM_EMIX_KM),
            "force_below_observatory": True,
            "humidity_conversion_policy": ("murphy_koop_min_liquid_ice_rh_to_vmr_v1"),
            "policy": "multiplicative_shift_linear_taper_v1",
        },
        "upper_atmosphere": {
            "source": str(upper_source),
            "reference_schema": reference_provenance.get("schema"),
            "reference_sha256": reference_sha256,
            "temperature_pressure_policy": (
                "nrlmsise00_pymsis_version_00_total_number_density_kbt_v1"
                if upper_source == "msis00"
                else "official_rfm_reference_temperature_pressure_v1"
            ),
            "space_weather_policy": (
                "hash_pinned_celestrak_explicit_pymsis_inputs_v1"
                if upper_source == "msis00"
                else None
            ),
            "geomagnetic_activity_policy": (
                "pymsis_daily_ap_mode_1_only_aps0_consumed_v1" if upper_source == "msis00" else None
            ),
            "pymsis_package_version": (
                _PYMSIS_VERSION_CONTRACT if upper_source == "msis00" else None
            ),
            "msis_model_version": ("00" if upper_source == "msis00" else None),
            "space_weather_inputs": canonical_space_weather,
            "water_policy": "official_rfm_reference_h2o_v1",
        },
        "stitching": {
            "blend_zone_source": blend_zone_source,
            "default_blend_width_km": default_width_km,
            "resolved_blend_zone_km": resolved_blend_zone_km,
            "interpolation_policy": "numpy_linear_edge_clamp_v1",
            "blend_policy": "linear_weighted_transition_v1",
        },
        "output_grid": {
            "policy": "gtellcorr_canonical_0_120_km_1_km_v1",
            "dtype": "float64_le",
            "shape": [int(size) for size in canonical_grid.shape],
            "sha256": sha256(canonical_grid.tobytes(order="C")).hexdigest(),
        },
    }
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": _FULL_PROFILE_CACHE_SCHEMA,
        "digest": sha256(_FULL_PROFILE_KEY_DOMAIN + payload).hexdigest(),
        "request": request,
    }


def _request_keyed_cache_basename(
    requested_basename: str, *, identity_token: str, profile_kind: str
) -> str:
    """Insert a key token before the time suffix of the file name."""
    kind_suffix = f"_{profile_kind}"
    if requested_basename.endswith(kind_suffix):
        stem_without_kind = requested_basename[: -len(kind_suffix)]
        prefix, separator, timestamp = stem_without_kind.rpartition("_")
        if (
            separator
            and len(timestamp) == 15
            and timestamp[8] == "T"
            and timestamp[:8].isdigit()
            and timestamp[9:].isdigit()
        ):
            return f"{prefix}_{identity_token}_{timestamp}{kind_suffix}"
        return f"{stem_without_kind}_{identity_token}{kind_suffix}"
    return f"{requested_basename}_{identity_token}"


def _full_profile_cache_basename(requested_basename: str, request_digest: str) -> str:
    """Return a v3 cache basename while leaving older cache files untouched."""
    return _request_keyed_cache_basename(
        requested_basename,
        identity_token=f"request-v3-{request_digest}",
        profile_kind="full",
    )


def _lower_profile_cache_basename(
    *,
    source: str,
    site_label: str,
    backend: str,
    dt_utc: datetime,
    site_lat_deg: float,
    site_lon_deg: float,
    site_alt_km: float,
    interpolate: bool,
) -> str:
    """Return a request-keyed raw-cache basename for ``get_meteo``.

    The historical raw cache keyed only source/site-label/cycle, so two
    backends sharing one source token or a changed interpolation/site request
    could silently reuse the wrong lower atmosphere.  The v2 filename keeps
    those variants separate without deleting the historical artifact.
    """
    cycle_datetime = backend_cycle(backend, dt_utc)
    request = {
        "schema": _LOWER_PROFILE_CACHE_SCHEMA,
        "source": str(source),
        "backend": str(backend),
        "cycle_datetime_utc": cycle_datetime.astimezone(UTC).isoformat(),
        "site_label": str(site_label),
        "site_lat_deg": float(site_lat_deg),
        "site_lon_deg": float(site_lon_deg),
        "site_alt_km": float(site_alt_km),
        "horizontal_interpolation_requested": bool(interpolate),
        "retrieval_policy": "backend_cycle_site_interpolation_v1",
    }
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    request_digest = sha256(_LOWER_PROFILE_KEY_DOMAIN + payload).hexdigest()
    requested_basename = atmos_profile_basename(
        source,
        site_label,
        cycle_datetime,
        "raw",
    )
    return _request_keyed_cache_basename(
        requested_basename,
        identity_token=f"request-v2-{request_digest}",
        profile_kind="raw",
    )


def _full_profile_result_identity(full: Mapping) -> dict:
    """Hash the four persisted output arrays for cache-integrity checks."""
    arrays = {}
    framed_hash = sha256(_FULL_PROFILE_RESULT_DOMAIN)
    for field_name in _FULL_PROFILE_KEYS:
        if field_name not in full:
            raise ValueError(f"full profile missing required key {field_name!r}")
        values = np.asarray(full[field_name], dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"full profile {field_name} must be a finite one-dimensional array")
        canonical_values = np.ascontiguousarray(values, dtype="<f8")
        field_bytes = canonical_values.tobytes(order="C")
        field_digest = sha256(field_bytes).hexdigest()
        arrays[field_name] = {
            "dtype": "float64_le",
            "shape": [int(size) for size in canonical_values.shape],
            "sha256": field_digest,
        }
        encoded_name = field_name.encode("utf-8")
        framed_hash.update(len(encoded_name).to_bytes(4, "big"))
        framed_hash.update(encoded_name)
        framed_hash.update(len(field_bytes).to_bytes(8, "big"))
        framed_hash.update(field_bytes)
    return {
        "schema": _FULL_PROFILE_RESULT_SCHEMA,
        "digest": framed_hash.hexdigest(),
        "arrays": arrays,
    }


def get_full_profile(
    dt_utc,
    *,
    observatory=None,
    observatory_name=None,
    observatory_label=None,
    backend=None,
    interpolate=True,
    merge_onsite=False,
    surface_T_K=None,
    surface_RH_pct=None,
    surface_P_hpa=None,
    upper_source=None,
    blend_zone_km=None,
    base=None,
    save=True,
    reuse_cache=True,
    out_dir=None,
    basename=None,
    return_meta=False,
    verbose=False,
):
    """Full 0--120 km profile for one exposure, with request-aware caching.

    This is the ``get_meteo`` sibling for the *stitched* profile and the
    single place the "is it already built?" check lives for the API path:

    * The current lower profile is obtained through ``get_meteo``.  Its raw
      cache remains keyed by data cycle, so an existing raw profile needs no
      network access.
    * A cache key then covers the lower numerical arrays, upper source, RFM
      reference, interpolation/ASM settings, blend zone and construction rules.
    * A matching v3 cache skips upper-atmosphere generation and stitching.
      Earlier files have no complete v3 key and are left untouched.

    Parameters mirror ``get_meteo`` (site / backend / on-site merge) plus
    :func:`build_full_profile` (``upper_source`` / ``blend_zone_km`` /
    ``base``).  Load the RFM reference (checked against its SHA-256) once with
    :func:`extract_base_atmosphere` and reuse that ``base`` for many
    exposures.

    Returns ``full`` (dict ``z_km``/``T_K``/``P_hPa``/``H2O_ppmv``), or
    ``(full, metadata)`` when ``return_meta=True``.
    """
    # Lazy import keeps this module importable without pulling the meteo /
    # plotting facade (and avoids any import-order coupling).
    from gtellcorr.data.atmosphere.meteo import _resolve_site, get_meteo

    backend = backend or atm.LOWER_BACKEND_DEFAULT
    upper_source = upper_source or atm.UPPER_SOURCE_DEFAULT
    if not isinstance(dt_utc, datetime):
        raise TypeError("dt_utc must be a datetime")
    if upper_source not in ("msis00", "mipas"):
        raise ValueError(f'upper_source must be "msis00" or "mipas", got {upper_source!r}')
    if base is None:
        base = extract_base_atmosphere(
            atm.RFM_MIPAS_REFERENCE_PATH,
            expected_sha256=atm.RFM_MIPAS_REFERENCE_SHA256,
            source_url=atm.RFM_MIPAS_REFERENCE_SOURCE_URL,
        )
    if not isinstance(base, Mapping):
        raise TypeError("base must be a reference-atmosphere mapping")
    reference_provenance = base.get("reference_provenance")
    if not isinstance(reference_provenance, Mapping):
        raise ValueError(
            "get_full_profile needs the MIPAS reference atmosphere loaded by "
            "extract_base_atmosphere (with its SHA-256)"
        )
    expected_reference_sha256 = reference_provenance.get("source_sha256")
    if (
        not isinstance(expected_reference_sha256, str)
        or len(expected_reference_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_reference_sha256)
    ):
        raise ValueError("the reference atmosphere has no valid source_sha256")
    src = BACKEND_SOURCE.get(backend, backend)
    out_dir = out_dir or full_profile_dir()
    # Resolve the same site label get_meteo would, so the full filename's
    # site token matches the raw one (keeps the two caches aligned).
    site_lat_deg, site_lon_deg, site_alt_km, label = _resolve_site(
        observatory,
        observatory_name,
        observatory_label,
    )
    # Full profiles are per-exposure -> key by the exposure time (seconds).
    requested_stem = basename or atmos_profile_basename(src, label, dt_utc, "full")
    raw_cache_dir = atmos_profile_dir(src)
    raw_cache_stem = _lower_profile_cache_basename(
        source=src,
        site_label=label,
        backend=backend,
        dt_utc=dt_utc,
        site_lat_deg=site_lat_deg,
        site_lon_deg=site_lon_deg,
        site_alt_km=site_alt_km,
        interpolate=interpolate,
    )
    raw_cache_path = os.path.join(raw_cache_dir, raw_cache_stem + ".dat")
    raw_cache_existed = os.path.exists(raw_cache_path)
    lower = get_meteo(
        dt_utc=dt_utc,
        observatory=observatory,
        observatory_name=observatory_name,
        observatory_label=observatory_label,
        backend=backend,
        interpolate=interpolate,
        merge_onsite=merge_onsite,
        surface_T_K=surface_T_K,
        surface_RH_pct=surface_RH_pct,
        surface_P_hpa=surface_P_hpa,
        out_dir=raw_cache_dir,
        basename=raw_cache_stem,
        verbose=verbose,
    )
    # save_profile currently persists text.  On the first fetch, reload that
    # just-published representation before hashing/building so the fresh and
    # later cache-loaded paths consume exactly the same lower arrays.
    if not raw_cache_existed and os.path.exists(raw_cache_path):
        lower = get_meteo(
            dt_utc=dt_utc,
            observatory=observatory,
            observatory_name=observatory_name,
            observatory_label=observatory_label,
            backend=backend,
            interpolate=interpolate,
            merge_onsite=merge_onsite,
            surface_T_K=surface_T_K,
            surface_RH_pct=surface_RH_pct,
            surface_P_hpa=surface_P_hpa,
            out_dir=raw_cache_dir,
            basename=raw_cache_stem,
            verbose=False,
        )
    if lower.backend != backend:
        raise ValueError(
            "lower-profile cache backend mismatch: "
            f"requested {backend!r}, received {lower.backend!r}"
        )
    returned_site = np.asarray(
        [lower.obs_lat_deg, lower.obs_lon_deg, lower.obs_alt_km],
        dtype=np.float64,
    )
    requested_site = np.asarray(
        [site_lat_deg, site_lon_deg, site_alt_km],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(returned_site)) or not np.allclose(
        returned_site,
        requested_site,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "lower-profile cache site mismatch: "
            f"requested {requested_site.tolist()}, "
            f"received {returned_site.tolist()}"
        )
    request_datetime = dt_utc
    if request_datetime.tzinfo is None:
        request_datetime = request_datetime.replace(tzinfo=UTC)
    request_datetime = request_datetime.astimezone(UTC)
    lower = replace(
        lower,
        obs_lat_deg=float(site_lat_deg),
        obs_lon_deg=float(site_lon_deg),
        obs_alt_km=float(site_alt_km),
        obs_datetime_utc=request_datetime.isoformat(),
    )
    check_site_coverage(lower, backend)
    space_weather_inputs = (
        configured_msis_space_weather_inputs(request_datetime) if upper_source == "msis00" else None
    )
    cache_identity = _full_profile_request_identity(
        lower,
        backend=backend,
        interpolate=interpolate,
        merge_onsite=merge_onsite,
        surface_T_K=surface_T_K,
        surface_RH_pct=surface_RH_pct,
        surface_P_hpa=surface_P_hpa,
        upper_source=upper_source,
        blend_zone_km=blend_zone_km,
        reference_provenance=reference_provenance,
        space_weather_inputs=space_weather_inputs,
    )
    cache_stem = _full_profile_cache_basename(
        requested_stem,
        cache_identity["digest"],
    )
    if reuse_cache:
        hit = load_full_profile(out_dir, cache_stem)
        if hit is not None:
            full, meta = hit
            cached_result_identity = meta.get("result_identity")
            try:
                computed_result_identity = _full_profile_result_identity(full)
            except ValueError:
                computed_result_identity = None
            if (
                meta.get("cache_identity") == cache_identity
                and isinstance(cached_result_identity, Mapping)
                and cached_result_identity == computed_result_identity
            ):
                _report(verbose, "full profile: reused")
                return (full, meta) if return_meta else full
    full, meta = _build_full_profile_resolved(
        lower,
        upper_source=upper_source,
        blend_zone_km=blend_zone_km,
        base=base,
        space_weather_inputs=space_weather_inputs,
    )
    if upper_source == "msis00":
        _report(verbose, "NRLMSISE-00 upper profile: constructed")
    else:
        _report(verbose, "MIPAS upper profile: loaded")
    if upper_source == "msis00":
        expected_space_weather = cache_identity["request"]["upper_atmosphere"][
            "space_weather_inputs"
        ]
        if meta.get("space_weather_inputs") != expected_space_weather:
            raise RuntimeError(
                "internal error: the MSIS inputs differ from those of the cached full profile"
            )
    meta["cache_identity"] = cache_identity
    meta["result_identity"] = _full_profile_result_identity(full)
    meta["requested_basename"] = requested_stem
    meta["cache_basename"] = cache_stem
    if save:
        save_full_profile(full, out_dir, cache_stem, metadata=meta)
    _report(verbose, "full profile: constructed")
    return (full, meta) if return_meta else full


# ==================================================================
# Profile stitching (molecfit-style weighted blend)
# ==================================================================
def edit_profile(
    base_height_km,
    base_value,
    profile_height_km,
    profile_value,
    blend_zone_km=None,
):
    """Stitch a user-supplied lower-atmosphere profile onto a
    background / upper-atmosphere profile using molecfit-style
    linearly-weighted blending.

    This follows the molecfit profile-combination prescription instead of
    a 1-km exponential damping at the boundary between the user and MIPAS
    layers, which can produce ~7 K/km artificial kinks where the user profile
    ends (e.g. ERA5 at ~47 km).  The molecfit prescription (Smette+15,
    ``telluriccorr/mf_atm_combined_gdas_grid_fixed``) instead linearly
    tapers between the two sources over a configurable transition zone.

    Pure function: no I/O, network access, or forward-model state.  Both
    sources must be downloaded or computed up-front (e.g. ERA5 via
    ``fetch('era5_cdsapi', ...)``, NRLMSISE-00 via
    ``pymsis.calculate(...)``, MIPAS via the official RFM reference, checked
    against its SHA-256).

    Parameters
    ----------
    base_height_km, base_value : array_like
        The background / upper profile (e.g. NRLMSISE-00 or MIPAS) and
        the altitude grid the output is returned on.  Must span the
        full vertical range needed downstream (typically 0--120 km).
    profile_height_km, profile_value : array_like
        The user-supplied lower-atmosphere profile (e.g. ERA5).  May
        cover only a subset of ``base_height_km``; outside its
        coverage the base values are kept unchanged.
    blend_zone_km : tuple of (z_start_km, z_end_km), optional
        Altitude range of the linear-weighted transition.  Defaults to
        the topmost 6 km of ``profile_height_km`` (anchored to its
        top), matching molecfit's 6-km GDAS-to-MIPAS blend.
        Below ``z_start``: pure user profile.  Inside the zone:
        ``w_user(z) = (z_end - z)/(z_end - z_start)``.  Above ``z_end``:
        pure base.

    Returns
    -------
    np.ndarray
        Merged values sampled on ``base_height_km``.

    Notes
    -----
    The function operates on a single scalar field at a time.  To merge
    temperature, pressure and water-vapour profiles call it three times
    on the same blend zone.  The resulting arrays already span the complete
    grid and can be passed directly to LBLRTM without stitching the boundaries
    again.

    Examples
    --------
    >>> # ERA5 lower atmosphere + MIPAS upper, 6-km blend at ERA5 top
    >>> T_merged = edit_profile(z_mipas, T_mipas,
    ...                         era5.height_km, era5.temperature_K)
    """
    base_z = np.asarray(base_height_km, float)
    base_v = np.asarray(base_value, float)
    prof_z = np.asarray(profile_height_km, float)
    prof_v = np.asarray(profile_value, float)
    if base_z.shape != base_v.shape:
        raise ValueError("base_height_km and base_value must have the same length")
    if prof_z.shape != prof_v.shape:
        raise ValueError("profile_height_km and profile_value must have the same length")
    if prof_z.size < 2:
        raise ValueError("profile must have >= 2 levels for interpolation")

    if blend_zone_km is None:
        z_top = float(prof_z.max())
        blend_zone_km = (z_top - atm.BLEND_ZONE_WIDTH_KM, z_top)
    z_start, z_end = float(blend_zone_km[0]), float(blend_zone_km[1])
    if not (z_end > z_start):
        raise ValueError(f"blend_zone_km must have end > start, got ({z_start}, {z_end})")

    out = base_v.copy()
    below = base_z < z_start
    in_blend = (base_z >= z_start) & (base_z <= z_end)
    # above (base_z > z_end) -> base values stay unchanged

    if below.any():
        # np.interp clamps outside prof_z to its boundary values, which is
        # the sensible behaviour just below the user profile's bottom layer.
        out[below] = np.interp(base_z[below], prof_z, prof_v)
    if in_blend.any():
        w_user = (z_end - base_z[in_blend]) / (z_end - z_start)
        prof_at = np.interp(base_z[in_blend], prof_z, prof_v)
        out[in_blend] = w_user * prof_at + (1.0 - w_user) * base_v[in_blend]
    return out


# ==================================================================
# Full-profile (lower-stitched-with-upper) helpers and I/O
# ==================================================================
# A "full profile" is a dict on the fixed project altitude grid spanning
# 0--120 km, with keys
#     'z_km', 'T_K', 'P_hPa', 'H2O_ppmv'
# It is the merger of (lower-atmosphere, e.g. ERA5) with (upper-atmosphere,
# e.g. NRLMSISE-00 or MIPAS), produced via :func:`edit_profile` and
# typically saved to disk during a preprocessing pass; downstream code
# reads it back and pushes it straight into a forward model / TauCache,
# avoiding any per-fit retrieval or stitching.


_FULL_PROFILE_KEYS = ("z_km", "T_K", "P_hPa", "H2O_ppmv")

_REFERENCE_MOLECULE_FIELDS = (
    "CO2_ppmv",
    "O3_ppmv",
    "N2O_ppmv",
    "CO_ppmv",
    "CH4_ppmv",
    "O2_ppmv",
    "NO_ppmv",
    "SO2_ppmv",
    "NO2_ppmv",
    "NH3_ppmv",
    "HNO3_ppmv",
)


def add_reference_molecular_profiles(
    full: dict,
    reference_source: (
        Mapping | str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO
    ),
    *,
    expected_sha256: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Add the eleven non-water LBLRTM molecule profiles to a full profile.

    The full profile gives exposure/site-specific ``T_K``, ``P_hPa`` and ``H2O_ppmv``
    from the reanalysis/upper-atmosphere workflow.  The remaining engine
    molecules follow the climatological reference, as before.  This
    helper consumes either the already-loaded official RFM reference mapping
    or the same caller-supplied ``ngt.atm`` bytes/path and interpolates its
    dry-air ppmv fields onto the full-profile grid.  It does not modify the input
    mapping or replace the full profile's water profile.

    When ``reference_source`` is not an already-loaded mapping,
    ``expected_sha256`` and ``source_url`` are mandatory.
    """
    required = (*_FULL_PROFILE_KEYS,)
    missing = [key for key in required if key not in full]
    if missing:
        raise ValueError(f"full profile missing required keys: {missing}")
    target_z = np.asarray(full["z_km"], dtype=float)
    if target_z.ndim != 1 or target_z.size < 2 or not np.all(np.diff(target_z) > 0):
        raise ValueError("full profile z_km must be strictly increasing")
    if isinstance(reference_source, Mapping):
        reference = reference_source
        if expected_sha256 is not None or source_url is not None:
            raise ValueError(
                "expected_sha256/source_url must not accompany an already loaded reference mapping"
            )
    else:
        if expected_sha256 is None or source_url is None:
            raise ValueError("explicit RFM reference requires expected_sha256 and source_url")
        reference = extract_base_atmosphere(
            reference_source,
            expected_sha256=expected_sha256,
            source_url=source_url,
        )
    provenance = reference.get("reference_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("the molecular reference has no record of its source file")
    source_z = np.asarray(reference.get("z_km"), dtype=np.float64)
    if not np.array_equal(source_z, CANONICAL_ALTITUDE_GRID_KM):
        raise ValueError("molecular reference grid must match the 0--120 km grid")
    missing_fields = [
        field_name for field_name in _REFERENCE_MOLECULE_FIELDS if field_name not in reference
    ]
    if missing_fields:
        raise ValueError(f"molecular reference missing fields: {missing_fields}")
    result = {key: np.array(value, dtype=np.float64, copy=True) for key, value in full.items()}
    for field_name in _REFERENCE_MOLECULE_FIELDS:
        values = np.asarray(reference[field_name], dtype=np.float64)
        if (
            values.shape != source_z.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(f"invalid non-negative finite reference field {field_name}")
        result[field_name] = np.interp(target_z, source_z, values)
    return result


def save_full_profile(full: dict, dirpath: str, basename: str, metadata: dict | None = None) -> str:
    """Save a full profile (``z_km``, ``T_K``, ``P_hPa``, ``H2O_ppmv``) to
    ``{dirpath}/{basename}.dat`` with a ``#``-header saying how it was made.

    Format follows :func:`save_profile`'s convention (JSON-encoded scalar
    metadata + column block) so it round-trips exactly.  Returns the path.
    """
    for k in _FULL_PROFILE_KEYS:
        if k not in full:
            raise ValueError(f"full profile missing required key '{k}'")
    os.makedirs(dirpath, exist_ok=True)
    out_fn = os.path.join(dirpath, f"{basename}.dat")
    tmp_fn = out_fn + ".tmp"

    header = ["# full atmospheric profile (lower-stitched-with-upper)"]
    for k, v in (metadata or {}).items():
        header.append(f"# {k}: {json.dumps(v)}")
    header.append("# columns: " + " ".join(_FULL_PROFILE_KEYS))
    table = np.column_stack([np.asarray(full[k], float) for k in _FULL_PROFILE_KEYS])
    with open(tmp_fn, "w") as f:
        f.write("\n".join(header) + "\n")
        # Seventeen significant decimal digits round-trip every binary64
        # value.  The cache-hit profile must be numerically identical to the
        # just-built in-memory profile or its downstream profile SHA changes.
        np.savetxt(f, table, fmt="%.17g")
    os.replace(tmp_fn, out_fn)
    logger.info("Wrote %s", out_fn)
    return out_fn


def load_full_profile(dirpath: str, basename: str) -> tuple | None:
    """Load a full profile saved by :func:`save_full_profile`.

    Returns ``(full_dict, metadata_dict)`` or ``None`` if the file does
    not exist.  ``full_dict`` keys are ``z_km``, ``T_K``, ``P_hPa``,
    ``H2O_ppmv``; ``metadata_dict`` carries all the ``#``-header fields
    (lower_source, upper_source, blend_zone_km, pwv_mm, etc.).
    """
    fn = os.path.join(dirpath, f"{basename}.dat")
    if not os.path.exists(fn):
        return None
    meta: dict = {}
    cols: list = []
    with open(fn) as f:
        for line in f:
            if not line.startswith("#"):
                break
            s = line[1:].strip()
            if s.startswith("columns:"):
                cols = s[len("columns:") :].split()
            elif ": " in s:
                key, _, val = s.partition(": ")
                try:
                    meta[key] = json.loads(val)
                except json.JSONDecodeError:
                    meta[key] = val
    if cols != list(_FULL_PROFILE_KEYS):
        raise ValueError(f"unexpected columns in {fn}: got {cols}, want {list(_FULL_PROFILE_KEYS)}")
    table = np.loadtxt(fn)
    full = {c: table[:, i] for i, c in enumerate(cols)}
    return full, meta
