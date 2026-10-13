"""Atmospheric profile data object, physics helpers and storage."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np

# Physical constants: single source of truth in gtellcorr.constants.
from gtellcorr.constants import M_DRY, M_H2O, R_UNIV
from gtellcorr.data.atmosphere._context import atm as _atm

logger = logging.getLogger(__name__)

#: ``numpy.trapezoid`` (NumPy >= 2), else the older name ``trapz``.
_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201


def angular_sep_deg(lat1, lon1, lat2, lon2):
    """Great-circle separation [deg] between two (lat, lon) points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return float(np.degrees(2 * np.arcsin(np.sqrt(a))))


def check_site_coverage(prof, backend=""):
    """Raise ``ProfileFetchError`` if ``prof``'s grid point / archived site is
    farther than ``config.atmosphere.MAX_GRID_SEPARATION_DEG`` from the
    requested observatory.  Applied to BOTH freshly fetched and cache-loaded
    profiles (so a source that cannot cover the site is skipped instead of
    silently returning a wrong-location profile).  Global reanalyses are always
    within their grid spacing, so this never trips for them.
    """
    o_lat, o_lon = prof.obs_lat_deg, prof.obs_lon_deg
    g_lat, g_lon = prof.grid_lat_deg, prof.grid_lon_deg
    if all(np.isfinite(v) for v in (o_lat, o_lon, g_lat, g_lon)):
        sep = angular_sep_deg(o_lat, o_lon, g_lat, g_lon)
        if sep > _atm.MAX_GRID_SEPARATION_DEG:
            raise ProfileFetchError(
                f"{backend or prof.backend}: nearest grid/site is {sep:.1f} deg "
                f"(~{sep * 111:.0f} km) from the requested site, beyond the "
                f"{_atm.MAX_GRID_SEPARATION_DEG} deg coverage limit — "
                f"this source does not cover this location"
            )
    return prof


def has_grid_coords(prof):
    """True if ``prof`` records a finite grid/archive (lat, lon).

    ``check_site_coverage`` can only evaluate the separation when these are
    finite; a cached profile written before grid coordinates were recorded
    has NaN here and would bypass the guard, so callers loading from cache
    re-fetch when this returns False.
    """
    return bool(np.isfinite(prof.grid_lat_deg) and np.isfinite(prof.grid_lon_deg))


# ==================================================================
# Data container
# ==================================================================
@dataclass
class AtmosphereProfile:
    """A single atmospheric profile retrieved for one (lat, lon, time).

    All vertical arrays share length ``N`` and are ordered **bottom-up**
    (height increasing).  Units are explicit in the field names.
    """

    # -- Where the data came from / metadata --
    backend: str
    fetched_utc: str  # ISO-8601 UTC timestamp of retrieval
    source_url: str = ""  # URL or path the data came from

    # -- Requested location / time --
    obs_lat_deg: float = float("nan")  # decimal degrees North
    obs_lon_deg: float = float("nan")  # decimal degrees East (-180..180)
    obs_alt_km: float = float("nan")  # observatory altitude (km above MSL)
    obs_datetime_utc: str = ""  # ISO-8601 UTC of the science exposure

    # -- Grid point actually used --
    grid_lat_deg: float = float("nan")
    grid_lon_deg: float = float("nan")
    grid_datetime_utc: str = ""  # closest analysis cycle used

    # -- Vertical profile (bottom-up, length N) --
    height_km: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_hpa: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_K: np.ndarray = field(default_factory=lambda: np.array([]))
    h2o_vmr_ppmv: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- Optional auxiliary fields (length N or None) --
    relative_humidity_pct: np.ndarray | None = None
    geopotential_height_km: np.ndarray | None = None

    # -- Derived scalars --
    # The ``surface_*`` values are evaluated at the OBSERVATORY altitude
    # (``obs_alt_km``), i.e. the lowest level of the telescope line-of-sight
    # column -- not the sea-level surface.  Name kept for continuity.
    pwv_mm: float | None = None
    surface_temperature_K: float | None = None
    surface_pressure_hpa: float | None = None
    surface_h2o_ppmv: float | None = None
    surface_rh_pct: float | None = None

    # -- ASM merge trace (filled by merge_site_measurements) --
    merged_with_asm: bool = False
    asm_inputs: dict | None = None  # {'T_K':..., 'RH_pct':..., 'P_hpa':..., 'emix_km':...}
    asm_shifts: dict | None = None  # {'dpres':..., 'dtemp':..., 'dhum':...}

    # -- Notes --
    notes: str = ""

    # ---------- (de)serialization ----------
    def to_json_dict(self) -> dict:
        """All fields as JSON-ready values (arrays as lists)."""
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> AtmosphereProfile:
        """Inverse of :meth:`to_json_dict`."""
        kwargs = dict(d)
        for k in (
            "height_km",
            "pressure_hpa",
            "temperature_K",
            "h2o_vmr_ppmv",
            "relative_humidity_pct",
            "geopotential_height_km",
        ):
            if k in kwargs and kwargs[k] is not None:
                kwargs[k] = np.asarray(kwargs[k], dtype=float)
        # Drop any keys not in the dataclass (forward compatibility).
        valid = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**kwargs)


# ==================================================================
# Physics helpers
# ==================================================================
def saturation_vapor_pressure_pa(T_K: np.ndarray) -> np.ndarray:
    """Murphy & Koop (2005) saturation vapor pressure (Pa).

    Computes both the liquid (Murphy&Koop Eq.10) and ice (Eq.7) branches,
    e_sat = min(liquid, ice), matching molecfit mf_atm_rhum_to_ppmv
    (mf_atm_combined_gdas.c:1471-1484)."""
    T = np.asarray(T_K, dtype=float)
    logt = np.log(T)
    # log_ew: vapor pressure over liquid water (Murphy&Koop Eq.10)
    # log_ew = 54.842763 − 6763.22/T − 4.210·ln(T) + 0.000367·T
    #         + tanh(0.0415·(T − 218.8))·(53.878 − 1331.22/T
    #         − 9.44523·ln(T) + 0.014025·T)
    log_ew = (
        54.842763
        - 6763.22 / T
        - 4.210 * logt
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * logt + 0.014025 * T)
    )
    # log_ei: vapor pressure over hexagonal ice (Murphy&Koop Eq.7)
    # log_ei = 9.550426 − 5723.265/T + 3.53068·ln(T) − 0.00728332·T
    log_ei = 9.550426 - 5723.265 / T + 3.53068 * logt - 0.00728332 * T
    # e_sat = min(liquid, ice)  (molecfit CPL_MIN, line ~1484)
    return np.exp(np.minimum(log_ew, log_ei))


def rh_to_vmr_ppmv(rh_pct: np.ndarray, T_K: np.ndarray, P_hpa: np.ndarray) -> np.ndarray:
    """Relative humidity (%) → H₂O volume mixing ratio (ppmv).

    VMR = e_actual / P_dry ≈ e_actual / P (small-mixing-ratio limit).
    Returns ppmv = mol H₂O / (10⁶ mol total air).
    """
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 110.0)  # tiny tolerance
    e_sat_pa = saturation_vapor_pressure_pa(T_K)
    e_actual_pa = rh / 100.0 * e_sat_pa
    P_pa = np.asarray(P_hpa, dtype=float) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vmr = e_actual_pa / np.maximum(P_pa, 1.0)  # dimensionless mole fraction
    return 1.0e6 * vmr


def compute_pwv_mm(profile: AtmosphereProfile, *, above_km=None) -> float:
    """Integrate H₂O VMR over altitude to a precipitable water value.

    Uses molecfit Eq. (9):  PWV [kg/m²] = (M_H₂O / R_univ)
    · ∫ x_H₂O(h) · P(h) / T(h) dh,   with 1 kg/m² ≡ 1 mm liquid column.

    Integration is performed only over the column ABOVE the observatory
    (``profile.obs_alt_km``), i.e. the telescope line-of-sight column,
    mirroring molecfit ``mf_atm_calpwv`` which skips layers at/below the
    site elevation and uses a partial bottom layer starting at geoelev
    (mf_atm_combined_gdas.c:1398-1403).  If ``obs_alt_km`` is not finite
    or lies below the lowest profile level, the full column is integrated
    (backward-compatible fallback).

    ``above_km`` overrides the lower integration bound (``None`` → use
    ``profile.obs_alt_km``).  Pass a value below the lowest level for the
    whole-column PWV, or any altitude to integrate only above it.

    Returns
    -------
    pwv_mm : float
        Precipitable water vapor column in mm (= kg/m²).
    """
    z = profile.height_km
    if z.size < 2:
        return float("nan")
    # Sort bottom-up if needed
    if z[0] > z[-1]:
        order = np.argsort(z)
        z = z[order]
        T = profile.temperature_K[order]
        P = profile.pressure_hpa[order]
        x = profile.h2o_vmr_ppmv[order] * 1e-6
    else:
        z = np.asarray(z, dtype=float)
        T = np.asarray(profile.temperature_K, dtype=float)
        P = np.asarray(profile.pressure_hpa, dtype=float)
        x = np.asarray(profile.h2o_vmr_ppmv, dtype=float) * 1e-6

    # Restrict to the column above the observatory (telescope line of sight),
    # matching molecfit mf_atm_calpwv geoelev skip + partial bottom layer.
    obs = float(profile.obs_alt_km if above_km is None else above_km)
    if np.isfinite(obs) and z[0] < obs < z[-1]:
        # Linearly interpolate the bottom node onto the observatory altitude.
        T0 = float(np.interp(obs, z, T))
        P0 = float(np.interp(obs, z, P))
        x0 = float(np.interp(obs, z, x))
        # Keep only levels strictly above the observatory, prepend obs node.
        above = z > obs
        z = np.concatenate(([obs], z[above]))
        T = np.concatenate(([T0], T[above]))
        P = np.concatenate(([P0], P[above]))
        x = np.concatenate(([x0], x[above]))

    # P in Pa, h in m, T in K → integrand SI
    P_pa = P * 100.0
    integrand = x * P_pa / np.maximum(T, 1.0)  # Pa / K
    # Trapezoidal integration over altitude in metres.
    integral = float(_TRAPEZOID(integrand, z * 1000.0))
    # Convert: PWV [kg/m²] = M_H₂O / R_univ · ∫ x P / T dh   (h in m, P in Pa)
    # M_H2O = 0.0180153 kg/mol is the physically correct molar mass of water
    # (IUPAC 18.015 g/mol).  molecfit uses 0.0182 (MF_MOL_MASS_H2O), ~1% high,
    # so our PWV runs ~1% below molecfit's for an identical profile — this is
    # the molar-mass choice, not a formula difference.
    pwv_kg_m2 = M_H2O / R_UNIV * integral
    # 1 kg/m² = 1 mm of liquid water column
    return float(pwv_kg_m2)


# Molar-mass ratio M_dry / M_H2O used to convert specific humidity
# (kg/kg) to volume mixing ratio (mol/mol).
_R_MOLMASS = M_DRY / M_H2O  # ≈ 1.6078


def _specific_humidity_to_ppmv_rh(q, P_hpa, T_K):
    """Specific humidity ``q`` (kg/kg) + pressure (hPa) + temperature (K)
    → (H2O volume mixing ratio in ppmv, relative humidity % for the
    diagnostic slot).  Shared by the ERA5 and MERRA-2 backends.

        x    = q / ((M_H2O/M_dry)·(1−q) + q)        # mole fraction
        ppmv = 1e6 · x
        RH   = 100 · (x·P) / e_sat(T)               # clipped to [0, 200]
    """
    x = q / ((1.0 / _R_MOLMASS) * (1.0 - q) + q)
    ppmv = 1.0e6 * x
    e_actual_pa = x * np.asarray(P_hpa, float) * 100.0
    e_sat_pa = saturation_vapor_pressure_pa(T_K)
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = np.clip(100.0 * e_actual_pa / np.maximum(e_sat_pa, 1.0), 0.0, 200.0)
    return ppmv, rh


def _populate_derived(profile: AtmosphereProfile) -> None:
    """Fill PWV and *observatory-altitude* (a.k.a. "surface") scalars
    in-place.  Values at the obs altitude are linearly interpolated from
    the profile (height-monotonic); if obs_alt_km is unknown we fall
    back to the lowest level in the profile."""
    z = profile.height_km
    if z.size == 0:
        return
    profile.pwv_mm = compute_pwv_mm(profile)

    order = np.argsort(z)
    z_sorted = z[order]
    if np.isfinite(profile.obs_alt_km):
        z_query = float(np.clip(profile.obs_alt_km, z_sorted[0], z_sorted[-1]))
    else:
        z_query = float(z_sorted[0])

    def _interp(arr):
        """*arr* at the query altitude (``None`` stays ``None``)."""
        if arr is None:
            return None
        return float(np.interp(z_query, z_sorted, np.asarray(arr)[order]))

    profile.surface_temperature_K = _interp(profile.temperature_K)
    profile.surface_pressure_hpa = _interp(profile.pressure_hpa)
    profile.surface_h2o_ppmv = _interp(profile.h2o_vmr_ppmv)
    if profile.relative_humidity_pct is not None:
        profile.surface_rh_pct = _interp(profile.relative_humidity_pct)


# ==================================================================
# Storage
# ==================================================================
# vertical-array fields, in column order; optional ones appended when present
_PROFILE_ARRAY_KEYS = (
    "height_km",
    "pressure_hpa",
    "temperature_K",
    "h2o_vmr_ppmv",
    "relative_humidity_pct",
    "geopotential_height_km",
)


def save_profile(profile: AtmosphereProfile, dirpath: str, basename: str) -> str:
    """Save a profile as ``{dirpath}/{basename}.dat``.

    The .dat file is a text table: a ``#``-commented header carrying the
    scalar metadata (each value JSON-encoded so it round-trips exactly)
    followed by a columnar block of the vertical arrays.  Returns the full
    path written.  Atomic rename via .tmp.
    """
    os.makedirs(dirpath, exist_ok=True)
    out_fn = os.path.join(dirpath, f"{basename}.dat")
    tmp_fn = out_fn + ".tmp"

    d = asdict(profile)
    cols = [k for k in _PROFILE_ARRAY_KEYS if d.get(k) is not None and np.size(d[k]) > 0]
    header = ["# atmospheric profile (lower atmosphere)"]
    for k, v in d.items():
        if k not in _PROFILE_ARRAY_KEYS:
            header.append(f"# {k}: {json.dumps(v)}")
    header.append("# columns: " + " ".join(cols))
    table = np.column_stack([np.asarray(d[k], float) for k in cols])
    with open(tmp_fn, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, table, fmt="%.8g")
    os.replace(tmp_fn, out_fn)
    logger.info("Wrote %s", out_fn)
    return out_fn


def load_profile(dirpath: str, basename: str) -> AtmosphereProfile | None:
    """Load a previously saved ``{basename}.dat`` profile, or ``None``."""
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
                meta[key] = json.loads(val)
    table = np.loadtxt(fn)
    if table.ndim == 1:
        table = table.reshape(1, -1) if table.size else table.reshape(0, len(cols))
    for i, c in enumerate(cols):
        meta[c] = table[:, i]
    return AtmosphereProfile.from_json_dict(meta)


# ==================================================================
# Small utilities
# ==================================================================
class ProfileFetchError(RuntimeError):
    """Raised when a remote profile fetch fails."""


def _now_iso() -> str:
    """Current UTC time, ISO 8601, whole seconds."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
