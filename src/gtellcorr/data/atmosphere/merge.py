"""On-site meteorology merge (port of molecfit mf_atm_adapt)."""

from __future__ import annotations

import numpy as np

# atmosphere parameters of the current call (AtmosphereConfig)
from gtellcorr.data.atmosphere._context import atm as _atm

from .profile import (
    AtmosphereProfile,
    _populate_derived,
    rh_to_vmr_ppmv,
)


# ==================================================================
# On-site meteorology merge (port of molecfit mf_atm_adapt)
# ==================================================================
def merge_site_measurements(
    profile: AtmosphereProfile,
    *,
    surface_T_K: float | None = None,
    surface_RH_pct: float | None = None,
    surface_P_hpa: float | None = None,
    emix_km: float | None = None,
    force_below_obs: bool = True,
) -> AtmosphereProfile:
    """Blend on-site ASM measurements into the lowest layers of a profile.

    This is a Python port of telluriccorr's ``mf_atm_adapt``
    (``mf_atm_combined_gdas.c:1219``).  The algorithm:

      1. Interpolate the *background* profile (e.g. NCEP / GFS) at the
         observatory altitude → background values ``P_bg``, ``T_bg``,
         ``h2o_bg`` (ppmv, converted from RH if needed).
      2. Compute multiplicative shifts at the surface::

             dpres = P_site / P_bg − 1
             dtemp = T_site / T_bg − 1
             dhum  = h2o_site / h2o_bg − 1

         (any of the three is set to 0 if the corresponding site value
         is ``None``; that quantity is then left untouched.)
      3. Apply a *linear* taper between the observatory altitude and
         ``emix_km``::

             frac = clip((z − obs_alt) / (emix − obs_alt), 0, 1)
             P[i] *= 1 + dpres · (1 − frac)
             T[i] *= 1 + dtemp · (1 − frac)
             h2o[i] *= 1 + dhum  · (1 − frac)

         The taper bottom (frac = 0) is full ASM correction; the taper
         top (frac = 1, at ``emix_km``) is the unmodified background.
      4. *Optional* (``force_below_obs=True``, matches molecfit):
         layers with z < obs_alt have T and h2o forced to the site
         values exactly — improves LBLRTM numerical stability at the
         bottom boundary.  Pressure still gets the multiplicative
         correction.

    Notes
    -----
    - This is **multiplicative**, not additive, on T.  Mathematically a
      bit odd for temperature but it's what molecfit does (and it
      avoids negative-T blowups when the shift is large).
    - ``emix_km`` is the **absolute geopotential altitude (km MSL)**
      where the ASM correction tapers to zero — same convention as
      molecfit's ``atmospheric.emix``.  Default 5.0 km matches
      molecfit's Paranal default and works at KPNO too (obs_alt =
      2.091 km → ~2.9 km of taper above the observatory).  For high
      observatories choose a value at least 1–2 km above ``obs_alt``.
    - ``surface_RH_pct`` is converted to ppmv via Murphy-Koop ``e_sat``
      at the *site-measured* T and P (the same formula molecfit uses).

    Parameters
    ----------
    profile : AtmosphereProfile
        Background profile (typically from ``fetch('ncep_r1_psl', ...)``).
        Not modified.
    surface_T_K : float or None
        Observatory-altitude air temperature in Kelvin (NEID:
        ``ENVOTMP`` + 273.15).
    surface_RH_pct : float or None
        Relative humidity at the observatory (NEID: ``ENVOHUM``).
    surface_P_hpa : float or None
        Surface pressure in hPa.  NEID L2 doesn't expose this in the
        primary header — pass the climatological value at the
        observatory altitude, or ``None`` to leave pressure untouched.
    emix_km : float
        Upper mixing height in km (geopotential).  Default from
        config.atmosphere.ASM_EMIX_KM (5.0 km), matching molecfit's
        MF_PARAMETERS_EMIX_INIT = 5.0 (mf_parameters.h:560).
    force_below_obs : bool
        If True (default), layers below the observatory altitude get
        T and h2o forced to the site values; only P keeps the
        multiplicative correction.  Set False to keep the linear taper
        all the way down.

    Returns
    -------
    AtmosphereProfile
        A new profile (deep copy of ``profile``) with merged arrays and
        ``merged_with_asm=True`` plus ``asm_inputs`` / ``asm_shifts``
        recording what was applied.
    """
    z = np.asarray(profile.height_km, dtype=float).copy()
    P = np.asarray(profile.pressure_hpa, dtype=float).copy()
    T = np.asarray(profile.temperature_K, dtype=float).copy()
    h2o = np.asarray(profile.h2o_vmr_ppmv, dtype=float).copy()
    RH = (
        None
        if profile.relative_humidity_pct is None
        else np.asarray(profile.relative_humidity_pct, dtype=float).copy()
    )
    geo = (
        None
        if profile.geopotential_height_km is None
        else np.asarray(profile.geopotential_height_km, dtype=float).copy()
    )

    obs_alt = float(profile.obs_alt_km)
    if not np.isfinite(obs_alt):
        raise ValueError(
            "profile.obs_alt_km is not set; cannot anchor "
            "the ASM taper.  Edit the profile metadata."
        )
    if emix_km is None:  # default: atmosphere.asm_emix_km of the current call
        emix_km = _atm.ASM_EMIX_KM
    if emix_km <= obs_alt:
        raise ValueError(
            f"emix_km ({emix_km}) must be greater than obs_alt_km "
            f"({obs_alt}); cannot taper downward."
        )

    # Ensure the profile has a layer exactly at obs_alt — otherwise the
    # "force below / taper above" treatment produces a step at obs_alt
    # because nearby layers may be a km away on either side.  Insert
    # a fresh layer interpolated from background; subsequent ASM
    # corrections at frac=0 will land cleanly on it.
    order = np.argsort(z)
    z_s = z[order]
    if not np.any(np.isclose(z_s, obs_alt, atol=1e-3)):
        i_insert = int(np.searchsorted(z_s, obs_alt))
        # Original arrays — interpolate, then splice
        P_at = float(np.interp(obs_alt, z_s, P[order]))
        T_at = float(np.interp(obs_alt, z_s, T[order]))
        h2o_at = float(np.interp(obs_alt, z_s, h2o[order]))
        # Splice into the *original* (un-sorted) arrays in sorted order
        z = np.insert(z[order], i_insert, obs_alt)
        P = np.insert(P[order], i_insert, P_at)
        T = np.insert(T[order], i_insert, T_at)
        h2o = np.insert(h2o[order], i_insert, h2o_at)
        if RH is not None:
            RH_at = float(np.interp(obs_alt, z_s, RH[order]))
            RH = np.insert(RH[order], i_insert, RH_at)
        # geopotential height must track height_km (same length + ordering),
        # or save_profile's column_stack and any height/geopotential pairing
        # downstream desync after the splice.
        if geo is not None:
            geo_at = float(np.interp(obs_alt, z_s, geo[order]))
            geo = np.insert(geo[order], i_insert, geo_at)

    # ---- 1) Background values at obs altitude (linear in z) ----
    order = np.argsort(z)
    z_s, P_s, T_s, h2o_s = z[order], P[order], T[order], h2o[order]
    P_bg = float(np.interp(obs_alt, z_s, P_s))
    T_bg = float(np.interp(obs_alt, z_s, T_s))
    h2o_bg = float(np.interp(obs_alt, z_s, h2o_s))

    # ---- 2) Site values & shifts ----
    P_site = float(surface_P_hpa) if surface_P_hpa is not None else None
    T_site = float(surface_T_K) if surface_T_K is not None else None
    if surface_RH_pct is not None:
        # Convert site RH to ppmv at the *site* (T_site, P_site) pair —
        # falling back to background if either is missing.
        T_for_conv = T_site if T_site is not None else T_bg
        P_for_conv = P_site if P_site is not None else P_bg
        h2o_site = float(
            rh_to_vmr_ppmv(
                np.atleast_1d(surface_RH_pct),
                np.atleast_1d(T_for_conv),
                np.atleast_1d(P_for_conv),
            )[0]
        )
    else:
        h2o_site = None

    dpres = (P_site / P_bg - 1.0) if (P_site is not None and P_bg > 0) else 0.0
    dtemp = (T_site / T_bg - 1.0) if (T_site is not None and T_bg > 0) else 0.0
    # Zero-H2O guard (molecfit mf_atm_adapt, mf_atm_combined_gdas.c:1261-1264):
    # when the interpolated model/background H2O is effectively zero (or not
    # finite), skip the humidity correction (dhum=0) rather than forming a huge
    # or non-finite multiplier — molecfit reverts to the station value there.
    if h2o_site is not None and np.isfinite(h2o_bg) and h2o_bg > 1e-30:
        dhum = h2o_site / h2o_bg - 1.0
    else:
        dhum = 0.0

    # ---- 3) Apply linear taper ----
    frac = np.clip((z - obs_alt) / (emix_km - obs_alt), 0.0, 1.0)
    w = 1.0 - frac  # full at obs_alt, zero at emix_km

    # Pressure: multiplicative everywhere within taper
    P_new = P * (1.0 + dpres * w)
    T_new = T * (1.0 + dtemp * w)
    h2o_new = h2o * (1.0 + dhum * w)

    # ---- 4) Bottom-boundary clamp (molecfit behaviour) ----
    if force_below_obs:
        below = z < obs_alt
        if T_site is not None:
            T_new[below] = T_site
        if h2o_site is not None:
            h2o_new[below] = h2o_site
        # P stays as multiplicative — already correct.

    # ---- 5) Build new profile (deep copy + replacement) ----
    new_prof = AtmosphereProfile(
        backend=profile.backend,
        fetched_utc=profile.fetched_utc,
        source_url=profile.source_url,
        obs_lat_deg=profile.obs_lat_deg,
        obs_lon_deg=profile.obs_lon_deg,
        obs_alt_km=profile.obs_alt_km,
        obs_datetime_utc=profile.obs_datetime_utc,
        grid_lat_deg=profile.grid_lat_deg,
        grid_lon_deg=profile.grid_lon_deg,
        grid_datetime_utc=profile.grid_datetime_utc,
        height_km=z,
        pressure_hpa=P_new,
        temperature_K=T_new,
        h2o_vmr_ppmv=h2o_new,
        relative_humidity_pct=RH,  # not renormalised; advisory only
        geopotential_height_km=geo,
        merged_with_asm=True,
        asm_inputs={
            "T_K": T_site,
            "RH_pct": float(surface_RH_pct) if surface_RH_pct is not None else None,
            "h2o_ppmv": h2o_site,
            "P_hpa": P_site,
            "emix_km": float(emix_km),
            "force_below_obs": bool(force_below_obs),
        },
        asm_shifts={
            "dpres": float(dpres),
            "dtemp": float(dtemp),
            "dhum": float(dhum),
            "P_bg_at_obs": P_bg,
            "T_bg_at_obs": T_bg,
            "h2o_bg_at_obs": h2o_bg,
        },
        notes=(
            profile.notes + " [+ASM merge: "
            f"emix={emix_km:.1f} km, dpres={dpres:+.3g}, "
            f"dtemp={dtemp:+.3g}, dhum={dhum:+.3g}]"
        ).strip(),
    )
    _populate_derived(new_prof)
    return new_prof
