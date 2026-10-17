"""Backend: NASA MERRA-2 via Earthdata Cloud OPeNDAP (DAP4 CSV)."""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import UTC, datetime
from urllib.parse import quote

import numpy as np

from ..cycles import _nearest_merra2_cycle
from ..profile import (
    AtmosphereProfile,
    ProfileFetchError,
    _now_iso,
    _populate_derived,
    _specific_humidity_to_ppmv_rh,
)
from ..secrets import load_weather_keys

logger = logging.getLogger(__name__)


# ==================================================================
# Backend: NASA MERRA-2 via Earthdata Cloud OPeNDAP (DAP4 CSV)
# ==================================================================
# Product:  M2I3NPASM  — Modern-Era Retrospective analysis for Research
#           and Applications, v2; 3-hourly, instantaneous, pressure-
#           level, assimilation, meteorological fields.
#           This is the same product PSG (Villanueva+18) uses.
# Grid   :  0.5° lat × 0.625° lon (361×576), 42 pressure levels
#           (1000 → 0.1 hPa), 8 cycles/day (00, 03, …, 21 UTC).
# Auth   :  NASA Earthdata Login bearer token,
#           WEATHER_KEYS['merra2_gesdisc']['EARTHDATA_TOKEN'] in the
#           credentials file (Settings.credentials_file).
# Access :  Earthdata Cloud OPeNDAP returns one CSV response per query;
#           we request the DAP4 subset of the 2x2 grid cell around the
#           site for (T, QV, H) plus
#           the coordinate variables, parse the text, and assemble a
#           AtmosphereProfile.  Stays inside the requests-only stack.
#           (GES DISC retired its own OPeNDAP server, goldsmr5, in 2026;
#           it now answers HTTP 410.)
#
# File names rotate by "stream" (= reprocessing version):
#       MERRA2_100   1980-01-01 → 1991-12-31
#       MERRA2_200   1992-01-01 → 2000-12-31
#       MERRA2_300   2001-01-01 → 2010-12-31
#       MERRA2_400   2011-01-01 → present
# (Occasional MERRA2_401 replacements occur in the post-2020 window;
#  the loader retries with _401 if _400 returns 404.)
#: Earthdata Cloud OPeNDAP granules of M2I3NPASM (CMR collection C1276812879-GES_DISC).
OPENDAP_GRANULES = (
    "https://opendap.earthdata.nasa.gov/collections/C1276812879-GES_DISC/granules/"
    "M2I3NPASM.5.12.4%3A"
)

# MERRA-2 pressure-level vertical coordinate (hPa).  Hard-coded because
# it's a documented fixed grid; we still cross-check against the
# returned ``lev[]`` array.
MERRA2_LEVELS_HPA = np.array(
    [
        1000.0,
        975.0,
        950.0,
        925.0,
        900.0,
        875.0,
        850.0,
        825.0,
        800.0,
        775.0,
        750.0,
        725.0,
        700.0,
        650.0,
        600.0,
        550.0,
        500.0,
        450.0,
        400.0,
        350.0,
        300.0,
        250.0,
        200.0,
        150.0,
        100.0,
        70.0,
        50.0,
        40.0,
        30.0,
        20.0,
        10.0,
        7.0,
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
        0.7,
        0.5,
        0.4,
        0.3,
        0.1,
    ]
)


def _merra2_stream_for(date_utc: datetime) -> str:
    """Return the MERRA-2 file-name stream (``MERRA2_xxx``) for a date."""
    y = date_utc.year
    if y < 1992:
        return "MERRA2_100"
    if y < 2001:
        return "MERRA2_200"
    if y < 2011:
        return "MERRA2_300"
    return "MERRA2_400"


def _merra2_brackets(lat: float, lon: float) -> tuple[int, int, int, int]:
    """Contiguous bracketing grid indices (i0, i1, j0, j1) of the 2x2 cell
    surrounding (lat, lon), for bilinear interpolation.  lat grid is
    -90 + 0.5*i (i = 0..360); lon grid is -180 + 0.625*j (j = 0..575).
    Clamped at the poles / antimeridian (degrades to an edge extrapolation
    there rather than wrapping the index range)."""
    fi = (lat + 90.0) / 0.5
    i0 = min(max(int(np.floor(fi)), 0), 359)
    lon_n = lon - 360.0 if lon > 180.0 else lon
    fj = (lon_n + 180.0) / 0.625
    j0 = min(max(int(np.floor(fj)), 0), 574)
    return i0, i0 + 1, j0, j0 + 1


def _earthdata_session():
    """Return a ``requests.Session`` that survives the URS redirect chain.

    GES DISC OPeNDAP redirects through ``urs.earthdata.nasa.gov`` (cross-
    host), and ``requests`` strips the ``Authorization`` header on cross-
    host redirects by default for security.  We subclass ``Session`` and
    no-op the ``rebuild_auth`` hook so the bearer token from the
    credentials file survives the bounce.
    """
    import requests

    class _EarthdataSession(requests.Session):
        """A session that keeps the bearer token across redirects."""

        def rebuild_auth(self, prepared_request, response):
            """Keep Authorization across Earthdata Login redirects."""
            return

    s = _EarthdataSession()
    keys = load_weather_keys("merra2_gesdisc")
    tok = keys.get("EARTHDATA_TOKEN", "").strip()
    if not tok:
        raise ProfileFetchError('WEATHER_KEYS["merra2_gesdisc"] must define EARTHDATA_TOKEN')
    s.headers["Authorization"] = f"Bearer {tok}"
    return s


def fetch_merra2_gesdisc(
    lat: float,
    lon: float,
    dt_utc: datetime,
    *,
    obs_alt_km: float = float("nan"),
    timeout_s: float = 90.0,
    session=None,
    interpolate: bool = True,
) -> AtmosphereProfile:
    """Fetch a single-point MERRA-2 ``M2I3NPASM`` profile via Earthdata Cloud OPeNDAP.

    Parameters
    ----------
    lat, lon : float
        Site coordinates in decimal degrees.  ``lon`` may be in (-180,
        180] or [0, 360); normalised internally.
    dt_utc : datetime
        Observation time in UTC.  The closest MERRA-2 3-hourly cycle is
        selected (no time interpolation — to keep parity with the
        single-cycle behaviour of the gfs_thredds backend).
    obs_alt_km : float
        Observatory altitude in km; stored in metadata.
    timeout_s : float
        Per-request timeout (s).  OPeNDAP can be slow on large queries.
    session : requests.Session or None
        Optional persistent session.

    Returns
    -------
    AtmosphereProfile

    Raises
    ------
    ProfileFetchError : on any network / parsing failure (including auth
        errors, missing date in the archive, or unexpected OPeNDAP
        response layout).

    Notes
    -----
    The OPeNDAP DAP4 CSV response is plain text; we use a small hand-
    written parser to avoid pulling in ``pydap`` / ``xarray``.  Specific
    humidity (kg H2O / kg moist air) is converted to volume mixing ratio
    (ppmv) via the dry/moist molar-mass identity::

        x_h2o = q / (M_H2O / M_dry * (1 - q) + q)

    PWV is then re-computed from the resulting profile by
    ``_populate_derived``.
    """
    import requests

    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    dt_utc = dt_utc.astimezone(UTC)

    if session is None:
        session = _earthdata_session()

    cycle_dt, t_idx = _nearest_merra2_cycle(dt_utc)
    # Stream era is keyed off the resolved cycle date (not the raw exposure
    # date) so a cycle that rolls across a year boundary picks the correct
    # MERRA2_xxx stream.
    stream = _merra2_stream_for(cycle_dt)
    # Fetch the surrounding 2x2 grid cell, then either interpolate bilinearly to the
    # exact (lat, lon) -- parity with the ERA5 backend -- or take its node nearest to
    # the site. (The cloud OPeNDAP CSV returns no values when the last dimension is a
    # single index, so a single node is never requested on its own.)
    i0, i1, j0, j1 = _merra2_brackets(lat, lon)
    lat_sub, lon_sub = f"{i0}:1:{i1}", f"{j0}:1:{j1}"
    lat_g, lon_g = lat, lon

    # Build the DAP4 CSV subset URL.  Vars: T, QV (specific humidity), H
    # (geopotential height); plus the coord vars so we can verify the
    # grid point the server actually resolved to.
    def _build_url(stream_name: str) -> str:
        """OPeNDAP DAP4 CSV URL of the grid-cell subset in *stream_name*."""
        fname = f"{stream_name}.inst3_3d_asm_Np.{cycle_dt:%Y%m%d}.nc4"
        # DAP4 subset syntax: /VAR[start:stride:stop] per dimension, ";" between
        ce = ";".join(
            [
                f"/T[{t_idx}:1:{t_idx}][0:1:41][{lat_sub}][{lon_sub}]",
                f"/QV[{t_idx}:1:{t_idx}][0:1:41][{lat_sub}][{lon_sub}]",
                f"/H[{t_idx}:1:{t_idx}][0:1:41][{lat_sub}][{lon_sub}]",
                "/lev[0:1:41]",
                f"/lat[{lat_sub}]",
                f"/lon[{lon_sub}]",
                f"/time[{t_idx}:1:{t_idx}]",
            ]
        )
        return f"{OPENDAP_GRANULES}{fname}.dap.csv?dap4.ce={quote(ce, safe='/[]:;')}"

    url = _build_url(stream)
    logger.info("MERRA-2 fetch: %s", url)

    try:
        r = session.get(url, timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        raise ProfileFetchError(
            f"Network error retrieving MERRA-2 profile: {e}\n  url={url}"
        ) from e

    # Some dates post-2020 are served under MERRA2_401 instead of _400.
    if r.status_code == 404 and stream == "MERRA2_400":
        alt_url = _build_url("MERRA2_401")
        logger.info("MERRA-2 404 on _400; retrying _401: %s", alt_url)
        try:
            r = session.get(alt_url, timeout=timeout_s)
        except requests.exceptions.RequestException as e:
            raise ProfileFetchError(f"Network error on MERRA2_401 fallback: {e}") from e
        url = alt_url

    # 404 here means the daily file doesn't exist yet (MERRA-2 has ~3-week
    # latency for the standard product, but the lag can stretch to 6+
    # weeks depending on the GMAO ingest queue).
    if r.status_code == 404:
        raise ProfileFetchError(
            f"MERRA-2 file not yet available on GES DISC for "
            f"{cycle_dt:%Y-%m-%d} (HTTP 404).  Standard product "
            f"latency is ~3 weeks; pick an older date or wait for "
            f"GMAO to ingest.  URL: {url}"
        )

    if r.status_code == 401:
        raise ProfileFetchError(
            "MERRA-2 GES DISC returned HTTP 401 (unauthorised). Set "
            'WEATHER_KEYS["merra2_gesdisc"]["EARTHDATA_TOKEN"] in the credentials file '
            "(the credentials_file setting or the GTELLCORR_CREDENTIALS environment "
            "variable). See "
            "https://disc.gsfc.nasa.gov/data-access#python."
        )
    if r.status_code != 200:
        raise ProfileFetchError(
            f"MERRA-2 GES DISC returned HTTP {r.status_code} for {url}\n"
            f"Response head: {r.text[:400]}"
        )
    text = r.text.strip()
    if not text or "<html" in text[:200].lower():
        raise ProfileFetchError(
            f"MERRA-2 response is HTML / empty (likely auth failure or "
            f"invalid date).  Head:\n{text[:400]}"
        )

    parsed = _parse_opendap_csv(text)

    # Pull the four arrays we need from the parsed bag of variables.
    try:
        lev_hpa = np.asarray(parsed["lev"], dtype=float).ravel()
        nlev = lev_hpa.size

        def _block(name):
            """Variable *name* of the response as a (level, 2, 2) array."""
            a = np.asarray(parsed[name], dtype=float).ravel()
            if a.size != nlev * 4:
                raise ProfileFetchError(
                    f"MERRA-2 {name}: expected {nlev * 4} values for the "
                    f"2x2 grid cell, got {a.size}"
                )
            return a.reshape(nlev, 2, 2)  # [lev, lat, lon]

        lats = np.asarray(parsed["lat"], dtype=float).ravel()
        lons = np.asarray(parsed["lon"], dtype=float).ravel()
        lon_n = lon - 360.0 if lon > 180.0 else lon
        if interpolate:
            wlat = (
                0.0
                if lats[1] == lats[0]
                else float(np.clip((lat - lats[0]) / (lats[1] - lats[0]), 0.0, 1.0))
            )
            wlon = (
                0.0
                if lons[1] == lons[0]
                else float(np.clip((lon_n - lons[0]) / (lons[1] - lons[0]), 0.0, 1.0))
            )
            w = np.array(
                [[(1 - wlat) * (1 - wlon), (1 - wlat) * wlon], [wlat * (1 - wlon), wlat * wlon]]
            )  # [lat,lon]

            def _bilin(B):
                """Bilinear interpolation to the site, ignoring fill values."""
                # NaN-aware: mask MERRA-2 fill values (1e15) at any corner and
                # renormalise the weights over the finite corners per level, so
                # a level that is below ground at one node still interpolates
                # from the valid neighbours instead of being poisoned.
                B = np.where(np.abs(B) > 1e10, np.nan, B)
                num = np.nansum(B * w[None, :, :], axis=(1, 2))
                den = (w[None, :, :] * np.isfinite(B)).sum(axis=(1, 2))
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(den > 0, num / den, np.nan)

            T_K = _bilin(_block("T"))
            QV = _bilin(_block("QV"))
            H_m = _bilin(_block("H"))
        else:
            ia = int(np.argmin(np.abs(lats - lat)))
            ja = int(np.argmin(np.abs(lons - lon_n)))
            lat_g, lon_g = float(lats[ia]), float(lons[ja])
            T_K = _block("T")[:, ia, ja]
            QV = _block("QV")[:, ia, ja]
            H_m = _block("H")[:, ia, ja]
    except KeyError as e:
        raise ProfileFetchError(
            f"MERRA-2 response missing variable {e}; got keys={list(parsed)}"
        ) from None

    # Cross-check vertical grid (cheap sanity).
    if lev_hpa.size != MERRA2_LEVELS_HPA.size or not np.allclose(
        lev_hpa, MERRA2_LEVELS_HPA, rtol=1e-3
    ):
        logger.warning(
            "MERRA-2 returned %d levels (expected 42); using returned grid.",
            lev_hpa.size,
        )

    # Mask fill values (MERRA-2 uses 1e15).
    fill_mask = (np.abs(T_K) > 1e10) | (np.abs(QV) > 1e10) | (np.abs(H_m) > 1e10)
    if fill_mask.any():
        T_K = np.where(fill_mask, np.nan, T_K)
        QV = np.where(fill_mask, np.nan, QV)
        H_m = np.where(fill_mask, np.nan, H_m)

    # Drop NaN levels (e.g., very high levels above the model lid for
    # some early streams) — keep only finite layers.
    valid = np.isfinite(T_K) & np.isfinite(QV) & np.isfinite(H_m) & np.isfinite(lev_hpa)
    if valid.sum() < 5:
        raise ProfileFetchError(
            f"MERRA-2 returned too few valid levels ({valid.sum()}).  "
            f"Possibly the file is corrupted or the grid point is masked."
        )
    T_K = T_K[valid]
    QV = QV[valid]
    H_m = H_m[valid]
    lev_hpa = lev_hpa[valid]

    # Specific humidity → ppmv VMR (+ RH for the diagnostic slot).
    h2o_ppmv, RH = _specific_humidity_to_ppmv_rh(QV, lev_hpa, T_K)

    # Sort bottom-up (lowest geopotential height first).
    z_km = H_m / 1000.0
    order = np.argsort(z_km)
    z_km = z_km[order]
    lev_hpa = lev_hpa[order]
    T_K = T_K[order]
    h2o_ppmv = h2o_ppmv[order]
    RH = RH[order]

    # interpolate -> (lat, lon) of the site; nearest -> the resolved node.
    grid_lat, grid_lon = lat_g, lon_g

    prof = AtmosphereProfile(
        backend="merra2_gesdisc",
        fetched_utc=_now_iso(),
        source_url=url,
        obs_lat_deg=lat,
        obs_lon_deg=lon,
        obs_alt_km=obs_alt_km,
        obs_datetime_utc=dt_utc.isoformat(),
        grid_lat_deg=grid_lat,
        grid_lon_deg=grid_lon,
        grid_datetime_utc=cycle_dt.isoformat(),
        height_km=z_km,
        pressure_hpa=lev_hpa,
        temperature_K=T_K,
        h2o_vmr_ppmv=h2o_ppmv,
        relative_humidity_pct=RH,
        geopotential_height_km=z_km.copy(),
        notes=(
            f"MERRA-2 M2I3NPASM ({stream}), 0.5°×0.625°, "
            f"42 P-levels; QV→ppmv via M_dry/M_H2O; "
            f"{'bilinear-interpolated to site' if interpolate else 'nearest grid point'}."
        ),
    )
    _populate_derived(prof)
    return prof


def _parse_opendap_csv(text: str) -> dict:
    """Parse an OPeNDAP DAP4 CSV (or the older ``.ascii``) response into a dict of
    arrays.

    Format (MERRA-2 M2I3NPASM; the last dimension runs along each row)::

        Dataset: {filename}
        /time, 540
        /lat, 32.5, 33
        /lev, 1000, 975, 950, 925, ...
        /lon, -110, -109.375
        /T[0][0][0], 1e+15, 1e+15
        /T[0][0][1], 1e+15, 1e+15
        ...

    Each line starts with a variable name (``/`` optional), with or without
    bracket indices, followed by comma-separated values; the values of a
    variable are concatenated in the order of the lines.
    """
    out: dict[str, list[float]] = {}
    var_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Dataset:"):
            continue
        # Split off the first comma; everything before is the lhs.
        if "," not in line:
            continue
        lhs, _, rhs = line.partition(",")
        lhs = lhs.strip().lstrip("/")
        m = var_re.match(lhs)
        if not m:
            continue
        varname = m.group(1)
        # Data row (lhs has bracket indices)  OR  coord row (no brackets).
        # Both cases: rhs is a list of numeric tokens.
        toks = _split_floats(rhs)
        if not toks:
            continue
        out.setdefault(varname, []).extend(toks)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _split_floats(s: str):
    """All numbers in *s* (comma or space separated; other tokens skipped)."""
    out = []
    for tok in s.replace(",", " ").split():
        with contextlib.suppress(ValueError):
            out.append(float(tok))
    return out
