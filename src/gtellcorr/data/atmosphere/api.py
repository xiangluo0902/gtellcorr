"""Public entry points of the atmosphere layer.

Every function takes the scientific configuration (:class:`AtmosphereConfig`) and
the machine-local :class:`~gtellcorr.settings.Settings` explicitly, builds an
:class:`~gtellcorr.data.atmosphere._context.AtmosphereContext` for the duration
of the call, and delegates to the profile modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from gtellcorr.data.atmosphere import _context
from gtellcorr.data.atmosphere.config import AtmosphereConfig
from gtellcorr.data.atmosphere.reference import (
    CELESTRAK_URL,
    ReferenceFile,
    latest_celestrak,
)
from gtellcorr.data.observation.config import SiteCoords
from gtellcorr.data.observation.header import OnSiteWeather
from gtellcorr.data.observation.instrument import InstrumentSpec
from gtellcorr.data.observation.observation import Exposure
from gtellcorr.data.observation.site import Site, resolve_site
from gtellcorr.errors import ConfigError, ResourceError
from gtellcorr.settings import Settings

__all__ = [
    "atmosphere_context",
    "full_profile_for_exposure",
    "get_full_profile",
    "get_meteo",
    "reference_atmosphere",
]

SiteLike = Site | SiteCoords | str


def atmosphere_context(
    config: AtmosphereConfig | None = None,
    settings: Settings | None = None,
    *,
    celestrak: ReferenceFile | None = None,
    need_celestrak: bool | None = None,
) -> _context.AtmosphereContext:
    """The context the atmosphere modules read for one call.

    ``celestrak`` defaults to the recorded table in ``<resource_root>/atmosphere``.
    A missing table is an error if ``need_celestrak`` (default: only when
    ``config.upper_source == "msis00"``).
    """
    from gtellcorr.telluric.resources import load_setup

    cfg = config or AtmosphereConfig()
    if cfg.lower_backend == "manual":
        manual = Path(str(cfg.manual_profile_file)).expanduser()
        if not manual.is_file():
            raise ConfigError(f"atmosphere.manual_profile_file not found: {manual}")
    s = settings or Settings.load()
    folder = s.resource_root / "atmosphere"
    mipas = load_setup()["atmosphere_references"]["rfm_mipas_midlatitude_night"]
    if celestrak is None:
        try:
            celestrak = latest_celestrak(folder)
        except ResourceError:
            required = cfg.upper_source == "msis00" if need_celestrak is None else need_celestrak
            if required:
                raise
            celestrak = None
    assert s.atmosphere_cache is not None  # resolved by Settings
    return _context.AtmosphereContext(
        config=cfg,
        profile_dir=s.atmosphere_cache,
        mipas_path=folder / mipas["file_name"],
        mipas_sha256=mipas["sha256"],
        mipas_url=mipas["source_url"],
        celestrak_path=celestrak.path if celestrak else None,
        celestrak_sha256=celestrak.sha256 if celestrak else None,
        celestrak_url=CELESTRAK_URL,
        eso_gdas_dir=folder / "eso_gdas",
        secrets_env=s.credentials_file,
    )


def _observatory(site: SiteLike) -> tuple[float, float, float]:
    """``(lat_deg, lon_deg, alt_km)`` of a site."""
    resolved = resolve_site(site)
    return (resolved.lat_deg, resolved.lon_deg, resolved.alt_m / 1000.0)


def _surface(onsite: OnSiteWeather | None) -> dict[str, float | None]:
    """On-site weather as the keyword arguments of the profile functions."""
    if onsite is None:
        return {"surface_T_K": None, "surface_RH_pct": None, "surface_P_hpa": None}
    return {
        "surface_T_K": onsite.T_K,
        "surface_RH_pct": onsite.RH_pct,
        "surface_P_hpa": onsite.P_hpa,
    }


def get_meteo(
    time_utc: datetime,
    site: SiteLike,
    *,
    config: AtmosphereConfig | None = None,
    settings: Settings | None = None,
    site_label: str | None = None,
    onsite: OnSiteWeather | None = None,
    save: bool = True,
    reuse_cache: bool = True,
    verbose: bool = False,
) -> Any:
    """The lower-atmosphere profile (reanalysis) for one epoch and site.

    No space-weather table is needed: it only serves the upper atmosphere.
    """
    from gtellcorr.data.atmosphere import meteo

    ctx = atmosphere_context(config, settings, need_celestrak=False)
    with _context.using(ctx):
        return meteo.get_meteo(  # type: ignore[no-untyped-call]  # unannotated module
            dt_utc=time_utc,
            observatory=_observatory(site),
            observatory_label=site_label,
            backend=ctx.config.lower_backend,
            merge_onsite=ctx.config.merge_onsite,
            save=save,
            reuse_cache=reuse_cache,
            verbose=verbose,
            **_surface(onsite),
        )


def get_full_profile(
    time_utc: datetime,
    site: SiteLike,
    *,
    config: AtmosphereConfig | None = None,
    settings: Settings | None = None,
    site_label: str | None = None,
    onsite: OnSiteWeather | None = None,
    celestrak: ReferenceFile | None = None,
    save: bool = True,
    reuse_cache: bool = True,
    return_meta: bool = False,
    verbose: bool = False,
) -> Any:
    """The complete 0-120 km profile (lower reanalysis + upper climatology).

    Returns the full-profile mapping (and its metadata if ``return_meta``). Results
    are cached in ``Settings.atmosphere_cache`` under a request-derived key.
    """
    from gtellcorr.data.atmosphere import full_profile

    ctx = atmosphere_context(config, settings, celestrak=celestrak)
    with _context.using(ctx):
        return full_profile.get_full_profile(  # type: ignore[no-untyped-call]  # unannotated module
            time_utc,
            observatory=_observatory(site),
            observatory_label=site_label,
            backend=ctx.config.lower_backend,
            upper_source=ctx.config.upper_source,
            merge_onsite=ctx.config.merge_onsite,
            save=save,
            reuse_cache=reuse_cache,
            return_meta=return_meta,
            verbose=verbose,
            **_surface(onsite),
        )


def full_profile_for_exposure(
    exposure: Exposure,
    instrument: InstrumentSpec,
    *,
    config: AtmosphereConfig | None = None,
    settings: Settings | None = None,
    celestrak: ReferenceFile | None = None,
    save: bool = True,
    reuse_cache: bool = True,
    return_meta: bool = False,
    verbose: bool = False,
) -> Any:
    """The full profile of one exposure, as a run builds it: the exposure's site and
    mid-exposure epoch, the instrument's site token as label, and the header on-site
    weather (merged only if ``config.merge_onsite``)."""
    return get_full_profile(
        exposure.time_utc,
        exposure.site,
        config=config,
        settings=settings,
        site_label=instrument.site_token,
        onsite=exposure.weather,
        celestrak=celestrak,
        save=save,
        reuse_cache=reuse_cache,
        return_meta=return_meta,
        verbose=verbose,
    )


def reference_atmosphere(
    config: AtmosphereConfig | None = None, settings: Settings | None = None
) -> Mapping[str, Any]:
    """The RFM MIPAS reference atmosphere (checked against its SHA-256), which supplies the
    eleven fixed molecules."""
    from gtellcorr.data.atmosphere import full_profile

    # only the MIPAS file is read: no CelesTrak table needed
    ctx = atmosphere_context(config, settings, need_celestrak=False)
    with _context.using(ctx):
        base: Mapping[str, Any] = full_profile.extract_base_atmosphere(
            str(ctx.mipas_path), expected_sha256=ctx.mipas_sha256, source_url=ctx.mipas_url
        )
    return base
