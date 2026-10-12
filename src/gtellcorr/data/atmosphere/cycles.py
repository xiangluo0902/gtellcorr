"""Analysis-cycle time rounding helpers (stdlib-only leaf module)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _nearest_6h_cycle(dt_utc: datetime) -> datetime:
    """Round to nearest 6-hourly synoptic cycle (00, 06, 12, 18 UTC)."""
    hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    nearest_h = round(hours / 6.0) * 6
    base = datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=UTC)
    return base + timedelta(hours=nearest_h)


def _nearest_gfs_cycle(dt_utc: datetime) -> datetime:
    """Round to nearest 6-hourly GFS analysis cycle (00, 06, 12, 18 UTC).
    GFS analyses sit on the 6-hourly synoptic grid, so this delegates to
    _nearest_6h_cycle (shared with the NCEP-R1 backend)."""
    return _nearest_6h_cycle(dt_utc)


def _nearest_merra2_cycle(dt_utc: datetime) -> tuple[datetime, int]:
    """Round to nearest 3-hourly MERRA-2 cycle.  Returns (cycle_dt, idx)
    where idx is 0..7 along the ``time`` dimension of the daily file.

    Mirrors the GFS/NCEP helpers: the rounded offset is added to that
    day's midnight via ``timedelta`` so a time that rounds up to 24:00
    rolls cleanly into the next day's 00:00 (idx 0).  The earlier
    ``int(round(hours / 3.0)) % 8`` wrapped that carry back onto the
    same day, returning a cycle up to ~23 h away for any exposure
    between 22:30 and 24:00 UTC; ``idx`` is now read back from the
    (possibly day-rolled) cycle so it always indexes the correct file.
    """
    hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    base = datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=UTC)
    cycle_dt = base + timedelta(hours=3.0 * round(hours / 3.0))
    idx = cycle_dt.hour // 3  # 0..7, from the resolved cycle
    return cycle_dt, idx


def _nearest_3h_cycle(dt_utc: datetime) -> datetime:
    """Round to nearest 3-hourly cycle (00, 03, ..., 21 UTC).
    Uses the same timedelta-carry style as _nearest_merra2_cycle's cycle
    datetime — midnight + timedelta(hours=3*round(h/3))."""
    hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    base = datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=UTC)
    return base + timedelta(hours=3.0 * round(hours / 3.0))


def backend_cycle(backend: str, dt_utc: datetime) -> datetime:
    """Return the analysis-cycle UTC datetime that ``backend`` snaps
    ``dt_utc`` to -- i.e. the temporal resolution of that data source.

    Mirrors the rounding each ``fetch_*`` does internally, so callers can
    key the raw cache by the cycle (not the exposure time): exposures that
    share a cycle then reuse a single download.

        era5_cdsapi              -> nearest whole hour
        merra2_gesdisc           -> nearest 3-hourly cycle (00, 03, ... 21)
        ncep_r1_psl / gfs_thredds-> nearest 6-hourly cycle (00, 06, 12, 18)
        manual / unknown         -> dt_utc unchanged (no server cycle)
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    dt_utc = dt_utc.astimezone(UTC)
    if backend == "era5_cdsapi":
        c = dt_utc.replace(minute=0, second=0, microsecond=0)
        return c + timedelta(hours=1) if dt_utc.minute >= 30 else c
    if backend == "merra2_gesdisc":
        return _nearest_merra2_cycle(dt_utc)[0]
    if backend == "eso_gdas":
        return _nearest_3h_cycle(dt_utc)
    if backend in ("ncep_r1_psl", "gfs_thredds"):
        return _nearest_6h_cycle(dt_utc)
    return dt_utc  # manual / unknown: no server-side analysis cycle
