"""``Observation`` and ``Exposure``: which spectra to process, and their metadata.

Exposures are gathered in five steps:

1. glob the channel's file pattern in the directory;
2. keep science frames (``science_filter``; a frame without the card passes);
3. keep files whose name contains an ``include`` token (if any), then drop files
   whose name contains an ``exclude`` token;
4. read site, mid-exposure epoch, airmass and on-site weather from each header;
   an exposure without a usable epoch is an error;
5. an airmass that is missing, non-finite or < 1 is replaced by 1.5 and flagged
   (``airmass_source = "fallback_constant"``).

Site priority: explicit ``site`` > header > ``fallback_site`` > instrument default site.
"""

from __future__ import annotations

import glob
import math
import os
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from astropy.io import fits

from gtellcorr._logging import get_logger
from gtellcorr.data.observation.config import DataConfig, SiteCoords
from gtellcorr.data.observation.frames import VacuumAirFormula
from gtellcorr.data.observation.header import OnSiteWeather, read_on_site_weather, read_site_time
from gtellcorr.data.observation.instrument import (
    InstrumentSpec,
    identify,
    instruments,
    read_resolution,
)
from gtellcorr.data.observation.readers import readers
from gtellcorr.data.observation.readers._common import FALLBACK_AIRMASS
from gtellcorr.data.observation.site import Site, resolve_site
from gtellcorr.data.observation.spectrum import OrderSpectrum, RawOrder
from gtellcorr.data.observation.target import Star, Transit
from gtellcorr.errors import DataError

__all__ = ["Exposure", "Observation", "read_token_file"]

log = get_logger(__name__)


@dataclass(frozen=True)
class Exposure:
    """One science exposure and the metadata read from its primary header."""

    path: Path
    """The spectrum file."""
    time_utc: datetime
    """Mid-exposure epoch (UTC) when the instrument declares ``mid_exposure``."""
    site: Site
    """Observatory coordinates of the exposure."""
    site_sources: dict[str, str]
    """For lat/lon/alt: ``header``, ``override``, ``fallback`` or ``instrument_default``."""
    airmass: float
    airmass_source: str
    """``fits_header`` or ``fallback_constant``."""
    weather: OnSiteWeather
    """On-site meteorology from the header, where available."""
    resolution: float
    """Resolving power from the header (or the instrument default)."""
    object: str
    """``OBJECT`` card of the header."""

    @property
    def name(self) -> str:
        """File name of the exposure."""
        return self.path.name


def read_token_file(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Exposure tokens, one per line; ``#`` starts a comment."""
    tokens = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                tokens.append(line)
    return tuple(tokens)


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    """Whether *name* contains one of the non-empty *tokens*."""
    return any(tok and tok in name for tok in tokens)


def _open_header(path: Path) -> Any:
    """Primary header of a FITS file (verification warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. PEPSI cards trip FITS verification
        with fits.open(path) as h:
            return h[0].header.copy()


def _is_science(header: Any, spec: InstrumentSpec) -> bool:
    """Whether the frame passes the instrument's science filter."""
    if spec.science_filter is None:
        return True
    card, want = spec.science_filter
    value = str(header.get(card, "")).strip().lower()
    return not value or value == str(want).strip().lower()


def _identify_directory(directory: Path) -> InstrumentSpec:
    """The one instrument whose file patterns and header rules match *directory*."""
    candidates = []
    for name in instruments.names():
        spec = instruments.get(name)
        files = sorted(
            {p for pattern in spec.channels.values() for p in glob.glob(str(directory / pattern))}
        )
        if files:
            candidates.append((spec, Path(files[0])))
    if not candidates:
        raise DataError(f"{directory}: no file matches any registered instrument's file patterns")
    matched = [spec for spec, first in candidates if spec is identify_or_none(_open_header(first))]
    if len(matched) != 1:
        names = ", ".join(s.name for s, _ in candidates)
        raise DataError(
            f"{directory}: cannot identify the instrument (file patterns match {names}); "
            "pass instrument=... explicitly"
        )
    return matched[0]


def identify_or_none(header: Any) -> InstrumentSpec | None:
    """Like :func:`~gtellcorr.data.observation.instrument.identify`, but ``None`` when
    no or several instruments match."""
    try:
        return identify(header)
    except DataError:
        return None


def _exposure(
    path: Path,
    header: Any,
    spec: InstrumentSpec,
    override: Site | None,
    fallback: Site | None,
    fallback_source: str,
) -> Exposure:
    """Build the :class:`Exposure` of one file from its header (see the module notes)."""
    coords = override.coords if override is not None else fallback.coords if fallback else None
    st = read_site_time(header, spec.header, mid_exposure=spec.mid_exposure, fallback=coords)
    if override is not None:
        site = override
        sources = dict.fromkeys(("lat_deg", "lon_deg", "alt_m"), "override")
    else:
        site = Site(lat_deg=st.lat_deg, lon_deg=st.lon_deg, alt_m=st.alt_m)
        sources = {k: (fallback_source if v == "fallback" else v) for k, v in st.sources.items()}
    if st.time_utc is None:
        raise DataError(f"{path.name}: no usable epoch in header")
    airmass = st.airmass
    if airmass is None or not math.isfinite(airmass) or airmass < 1.0:
        airmass, airmass_source = FALLBACK_AIRMASS, "fallback_constant"
    else:
        airmass_source = "fits_header"
    return Exposure(
        path=path,
        time_utc=st.time_utc,
        site=site,
        site_sources=sources,
        airmass=float(airmass),
        airmass_source=airmass_source,
        weather=read_on_site_weather(header, spec.header),
        resolution=read_resolution(header, spec),
        object=str(header.get("OBJECT", "")).strip(),
    )


@dataclass(frozen=True)
class Observation:
    """A set of exposures of one instrument channel (immutable).

    Build it with :meth:`from_directory` or :meth:`from_config`; narrow it with
    :meth:`select`. Pixel data are read on demand with :meth:`read_order`.
    """

    instrument: InstrumentSpec
    """The instrument definition."""
    channel: str
    """The processed channel (fibre / arm)."""
    directory: Path
    """Directory of the spectra."""
    exposures: tuple[Exposure, ...]
    """The selected exposures, sorted by file name."""
    target: Star | None = None
    """The observed star (needed by the pipeline)."""
    transit: Transit | None = None
    """Transit ephemeris, if any."""

    # ------------------------------------------------------------------ building
    @classmethod
    def from_directory(
        cls,
        directory: str | os.PathLike[str],
        *,
        instrument: str | InstrumentSpec | None = None,
        channel: str | None = None,
        site: str | SiteCoords | Site | None = None,
        fallback_site: str | SiteCoords | Site | None = None,
        target: Star | None = None,
        transit: Transit | None = None,
        include: Sequence[str] | None = None,
        include_file: str | os.PathLike[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> Observation:
        """Gather the science exposures of one channel in *directory*.

        ``instrument=None`` identifies the instrument from the files' headers.
        """
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise DataError(f"data directory not found: {root}")
        if isinstance(instrument, InstrumentSpec):
            spec = instrument
        elif instrument is not None:
            spec = instruments.get(instrument)
        else:
            spec = _identify_directory(root)
        channel_name, pattern = spec.channel_glob(channel)

        override = resolve_site(site) if site is not None else None
        if fallback_site is not None:
            fallback, fallback_source = resolve_site(fallback_site), "fallback"
        elif spec.default_site is not None:
            fallback, fallback_source = resolve_site(spec.default_site), "instrument_default"
        else:
            fallback, fallback_source = None, "fallback"

        tokens = tuple(include) if include is not None else None
        if include_file is not None:
            tokens = (*(tokens or ()), *read_token_file(include_file))

        exposures = []
        for path in sorted({Path(p) for p in glob.glob(str(root / pattern))}):
            if tokens is not None and not _contains_any(path.name, tokens):
                continue
            if _contains_any(path.name, exclude):
                continue
            header = _open_header(path)
            if not _is_science(header, spec):
                continue
            exposures.append(_exposure(path, header, spec, override, fallback, fallback_source))
        if not exposures:
            raise DataError(f"{root}: no {spec.name} ({channel_name}) exposures selected")
        n_fallback = sum(e.airmass_source == "fallback_constant" for e in exposures)
        if n_fallback:
            log.warning(
                "%d exposure(s) without a usable header airmass; using %.1f",
                n_fallback,
                FALLBACK_AIRMASS,
            )
        return cls(
            instrument=spec,
            channel=channel_name,
            directory=root,
            exposures=tuple(exposures),
            target=target,
            transit=transit,
        )

    @classmethod
    def from_config(cls, config: DataConfig) -> Observation:
        """Build from the ``[data]`` table of a run configuration."""
        return cls.from_directory(
            config.directory,
            instrument=config.instrument,
            channel=config.channel,
            site=config.site,
            fallback_site=config.fallback_site,
            target=config.target,
            transit=config.transit,
            include=config.include,
            include_file=config.include_file,
            exclude=config.exclude,
        )

    def select(
        self, include: Sequence[str] | None = None, exclude: Sequence[str] = ()
    ) -> Observation:
        """A new observation keeping exposures whose names contain an ``include``
        token (if given) and none of the ``exclude`` tokens."""
        kept = tuple(
            e
            for e in self.exposures
            if (include is None or _contains_any(e.name, include))
            and not _contains_any(e.name, exclude)
        )
        if not kept:
            raise DataError("selection leaves no exposures")
        return replace(self, exposures=kept)

    # ------------------------------------------------------------------ reading
    def _exposure(self, exposure: Exposure | int | str) -> Exposure:
        """An exposure given as itself, its index, or a unique part of its file name."""
        if isinstance(exposure, Exposure):
            return exposure
        if isinstance(exposure, int):
            return self.exposures[exposure]
        matches = [e for e in self.exposures if exposure in e.name]
        if len(matches) != 1:
            raise DataError(f"{exposure!r} matches {len(matches)} exposures; be more specific")
        return matches[0]

    def read_order(
        self,
        exposure: Exposure | int | str,
        order: int,
        *,
        formula: VacuumAirFormula = "espresso",
    ) -> OrderSpectrum:
        """Fit-capable pixels of one order, in the air + topocentric frame."""
        reader = readers.get(self.instrument.spectral.reader)
        return reader.load_order(
            self._exposure(exposure).path, order, self.instrument.spectral, formula
        )

    def read_order_raw(
        self,
        exposure: Exposure | int | str,
        order: int,
        *,
        formula: VacuumAirFormula = "espresso",
    ) -> RawOrder:
        """All pixels of one order (for output placeholders / diagnostics)."""
        reader = readers.get(self.instrument.spectral.reader)
        return reader.load_order_raw(
            self._exposure(exposure).path, order, self.instrument.spectral, formula
        )

    def __len__(self) -> int:
        return len(self.exposures)

    def __repr__(self) -> str:
        return (
            f"Observation({self.instrument.name}, channel={self.channel!r}, "
            f"{len(self.exposures)} exposures, directory={str(self.directory)!r})"
        )
