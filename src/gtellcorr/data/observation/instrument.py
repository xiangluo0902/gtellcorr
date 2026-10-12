"""Instrument definitions and the instrument registry.

An instrument is described by data, not code: every built-in spectrograph is a
TOML file in ``instruments/`` holding an :class:`InstrumentSpec`. Third-party
packages add instruments through the ``gtellcorr.instruments`` entry-point group
(the entry point loads an :class:`InstrumentSpec`).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Literal

from gtellcorr.config.base import ConfigGroup, Windows, check_windows
from gtellcorr.data.observation.frames import FrameSpec
from gtellcorr.data.observation.header import HeaderSchema
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.registry import Registry

__all__ = [
    "InstrumentSpec",
    "ResolutionSpec",
    "SpectralSpec",
    "identify",
    "instruments",
    "load_instrument_file",
    "read_resolution",
]


@dataclass(frozen=True, kw_only=True)
class ResolutionSpec(ConfigGroup):
    """Resolving power: a default, optionally mapped from a header card."""

    default: float
    """R used when the header carries no resolution information."""
    keys: tuple[str, ...] = ()
    """Header cards read for R (first present wins); empty = always use ``default``."""
    map: Mapping[str, float] = field(default_factory=dict)
    """Card value -> R (e.g. PEPSI fibre diameter ``"200"`` -> 130000); empty = the card
    holds R itself."""
    fit_tol: float | None = None
    """Relative half-width of the R fit bounds for this instrument, used when
    ``lsf.resolution_tol`` is not set; ``None`` = 0.08."""

    def validate(self) -> None:
        """Positive resolving powers; ``fit_tol`` in (0, 1)."""
        if self.default <= 0:
            self.fail("default", f"must be > 0, got {self.default}")
        if self.fit_tol is not None and not 0 < self.fit_tol < 1:
            self.fail("fit_tol", f"must be in (0, 1), got {self.fit_tol}")
        for key, value in self.map.items():
            if value <= 0:
                self.fail(f"map.{key}", f"must be > 0, got {value}")


@dataclass(frozen=True, kw_only=True)
class SpectralSpec(ConfigGroup):
    """How to read the spectra: reader, HDUs or columns, units and frame."""

    reader: str
    """Name of a registered spectrum reader (``echelle_hdu``, ``pepsi_dxt``)."""
    frame: FrameSpec
    """Frame of the delivered wavelengths."""
    wave: str | None = None
    """HDU with wavelengths (echelle readers)."""
    flux: str | None = None
    """HDU with fluxes (echelle readers)."""
    uncertainty: str | None = None
    """HDU with uncertainties (echelle readers)."""
    blaze: str | None = None
    """HDU to divide out; ``None`` = flux already de-blazed."""
    columns: Mapping[str, str] = field(default_factory=dict)
    """Table columns (table readers): ``wave``, ``flux``, ``variance``, ``mask``."""
    uncertainty_kind: Literal["var", "err"] = "var"
    """Uncertainties given as variances (``var``) or standard deviations (``err``)."""
    wave_unit: Literal["A", "nm"] = "A"
    """Unit of the delivered wavelengths."""
    airmass_card: str = "AIRMASS"
    """Primary-header card read by the spectrum reader for the airmass."""
    wavelength_range_nm: tuple[float, float] | None = None
    """Approximate spectral coverage [nm]."""
    order_windows_nm: Windows | None = None
    """For merged 1-D spectra: wavelength windows that define 'orders'. ``None`` = the
    whole spectrum is order 0."""

    def validate(self) -> None:
        """Coverage and order windows must be increasing."""
        if self.wavelength_range_nm is not None:
            lo, hi = self.wavelength_range_nm
            if not 0 < lo < hi:
                self.fail("wavelength_range_nm", f"must satisfy 0 < lo < hi, got ({lo}, {hi})")
        if self.order_windows_nm is not None:
            check_windows(self, "order_windows_nm", self.order_windows_nm)


@dataclass(frozen=True, kw_only=True)
class InstrumentSpec(ConfigGroup):
    """Everything instrument-specific; one built-in TOML file per instrument."""

    name: str
    """Registered name, ``"telescope/instrument"`` (e.g. ``"WIYN/NEID"``)."""
    aliases: tuple[str, ...] = ()
    """Other accepted names, e.g. the bare instrument name."""
    identify: Mapping[str, str] = field(default_factory=dict)
    """Primary-header cards and values that identify the instrument (case-insensitive)."""
    site_token: str
    """Short site tag used in output directory names, e.g. ``KPNO``."""
    default_site: str | None = None
    """Site registry name used for coordinates missing from the headers."""
    channels: Mapping[str, str]
    """Channel (fibre / arm) name -> file-name glob of its reduced spectra."""
    default_channel: str | None = None
    """Channel used when none is requested; required if there are several channels."""
    science_filter: tuple[str, str] | None = None
    """(card, value): keep only frames whose card equals value (case-insensitive)."""
    mid_exposure: bool = True
    """Epoch = start + EXPTIME/2."""
    orders: tuple[int, ...]
    """All order indices of a spectrum file."""
    resolution: ResolutionSpec
    """Resolving power: default and how to read it from the header."""
    header: HeaderSchema
    """Where the header holds the site, epoch, airmass and on-site weather."""
    spectral: SpectralSpec
    """How to read the spectra."""
    anchor_orders: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    """Default per-molecule fit orders (molecule -> orders)."""

    def validate(self) -> None:
        """Name form, channels, orders and anchor orders must be consistent."""
        if "/" not in self.name:
            self.fail("name", f"must have the form 'telescope/instrument', got {self.name!r}")
        if not self.channels:
            self.fail("channels", "at least one channel is required")
        if self.default_channel is not None and self.default_channel not in self.channels:
            self.fail("default_channel", f"{self.default_channel!r} is not one of the channels")
        if len(self.channels) > 1 and self.default_channel is None:
            self.fail("default_channel", "required when there are several channels")
        if not self.orders:
            self.fail("orders", "at least one order is required")
        for molecule, orders in self.anchor_orders.items():
            unknown = sorted(set(orders) - set(self.orders))
            if unknown:
                self.fail(
                    f"anchor_orders.{molecule}", f"orders {unknown} are not instrument orders"
                )

    def channel_glob(self, channel: str | None) -> tuple[str, str]:
        """``(channel, glob)`` for *channel* (``None`` = the default / only channel)."""
        if channel is None:
            channel = self.default_channel or next(iter(self.channels))
        if channel not in self.channels:
            raise DataError(
                f"{self.name}: unknown channel {channel!r}; channels are {', '.join(self.channels)}"
            )
        return channel, self.channels[channel]


instruments: Registry[InstrumentSpec] = Registry(
    "instrument", entry_point_group="gtellcorr.instruments"
)


def load_instrument_file(path: str | os.PathLike[str]) -> InstrumentSpec:
    """Read an :class:`InstrumentSpec` from a TOML file.

    The file holds ``schema_version`` and one ``[instrument]`` table, and shares the
    schema-migration chain of the configuration files.
    """
    from gtellcorr.config.toml_format import loads

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read instrument file {p}: {exc}") from exc
    doc = loads(text, source=str(p))
    if set(doc) != {"instrument"}:
        raise ConfigError(f"{p}: expected exactly one [instrument] table")
    try:
        return InstrumentSpec.from_dict(doc["instrument"])
    except ConfigError as exc:
        raise ConfigError(f"{p}: instrument.{exc}") from None


def _load_builtin_instruments() -> None:
    """Register every TOML file of ``instruments/``."""
    folder = files("gtellcorr.data.observation").joinpath("instruments")
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".toml"):
            with as_file(entry) as path:
                spec = load_instrument_file(path)
            instruments.register(spec.name, spec, aliases=spec.aliases)


_load_builtin_instruments()


# ---------------------------------------------------------------------------
# Header-driven helpers
# ---------------------------------------------------------------------------
def _matches(spec: InstrumentSpec, header: Any) -> bool:
    """Whether every ``identify`` card of *spec* has its value in *header*."""
    if not spec.identify:
        return False
    for card, want in spec.identify.items():
        value = header.get(card) if hasattr(header, "get") else None
        if value is None or str(value).strip().casefold() != str(want).strip().casefold():
            return False
    return True


def identify(header: Any) -> InstrumentSpec:
    """The registered instrument whose ``identify`` rules match *header*.

    Raises :class:`~gtellcorr.errors.DataError` when no instrument or several match.
    """
    matches = [
        instruments.get(n) for n in instruments.names() if _matches(instruments.get(n), header)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DataError(
            "no registered instrument matches this header; pass instrument=... explicitly "
            f"(registered: {', '.join(instruments.names())})"
        )
    raise DataError("several instruments match this header: " + ", ".join(m.name for m in matches))


def read_resolution(header: Any, spec: InstrumentSpec) -> float:
    """Resolving power of one exposure.

    Without header cards the instrument default is returned. If cards are declared,
    one must be present and (if a map is given) its value must be in the map —
    a missing or unknown value is an error, never a silent default.
    """
    res = spec.resolution
    if not res.keys:
        return float(res.default)
    key = next((k for k in res.keys if k in header), None)
    if key is None:
        raise DataError(f"{spec.name}: none of the resolution cards {list(res.keys)} in header")
    raw = header[key]
    if not res.map:
        return float(raw)
    candidates = [str(raw), str(raw).strip()]
    with contextlib.suppress(TypeError, ValueError):
        candidates.append(str(int(raw)))
    for candidate in candidates:
        if candidate in res.map:
            return float(res.map[candidate])
    raise DataError(
        f"{spec.name}: header {key}={raw!r} is not in the resolution map {dict(res.map)}"
    )
