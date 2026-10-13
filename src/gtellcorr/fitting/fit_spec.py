"""Per-order fit specification: which molecules and nuisance parameters are free.

The decisions are made from an explicit :class:`~gtellcorr.config.FitConfig` and
:class:`~gtellcorr.data.observation.instrument.InstrumentSpec`; nothing is read
from global state.

Molecule selection per order:

* a molecule with ``fit = true`` is fitted in its *anchor orders* if it has any,
  otherwise in every order overlapping its ``bands_nm``;
* ``molecule_selection.add`` forces an engine molecule in: in its anchor orders
  if it has any, otherwise in **every** requested order (with a warning);
* ``molecule_selection.exclude`` removes a molecule with the highest priority.

Anchor orders of a molecule: ``Molecule.anchor_orders`` if set, otherwise the
instrument definition's ``anchor_orders``; ``()`` means "no assigned orders".
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gtellcorr.errors import DataError
from gtellcorr.telluric.molecules import ENGINE_MOLECULES

if TYPE_CHECKING:
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.instrument import InstrumentSpec

__all__ = [
    "DEFAULT_RESOLUTION_TOL",
    "FitSpec",
    "anchor_orders",
    "build_fit_spec",
    "configured_fit_molecules",
    "default_resolution_tol",
    "select_fit_molecules",
    "select_fit_orders_for_molecules",
]

WaveRange = tuple[float, float]

#: R-bound half-width when neither the fit configuration nor the instrument sets one.
DEFAULT_RESOLUTION_TOL = 0.08


@dataclass(frozen=True)
class FitSpec:
    """Free parameters and fixed settings of one order's fit
    (:func:`build_fit_spec`); the engine reads nothing else."""

    molecules: tuple[str, ...]
    """Fitted molecules, in LBLRTM ID order (the relcol order in theta)."""
    relcol_init: Mapping[str, float]
    """Initial relcol per fitted molecule."""
    relcol_bounds: Mapping[str, tuple[float, float]]
    """Fit bounds of relcol per fitted molecule."""
    fit_resolution: bool
    """Whether R is fitted."""
    resolution_init: float
    """Initial (or fixed) resolving power."""
    resolution_bounds: tuple[float, float]
    """Fit bounds of R; they also set the fixed convolution grid."""
    lsf_kind: str
    """``gauss``, ``sgauss`` or ``voigt``."""
    lsf_shape: float | None
    """Fixed or initial LSF shape (``None`` for ``gauss``)."""
    lsf_shape_bounds: tuple[float, float] | None
    """Fit bounds of the shape (``None`` for ``gauss``)."""
    fit_lsf_shape: bool
    """Whether the LSF shape is fitted."""
    continuum_order: int
    """Polynomial degree of the projected continuum."""
    continuum_sigma_clip: tuple[float, float]
    """Lower and upper clipping thresholds of the continuum fit."""
    continuum_niter: int
    """Clipping iterations of the continuum fit."""
    order: int | None = None
    """Echelle order (labelling only)."""
    fit_wave_shift: bool = False
    """Fit the wavelength-solution correction (:class:`~gtellcorr.fitting.Wavelength`)."""
    wave_shift_degree: int = 0
    """Chebyshev degree of the correction velocity across the order."""
    wave_shift_scope: str = "order"
    """``order`` (per-order correction) or ``exposure`` (shared in joint fits)."""
    wave_shift_bounds: tuple[float, float] = (-1.0, 1.0)
    """Bounds of each Chebyshev coefficient [km/s]."""
    wave_shift: tuple[float, ...] | None = None
    """Fixed correction coefficients [km/s] when not fitted (``None`` = no correction)."""

    def lsf_grid(self) -> tuple[float, float]:
        """``(grid_resolution, support_resolution)`` of the fixed-grid convolution.

        While R is fitted: the upper and lower R bounds (or the initial R if outside),
        so the grid stays fixed during the fit; otherwise the fixed R.
        """
        r0 = float(self.resolution_init)
        if not self.fit_resolution:
            return r0, r0
        lo, hi = self.resolution_bounds
        return max(float(hi), r0), min(float(lo), r0)


def _overlaps(bands_nm: Sequence[tuple[float, float]], lo: float, hi: float) -> bool:
    """Whether any band overlaps ``(lo, hi)`` (open intervals)."""
    return any(b_lo < hi and b_hi > lo for b_lo, b_hi in bands_nm)


def anchor_orders(config: FitConfig, instrument: InstrumentSpec) -> dict[str, tuple[int, ...]]:
    """Fit orders assigned to each molecule (configuration first, then the instrument)."""
    pins: dict[str, tuple[int, ...]] = {}
    for name in ENGINE_MOLECULES:
        own = config.molecules[name].anchor_orders
        orders = own if own is not None else instrument.anchor_orders.get(name, ())
        if orders:
            pins[name] = tuple(int(o) for o in orders)
    return pins


def _warn_add_without_anchors(instrument: InstrumentSpec, molecule: str) -> None:
    """Warn that an added molecule without anchor orders is fitted in every order."""
    warnings.warn(
        f"molecule_selection.add includes {molecule!r}, but {instrument.name} has no "
        "anchor_orders for it. It will be fitted in all requested orders, which may be slow.",
        RuntimeWarning,
        stacklevel=3,
    )


def select_fit_molecules(
    wave_range_nm: WaveRange,
    *,
    config: FitConfig,
    instrument: InstrumentSpec,
    order: int | None = None,
) -> tuple[str, ...]:
    """Engine molecules fitted in the order covering *wave_range_nm*."""
    lo, hi = float(wave_range_nm[0]), float(wave_range_nm[1])
    pins = anchor_orders(config, instrument)
    add = config.molecule_selection.add
    exclude = set(config.molecule_selection.exclude)
    selected = set()
    for m in ENGINE_MOLECULES:
        if m in exclude or not config.molecules[m].fit:
            continue
        pinned = pins.get(m)
        if pinned and order is not None:
            if int(order) in pinned:
                selected.add(m)
        elif _overlaps(config.molecules[m].bands_nm, lo, hi):
            selected.add(m)
    for m in add:
        if m in exclude:
            continue
        pinned = pins.get(m)
        if pinned and order is not None:
            if int(order) in pinned:
                selected.add(m)
        elif pinned and order is None:
            selected.add(m)
        else:
            _warn_add_without_anchors(instrument, m)
            selected.add(m)
    return tuple(m for m in ENGINE_MOLECULES if m in selected and m not in exclude)


def configured_fit_molecules(config: FitConfig) -> tuple[str, ...]:
    """The configured molecule set before per-order filtering."""
    selected = {m for m in ENGINE_MOLECULES if config.molecules[m].fit}
    selected.update(config.molecule_selection.add)
    selected.difference_update(config.molecule_selection.exclude)
    return tuple(m for m in ENGINE_MOLECULES if m in selected)


def select_fit_orders_for_molecules(
    order_wave_ranges_nm: Mapping[int, WaveRange],
    *,
    config: FitConfig,
    instrument: InstrumentSpec,
    molecules: Sequence[str] | None = None,
) -> list[int]:
    """Joint-fit orders for the configured molecules (anchor orders, else bands)."""
    pins = anchor_orders(config, instrument)
    mols = [str(m).lower() for m in (molecules or configured_fit_molecules(config))]
    selected: set[int] = set()
    for molecule in mols:
        pinned = pins.get(molecule)
        if pinned:
            available = [o for o in pinned if o in order_wave_ranges_nm]
            if not available:
                raise DataError(
                    f"{instrument.name}: fit molecule {molecule!r} is assigned to orders "
                    f"{list(pinned)}, but none are available in the data. Available orders: "
                    f"{sorted(int(o) for o in order_wave_ranges_nm)}"
                )
            selected.update(available)
            continue
        bands = config.molecules[molecule].bands_nm
        overlap = [
            int(o)
            for o, (lo, hi) in order_wave_ranges_nm.items()
            if _overlaps(bands, float(lo), float(hi))
        ]
        if not overlap:
            raise DataError(
                f"{instrument.name}: fit molecule {molecule!r} has no anchor_orders and no "
                "overlap with the available order wavelength ranges"
            )
        warnings.warn(
            f"{instrument.name}: fit molecule {molecule!r} has no anchor_orders; selecting "
            "joint-fit orders by bands_nm overlap.",
            RuntimeWarning,
            stacklevel=2,
        )
        selected.update(overlap)
    return sorted(selected)


def build_fit_spec(
    wave_range_nm: WaveRange,
    *,
    config: FitConfig,
    instrument: InstrumentSpec,
    order: int | None = None,
    resolution: float | None = None,
) -> FitSpec:
    """The fit spec of one order.

    The initial (or fixed) R is ``lsf.resolution`` if configured, else ``resolution``,
    the exposure's R (from its header,
    :func:`~gtellcorr.data.observation.instrument.read_resolution`), else the
    instrument default. R bounds are ``lsf.resolution_bounds`` if given, else
    ``R0 * (1 -/+ tol)`` with ``tol`` = ``lsf.resolution_tol``, else the instrument's
    ``resolution.fit_tol``, else 0.08.
    """
    lsf = config.lsf
    mols = select_fit_molecules(wave_range_nm, config=config, instrument=instrument, order=order)
    kind = lsf.kind
    shape = lsf.shape.get(kind)
    shape_bounds = lsf.shape_bounds.get(kind)
    if lsf.resolution is not None:
        resolution = lsf.resolution
    r0 = float(resolution if resolution is not None else instrument.resolution.default)
    if lsf.resolution_bounds is not None:
        r_bounds = tuple(lsf.resolution_bounds)
    else:
        tol = default_resolution_tol(config, instrument)
        r_bounds = (r0 * (1.0 - tol), r0 * (1.0 + tol))
    return FitSpec(
        molecules=mols,
        relcol_init={m: float(config.molecules[m].relcol) for m in mols},
        relcol_bounds={m: tuple(config.molecules[m].bounds) for m in mols},  # type: ignore[misc]
        fit_resolution=bool(lsf.fit_resolution),
        resolution_init=r0,
        resolution_bounds=(float(r_bounds[0]), float(r_bounds[1])),
        lsf_kind=kind,
        lsf_shape=shape,
        lsf_shape_bounds=tuple(shape_bounds) if shape_bounds is not None else None,  # type: ignore[arg-type]
        fit_lsf_shape=bool(lsf.fit_shape) and shape is not None,
        continuum_order=int(config.continuum.order),
        continuum_sigma_clip=tuple(config.continuum.sigma_clip),  # type: ignore[arg-type]
        continuum_niter=int(config.continuum.niter),
        order=order,
        fit_wave_shift=bool(config.wavelength.fit),
        wave_shift_degree=int(config.wavelength.order),
        wave_shift_scope=str(config.wavelength.scope),
        wave_shift_bounds=(
            float(config.wavelength.bounds_kms[0]),
            float(config.wavelength.bounds_kms[1]),
        ),
    )


def default_resolution_tol(config: FitConfig, instrument: InstrumentSpec) -> float:
    """``lsf.resolution_tol``, else the instrument's ``resolution.fit_tol``, else 0.08."""
    if config.lsf.resolution_tol is not None:
        return float(config.lsf.resolution_tol)
    if instrument.resolution.fit_tol is not None:
        return float(instrument.resolution.fit_tol)
    return DEFAULT_RESOLUTION_TOL
