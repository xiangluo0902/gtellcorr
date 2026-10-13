"""Parameter groups of the fitting engine: continuum, LSF, wavelength-solution
correction, molecules and order selection.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from gtellcorr.config.base import ConfigGroup, ConfigValue, FrozenDict, Windows, check_windows
from gtellcorr.errors import ConfigError, did_you_mean
from gtellcorr.telluric.molecules import ENGINE_MOLECULES, MOLECULE_NAMES

__all__ = [
    "LSF",
    "Continuum",
    "Molecule",
    "MoleculeSelection",
    "Molecules",
    "OrderSelection",
    "Wavelength",
]


@dataclass(frozen=True, kw_only=True)
class Continuum(ConfigGroup):
    """Telluric-fit continuum, projected out in every residual evaluation: a polynomial
    in wavelength fitted to ``data / model`` (weights ``w * model**2``, which minimises
    the fit's own objective over the continuum) with iterative sigma clipping.
    """

    order: int = 5
    """Polynomial degree of the continuum."""
    sigma_clip: tuple[float, float] = (2.0, 3.0)
    """Lower and upper sigma-clipping thresholds."""
    niter: int = 5
    """Number of clipping iterations."""

    def validate(self) -> None:
        """Non-negative degree, positive thresholds, at least one iteration."""
        if self.order < 0:
            self.fail("order", f"must be >= 0, got {self.order}")
        if min(self.sigma_clip) <= 0:
            self.fail("sigma_clip", f"thresholds must be positive, got {self.sigma_clip}")
        if self.niter < 1:
            self.fail("niter", f"must be >= 1, got {self.niter}")


#: LSF profile kinds.
LSFKind = Literal["gauss", "sgauss", "voigt"]
_SHAPED_KINDS = ("sgauss", "voigt")


@dataclass(frozen=True, kw_only=True)
class LSF(ConfigGroup):
    """Line-spread function: its kind and shape, and the resolving power R.

    The transmission is convolved on a velocity grid fixed by the R bounds, so the
    model is a smooth function of R and the optimiser sees a usable gradient in R.
    The fit parameters are scaled by their initial values, so the ~1e5 resolving
    power does not dominate the convergence test of the ~1 relative columns.
    """

    kind: LSFKind = "gauss"
    """``gauss``; ``sgauss`` (super-Gaussian, shape = exponent p, p=2 is Gaussian);
    ``voigt`` (shape = Lorentzian weight eta in [0, 1))."""
    shape: Mapping[str, float] = field(default_factory=lambda: {"sgauss": 2.0, "voigt": 0.2})
    """Shape value per shaped kind (each kind keeps its own value)."""
    shape_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {"sgauss": (1.2, 4.0), "voigt": (0.0, 0.9)}
    )
    """Fit bounds of the shape, per shaped kind (used when ``fit_shape``)."""
    fit_shape: bool = False
    """Fit the active kind's shape (ignored for ``gauss``)."""
    fit_resolution: bool = True
    """Fit the resolving power R."""
    resolution: float | None = None
    """Initial R, and the fixed R when ``fit_resolution`` is false; overrides the
    exposure's header R. ``None`` = the header R (e.g. PEPSI ``FIBER``), else the
    instrument default. Also the centre of the ``resolution_tol`` bounds."""
    resolution_tol: float | None = None
    """R bounds: R0 * (1 -/+ resolution_tol), R0 from the instrument. ``None`` = the
    instrument's ``resolution.fit_tol`` if it has one, else 0.08."""
    resolution_bounds: tuple[float, float] | None = None
    """Explicit (lo, hi) R bounds overriding ``resolution_tol``."""

    def validate(self) -> None:
        """Known shaped kinds, ordered bounds, R and its tolerance in range."""
        for name in ("shape", "shape_bounds"):
            for key in getattr(self, name):
                if key not in _SHAPED_KINDS:
                    self.fail(
                        f"{name}.{key}",
                        f"unknown LSF kind; shaped kinds are {', '.join(_SHAPED_KINDS)}",
                    )
        for key, (lo, hi) in self.shape_bounds.items():
            if not lo < hi:
                self.fail(f"shape_bounds.{key}", f"must satisfy lo < hi, got ({lo}, {hi})")
        if self.resolution is not None and not self.resolution > 0:
            self.fail("resolution", f"must be > 0, got {self.resolution}")
        if self.resolution_tol is not None and not 0 < self.resolution_tol < 1:
            self.fail("resolution_tol", f"must be in (0, 1), got {self.resolution_tol}")
        if self.resolution_bounds is not None:
            lo, hi = self.resolution_bounds
            if not 0 < lo < hi:
                self.fail("resolution_bounds", f"must satisfy 0 < lo < hi, got ({lo}, {hi})")


#: Scope of the wavelength-solution correction (see :attr:`Wavelength.scope`).
WavelengthScope = Literal["order", "exposure"]


@dataclass(frozen=True, kw_only=True)
class Wavelength(ConfigGroup):
    """Correction of the wavelength solution, fitted on the telluric lines.

    The correction is a velocity ``v(x) = sum_k c_k T_k(u)`` [km/s] (``u`` = ``ln x``
    mapped onto [-1, 1] across the order, i.e. a velocity coordinate): degree 0 corrects
    the zero point, higher degrees the dispersion as well. In the fit it is applied to
    the telluric model, ``T_obs(x) = T_model(x / (1 + v(x) / c))``; the data and the
    stellar model are not moved. The corrected wavelengths of the data pixels are
    written to the results (``WAVE_FIT_CORR``, ``WAVE_INPUT_CORR``).
    """

    fit: bool = False
    """Fit the correction."""
    order: int = 0
    """Chebyshev degree of v(x): 0 = one constant velocity per order."""
    scope: WavelengthScope = "order"
    """``order``: a correction per order (joint fits: order-scope parameters);
    ``exposure``: one correction shared by all orders (``order = 0`` only)."""
    bounds_kms: tuple[float, float] = (-1.0, 1.0)
    """Fit bounds of every Chebyshev coefficient [km/s]."""

    def validate(self) -> None:
        """Non-negative degree, constant for an exposure-wide correction, bounds around 0."""
        if self.order < 0:
            self.fail("order", f"must be >= 0, got {self.order}")
        if self.scope == "exposure" and self.order != 0:
            self.fail("scope", "an exposure-wide correction must be a constant (order = 0)")
        lo, hi = self.bounds_kms
        if not lo < 0.0 < hi:
            self.fail("bounds_kms", f"must satisfy lo < 0 < hi, got ({lo}, {hi})")


@dataclass(frozen=True, kw_only=True)
class Molecule(ConfigGroup):
    """Fit control of one molecule (molecfit-style relative column ``relcol``: the
    molecule's column as a multiple of its column in the atmospheric profile).
    """

    fit: bool = False
    """Fit this molecule's relcol (only the 12 engine molecules can be fitted)."""
    relcol: float = 1.0
    """Initial / fixed relative column."""
    bounds: tuple[float, float] = (1e-5, 1e2)
    """(lo, hi) fit bounds on relcol."""
    bands_nm: Windows = ()
    """Principal absorption bands [nm], used to pre-select orders."""
    anchor_orders: tuple[int, ...] | None = None
    """Orders in which to fit the molecule. ``None`` = the instrument's default;
    ``()`` = no assigned orders (select orders by ``bands_nm``)."""

    def validate(self) -> None:
        """Positive ordered bounds containing ``relcol``; valid band windows."""
        lo, hi = self.bounds
        if not 0 < lo < hi:
            self.fail("bounds", f"must satisfy 0 < lo < hi, got ({lo}, {hi})")
        if not lo <= self.relcol <= hi:
            self.fail("relcol", f"must lie within bounds ({lo}, {hi}), got {self.relcol}")
        check_windows(self, "bands_nm", self.bands_nm)


def _m(fit: bool, bounds: tuple[float, float], bands: list[tuple[float, float]]) -> Molecule:
    """A default-table entry (relcol 1, no anchor orders of its own)."""
    return Molecule(fit=fit, relcol=1.0, bounds=bounds, bands_nm=tuple(bands))


_WIDE = (1e-5, 1e2)

#: Default molecule table: H2O and O2 fitted, the others fixed at relcol 1. Bands
#: [nm] select orders for a fitted molecule without anchor orders.
DEFAULT_MOLECULES: Mapping[str, Molecule] = FrozenDict(
    {
        "h2o": _m(True, (1e-2, 1e2), [(570, 605), (640, 690), (700, 745), (780, 840),
                                      (890, 985), (1080, 1200), (1300, 1500),
                                      (1750, 2050), (2500, 3000), (5500, 7500)]),
        "co2": _m(False, (0.5, 2.0), [(1570, 1630), (1950, 2100), (2700, 2900),
                                      (4200, 4450), (4750, 5200), (13000, 17000)]),
        "o3": _m(False, (0.3, 3.0), [(310, 360), (450, 750), (4700, 4850), (9400, 9900)]),
        "n2o": _m(False, (0.5, 2.0), [(2100, 2300), (2850, 3000), (3850, 4150),
                                      (4250, 4650), (7500, 8800), (16000, 17500)]),
        "co": _m(False, (0.5, 2.0), [(2280, 2400), (4500, 4800)]),
        "ch4": _m(False, (0.5, 2.0), [(1600, 1700), (2150, 2480), (3100, 3500), (6300, 8200)]),
        "o2": _m(True, (0.5, 1.5), [(627, 630), (686, 695), (759, 772),
                                    (1060, 1080), (1255, 1285)]),
        "no": _m(False, _WIDE, [(5200, 5500)]),
        "so2": _m(False, _WIDE, [(3950, 4050), (7200, 7500), (8500, 9000)]),
        "no2": _m(False, _WIDE, [(400, 650), (3400, 3500), (6100, 6300)]),
        "nh3": _m(False, _WIDE, [(1500, 1600), (2000, 2100), (2900, 3100),
                                 (6000, 6300), (8800, 11500)]),
        "hno3": _m(False, _WIDE, [(5800, 6000), (7400, 7700), (11000, 11400)]),
        "oh": _m(False, _WIDE, [(2700, 2900)]),
        "hf": _m(False, _WIDE, [(2400, 2600)]),
        "hcl": _m(False, _WIDE, [(3300, 3600)]),
        "hbr": _m(False, _WIDE, [(3800, 4100)]),
        "hi": _m(False, _WIDE, [(4200, 4500)]),
        "clo": _m(False, _WIDE, [(11000, 12000)]),
        "ocs": _m(False, _WIDE, [(2000, 2100), (2400, 2500), (3400, 3500),
                                 (4800, 5000), (11400, 11900)]),
        "h2co": _m(False, _WIDE, [(3450, 3650), (5600, 5900)]),
        "hocl": _m(False, _WIDE, [(3200, 3400), (8000, 8500)]),
        "n2": _m(False, _WIDE, [(4000, 4500)]),
        "hcn": _m(False, _WIDE, [(3000, 3100), (7000, 7200), (13000, 14500)]),
        "ch3cl": _m(False, _WIDE, [(3200, 3500), (6800, 7200), (9500, 10500)]),
        "h2o2": _m(False, _WIDE, [(7600, 8000), (11000, 12500)]),
        "c2h2": _m(False, _WIDE, [(3000, 3100), (7300, 7800), (13000, 14000)]),
        "c2h6": _m(False, _WIDE, [(3300, 3500), (6700, 6900), (11900, 12300)]),
        "ph3": _m(False, _WIDE, [(4000, 4400), (8800, 11500)]),
        "cof2": _m(False, _WIDE, [(5000, 5300), (7600, 8000), (11800, 12200)]),
        "sf6": _m(False, _WIDE, [(10300, 10800)]),
        "h2s": _m(False, _WIDE, [(3700, 4100), (7000, 8000)]),
        "hcooh": _m(False, _WIDE, [(7000, 9500)]),
        "ho2": _m(False, _WIDE, [(9000, 9500)]),
        "o": _m(False, _WIDE, []),
        "clono2": _m(False, _WIDE, [(7600, 7900), (12000, 12700)]),
        "no+": _m(False, _WIDE, []),
        "hobr": _m(False, _WIDE, [(8000, 8500)]),
        "c2h4": _m(False, _WIDE, [(3200, 3400), (9900, 10200)]),
        "ch3oh": _m(False, _WIDE, [(3300, 3600), (9000, 10000)]),
    }
)  # fmt: skip


class Molecules(ConfigValue, Mapping[str, Molecule]):
    """The full molecule table (all 39 molecules, in LBLRTM ID order).

    Built by *merging* onto the defaults: a TOML table that only gives
    ``h2o = {bounds = [0.01, 50.0]}`` changes h2o's bounds and keeps its other
    fields, and every other molecule keeps its default entry.
    """

    __slots__ = ("_table",)

    def __init__(self, table: Mapping[str, Molecule | Mapping[str, Any]] | None = None) -> None:
        merged: dict[str, Molecule] = dict(DEFAULT_MOLECULES)
        for name, entry in (table or {}).items():
            if name not in DEFAULT_MOLECULES:
                raise ConfigError(
                    f"{name}: unknown molecule.{did_you_mean(str(name), MOLECULE_NAMES)}"
                )
            if isinstance(entry, Molecule):
                merged[name] = entry
            elif isinstance(entry, Mapping):
                base = merged[name].to_dict()
                try:
                    merged[name] = Molecule.from_dict({**base, **entry})
                except ConfigError as exc:
                    raise ConfigError(f"{name}.{exc}") from None
            else:
                raise ConfigError(f"{name}: expected a table of parameters, got {entry!r}")
        for name, molecule in merged.items():
            if molecule.fit and name not in ENGINE_MOLECULES:
                raise ConfigError(
                    f"{name}.fit: the telluric model can only fit {', '.join(ENGINE_MOLECULES)}"
                )
        self._table = FrozenDict({name: merged[name] for name in MOLECULE_NAMES})

    @classmethod
    def default(cls) -> Molecules:
        """The default table (fits h2o and o2)."""
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Molecules:
        """Merge a TOML-style table onto the defaults."""
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Every molecule's entry as a plain table."""
        return {name: molecule.to_dict() for name, molecule in self._table.items()}

    def update(self, **entries: Molecule | Mapping[str, Any]) -> Molecules:
        """Return a copy with the given molecules replaced (Molecule) or merged (mapping)."""
        return Molecules({**self._table, **entries})

    @property
    def fitted(self) -> tuple[str, ...]:
        """Names with ``fit = true``, in LBLRTM ID order."""
        return tuple(name for name, m in self._table.items() if m.fit)

    def __getitem__(self, name: str) -> Molecule:
        return self._table[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Molecules) and self._table == other._table

    def __hash__(self) -> int:
        return hash(self._table)

    def __repr__(self) -> str:
        return f"Molecules(fitted={list(self.fitted)})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (Molecules, (dict(self._table),))


@dataclass(frozen=True, kw_only=True)
class MoleculeSelection(ConfigGroup):
    """Per-run overrides of which molecules are fitted.

    Not equivalent to setting ``Molecule.fit``: an *added* molecule without anchor
    orders is fitted in **every** requested order, while a ``fit = true`` molecule
    without anchor orders is fitted only in orders overlapping its ``bands_nm``.
    """

    add: tuple[str, ...] = ()
    """Force these engine molecules into the fit."""
    exclude: tuple[str, ...] = ()
    """Remove these molecules from the fit (highest priority)."""

    def validate(self) -> None:
        """Engine molecules only, none both added and excluded."""
        for name in ("add", "exclude"):
            for molecule in getattr(self, name):
                if molecule not in ENGINE_MOLECULES:
                    self.fail(
                        name,
                        f"{molecule!r} cannot be fitted; engine molecules are "
                        f"{', '.join(ENGINE_MOLECULES)}.{did_you_mean(molecule, ENGINE_MOLECULES)}",
                    )
        both = sorted(set(self.add) & set(self.exclude))
        if both:
            self.fail("add", f"{both[0]!r} appears in both add and exclude")


@dataclass(frozen=True, kw_only=True)
class OrderSelection(ConfigGroup):
    """Which echelle orders are fitted and which are written out."""

    fit_orders: tuple[int, ...] | None = None
    """Orders to fit. ``None`` = from the molecules' anchor orders / bands."""
    output_orders: tuple[int, ...] | None = None
    """Orders to correct and write (joint mode). ``None`` = all instrument orders."""
