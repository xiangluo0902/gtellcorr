"""Parameter groups of the pre-fit preparation: fit masks, stellar master spectrum,
stellar template and radial velocities.

All wavelengths are in nm; windows are ``(lo, hi)`` pairs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from gtellcorr.config.base import ConfigGroup, Windows, check_windows
from gtellcorr.telluric.molecules import MOLECULE_NAMES

__all__ = [
    "BROAD_LINE_WINDOWS_OPTICAL",
    "OPTICAL_ACTIVITY_WINDOWS",
    "TEMPLATE_FRAMES",
    "FitMasks",
    "IspecSettings",
    "MasterContinuum",
    "MasterSpectrum",
    "PysmeSettings",
    "RVConfig",
    "StellarLineMask",
    "StellarModelConfig",
    "StrongTelluricMask",
]

#: Supported frames of a user-supplied stellar template (``StellarModelConfig``).
TEMPLATE_FRAMES = ("rest", "topocentric")

#: Chromospheric / activity-sensitive lines (air nm), pre-widened for the wings.
OPTICAL_ACTIVITY_WINDOWS: Windows = (
    (392.4, 394.4),  # Ca II K   393.366
    (395.9, 397.8),  # Ca II H   396.847
    (409.4, 410.9),  # H-delta   410.174
    (433.3, 434.8),  # H-gamma   434.047
    (485.3, 487.0),  # H-beta    486.134
    (587.1, 588.0),  # He I D3   587.562
    (588.3, 590.3),  # Na I D2+D1 588.995 / 589.592
    (654.8, 657.8),  # H-alpha   656.281
    (848.9, 850.7),  # Ca II IRT 849.802
    (853.3, 855.1),  # Ca II IRT 854.209
    (865.3, 867.1),  # Ca II IRT 866.214
)

#: Strong pressure-broadened lines excluded from the master-spectrum continuum
#: base points (air nm).
BROAD_LINE_WINDOWS_OPTICAL: Windows = (
    (392.0, 394.7),  # Ca II K
    (395.5, 398.2),  # Ca II H
    (409.4, 411.0),  # H-delta
    (421.8, 423.6),  # Ca I 422.67
    (433.2, 434.9),  # H-gamma
    (484.9, 487.4),  # H-beta
    (515.6, 519.4),  # Mg b
    (588.0, 590.6),  # Na I D
    (654.3, 658.3),  # H-alpha
    (848.6, 851.0),  # Ca II IRT
    (853.0, 855.4),  # Ca II IRT
    (865.0, 867.4),  # Ca II IRT
)


# ---------------------------------------------------------------------------
# Fit masks
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class StellarLineMask(ConfigGroup):
    """Stellar-line windows excluded from the telluric fit (``stellar = "mask"``).

    The windows come from ``user_file`` if it is given, otherwise from the spectrum
    chosen by ``from_spectrum``. They are placed on every exposure by its RV.
    """

    from_spectrum: Literal["master", "template"] = "master"
    """Where the stellar lines are found. ``master``: measured with iSpec on the
    out-of-transit master spectrum and identified in the iSpec line list, which keeps
    telluric lines out (only lines of ``stellar_model.line_elements`` deeper than
    ``stellar_model.line_min_depth``). ``template``: every line of the stellar
    template deeper than ``depth_below``, with no iSpec needed. It excludes more
    pixels, including those where stellar and telluric lines blend, so less of the
    telluric absorption enters the fit. A user-supplied ``stellar_model.template_file``
    is used as given: an observed spectrum without telluric correction is not
    suitable, as its telluric lines would be masked as stellar lines."""
    depth_below: float = 0.97
    """``from_spectrum = "template"``: template flux below this value marks a stellar
    line."""
    pad_kms: float = 2.0
    """``from_spectrum = "template"``: widen each window on each side [km/s]."""
    merge_gap_nm: float = 0.02
    """``from_spectrum = "template"``: merge windows separated by at most this [nm]."""
    user_file: str | None = None
    """Plain-text file of ``lo_nm hi_nm`` stellar-line windows in the target rest frame
    (air wavelengths), used instead of ``from_spectrum``."""

    def validate(self) -> None:
        """``depth_below`` in (0, 1]; non-negative padding and gap."""
        if not 0 < self.depth_below <= 1:
            self.fail("depth_below", f"must be in (0, 1], got {self.depth_below}")
        for name in ("pad_kms", "merge_gap_nm"):
            if getattr(self, name) < 0:
                self.fail(name, f"must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True, kw_only=True)
class StrongTelluricMask(ConfigGroup):
    """Saturated telluric cores excluded from the fit.

    One transmission model is computed for the run, with the atmospheric profile,
    airmass and resolving power of the first exposure; wavelengths where it falls below
    ``threshold`` are excluded from the fit of every exposure.
    """

    enabled: bool = True
    """Build and apply the mask."""
    threshold: float = 0.05
    """Transmission below this value marks a saturated core."""
    h2o_relcol: float = 1.0
    """Relative H2O column of the mask model (1 = the atmospheric profile's water); the
    other molecules keep their reference columns. The fitted columns are set in
    ``molecules``."""
    chunk_nm: float = 30.0
    """Wavelength chunk of each radiative-transfer calculation [nm]."""
    pad_nm: float = 0.0
    """Widen each window on each side [nm]."""
    merge_gap_nm: float = 0.0
    """Merge windows separated by at most this [nm]."""
    min_width_nm: float = 0.0
    """Drop (merged) windows narrower than this [nm]."""

    def validate(self) -> None:
        """Threshold in (0, 1); positive column and chunk; non-negative widths."""
        if not 0 < self.threshold < 1:
            self.fail("threshold", f"must be in (0, 1), got {self.threshold}")
        if self.h2o_relcol <= 0:
            self.fail("h2o_relcol", f"must be > 0, got {self.h2o_relcol}")
        if self.chunk_nm <= 0:
            self.fail("chunk_nm", f"must be > 0, got {self.chunk_nm}")
        for name in ("pad_nm", "merge_gap_nm", "min_width_nm"):
            if getattr(self, name) < 0:
                self.fail(name, f"must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True, kw_only=True)
class FitMasks(ConfigGroup):
    """Pixels excluded from the telluric fit."""

    stellar_linemask: StellarLineMask = field(default_factory=StellarLineMask)
    """Stellar-line windows (``stellar = "mask"`` only)."""
    strong_telluric: StrongTelluricMask = field(default_factory=StrongTelluricMask)
    """Saturated telluric cores."""
    activity_windows_nm: Windows = OPTICAL_ACTIVITY_WINDOWS
    """Stellar activity lines (air nm, topocentric), excluded from the fit; also masked
    in the RV cross-correlation when ``rv.mask_activity`` is on."""
    bad_regions_nm: Windows = ()
    """Further windows excluded from the fit (air nm, topocentric)."""

    def validate(self) -> None:
        """Every window list must hold increasing ``(lo, hi)`` pairs."""
        check_windows(self, "activity_windows_nm", self.activity_windows_nm)
        check_windows(self, "bad_regions_nm", self.bad_regions_nm)


# ---------------------------------------------------------------------------
# Stellar master spectrum
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MasterContinuum(ConfigGroup):
    """Continuum normalisation of each order of the stellar master spectrum.

    ``ispec`` uses iSpec's ``fit_continuum`` (the fields below up to
    ``ignore_broad_lines`` are passed to it); ``poly`` fits a polynomial with
    asymmetric sigma clipping. If iSpec fails, the polynomial is used.
    """

    method: Literal["ispec", "poly"] = "ispec"
    """Continuum method."""
    model: Literal["Splines", "Polynomy"] = "Splines"
    """iSpec continuum model."""
    order: str = "median+max"
    """iSpec filter order, ``median+max`` or ``max+median``."""
    nknots: int | None = None
    """Spline knots; leave out for one knot per ``knot_wave_nm``."""
    knot_wave_nm: float = 3.0
    """Knot spacing used when ``nknots`` is not given [nm]."""
    degree: int = 2
    """Spline or polynomial degree of the iSpec model."""
    median_wave_range: float = 0.05
    """Width of iSpec's median filter [nm]."""
    max_wave_range: float = 2.0
    """Width of iSpec's maximum filter [nm]."""
    strong_line_detection: bool = True
    """Let iSpec reject base points inside strong lines."""
    strong_line_probability: float = 0.5
    """Probability threshold of iSpec's strong-line detection."""
    from_resolution: float | None = None
    """Sets iSpec's resampling step to (shortest wavelength) / ``from_resolution``;
    leave out to resample at the median pixel step."""
    ignore_broad_lines: bool = True
    """Exclude ``master_spectrum.broad_line_mask_nm`` from the base points."""
    poly_order: int = 3
    """Degree of the ``poly`` continuum."""

    def validate(self) -> None:
        """Positive knot counts and filter widths; probability in [0, 1]."""
        if self.nknots is not None and self.nknots < 1:
            self.fail("nknots", f"must be >= 1, got {self.nknots}")
        for name in ("knot_wave_nm", "median_wave_range", "max_wave_range"):
            if getattr(self, name) <= 0:
                self.fail(name, f"must be > 0, got {getattr(self, name)}")
        if not 0 <= self.strong_line_probability <= 1:
            self.fail("strong_line_probability", "must be in [0, 1]")


@dataclass(frozen=True, kw_only=True)
class MasterSpectrum(ConfigGroup):
    """Stellar master spectrum: the out-of-transit exposures shifted to the stellar
    rest frame and co-added. Its lines give the stellar-line mask (``stellar = "mask"``)."""

    continuum: MasterContinuum = field(default_factory=MasterContinuum)
    """Continuum normalisation of the master."""
    broad_line_mask_nm: Windows = BROAD_LINE_WINDOWS_OPTICAL
    """Windows excluded from the continuum base points (air nm, rest frame)."""

    def validate(self) -> None:
        """Increasing ``(lo, hi)`` windows."""
        check_windows(self, "broad_line_mask_nm", self.broad_line_mask_nm)


# ---------------------------------------------------------------------------
# Stellar template synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class IspecSettings(ConfigGroup):
    """iSpec settings (``stellar_model.backend = "ispec"``); file names refer to the
    ``input/`` folders of the iSpec installation (``ispec_dir`` setting)."""

    code: Literal["spectrum", "turbospectrum", "moog", "sme", "synthe"] = "spectrum"
    """Radiative-transfer code iSpec runs."""
    atmosphere: str = "MARCS.GES"
    """Model-atmosphere grid in ``input/atmospheres``."""
    linelist: str = "GESv6_atom_hfs_iso.420_920nm"
    """Atomic line list in ``input/linelists/transitions``; also used to identify the
    lines measured on the master spectrum (``stellar = "mask"``, either backend)."""
    abundances: str = "Grevesse.2007"
    """Solar abundances in ``input/abundances``."""
    isotopes: str = "SPECTRUM.lst"
    """Isotope data in ``input/isotopes``."""
    min_depth: float = 0.01
    """Keep only lines with theoretical depth >= this."""
    wave_step_nm: float = 0.001
    """Wavelength step of the synthesis [nm]."""
    resolution: int = 500000
    """Resolving power iSpec gives the raw synthesis; the template is then broadened
    from it to the instrument resolution (must be higher than that). 0 = none."""


@dataclass(frozen=True, kw_only=True)
class PysmeSettings(ConfigGroup):
    """PySME synthesis settings (``stellar_model.backend = "pysme"``)."""

    atmosphere: str = "marcs2012"
    """Model-atmosphere grid in ``~/.sme`` (``.sav`` is appended if missing)."""
    abundances: str = "grevesse2007"
    """Solar abundance pattern scaled by [M/H]: one of PySME's named patterns
    (``grevesse2007``, ``asplund2009``, ``asplund2021``, ``lodders2003``, ...).
    ``grevesse2007`` is PySME's own default."""
    nlte: Mapping[str, str] = field(default_factory=dict)
    """Optional per-element NLTE grids, e.g. ``{Fe = "marcs2012_Fe2016.grd"}``."""
    wave_step_nm: float | None = None
    """Output wavelength step of the synthesis [nm]; ``None`` keeps PySME's own
    adaptive grid."""
    line_strength_cut: float | None = None
    """Weak-line cull on ``log gf - 5040 / Teff * E_low``; ``None`` keeps every line."""

    def validate(self) -> None:
        """A given ``wave_step_nm`` must be positive."""
        if self.wave_step_nm is not None and not self.wave_step_nm > 0:
            self.fail("wave_step_nm", f"must be > 0, got {self.wave_step_nm}")


@dataclass(frozen=True, kw_only=True)
class StellarModelConfig(ConfigGroup):
    """Stellar template: synthesised from ``data.target``, or user-supplied.

    The template is needed by both stellar treatments (the RV measurement
    cross-correlates with it); the stellar-line measurement settings apply to
    ``stellar = "mask"``.
    """

    backend: Literal["pysme", "ispec"] = "pysme"
    """Synthesis code: PySME (needs the ``stellar_linelist`` setting) or iSpec (needs
    the ``ispec_dir`` setting). The synthesised template is broadened to the
    instrument resolution by gtellcorr itself."""
    template_file: str | None = None
    """A user-supplied template instead of synthesis: two columns ``wave_nm flux``,
    continuum-normalised."""
    template_medium: Literal["air", "vacuum"] = "air"
    """Medium of the ``template_file`` wavelengths; ``vacuum`` is converted to air with
    ``frames.vacuum_air_formula`` when the template is loaded."""
    template_frame: str = "rest"
    """Frame of the ``template_file`` wavelengths: ``rest`` (target rest frame, shifted
    to every exposure by its topocentric RV) or ``topocentric`` (already at the
    observed positions, never shifted). Other frames (e.g. barycentric) are not
    supported: convert the template first."""
    template_resolution: float | None = None
    """Resolving power of the ``template_file``. ``None``: already at the observed
    resolution. A value above the exposures' R (header R, e.g. the PEPSI fibre; else the
    instrument default) broadens the template to it with a Gaussian in velocity space.
    Exposures with different R cannot share one broadened template."""
    line_elements: tuple[str, ...] = ("Fe", "Ca", "Na", "Mg", "Ti", "Cr", "Ni", "Si", "Mn")
    """Elements whose lines are measured on the master and masked."""
    line_min_depth: float = 0.10
    """Keep only measured lines with central depth >= this."""
    ispec: IspecSettings = field(default_factory=IspecSettings)
    """iSpec settings."""
    pysme: PysmeSettings = field(default_factory=PysmeSettings)
    """PySME settings."""

    def validate(self) -> None:
        """Supported template frame; template options only with a ``template_file``."""
        if not 0 < self.line_min_depth < 1:
            self.fail("line_min_depth", f"must be in (0, 1), got {self.line_min_depth}")
        if self.template_frame not in TEMPLATE_FRAMES:
            self.fail(
                "template_frame",
                f"{self.template_frame!r} is not supported; provide the template in the "
                "target rest frame ('rest') or the topocentric frame ('topocentric')",
            )
        if self.template_resolution is not None and not self.template_resolution > 0:
            self.fail("template_resolution", f"must be > 0, got {self.template_resolution}")
        if self.template_file is None:
            for name, default in (
                ("template_medium", "air"),
                ("template_frame", "rest"),
                ("template_resolution", None),
            ):
                if getattr(self, name) != default:
                    self.fail(name, "applies only to a user-supplied template_file")


# ---------------------------------------------------------------------------
# Radial velocities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class RVConfig(ConfigGroup):
    """Per-exposure stellar RV: measured by cross-correlation, or user-supplied."""

    source: Literal["measure", "user_file"] = "measure"
    """``measure``: CCF against the template + MCMC; ``user_file``: read ``user_rv_file``."""
    user_rv_file: str | None = None
    """CSV with columns ``file`` (exposure file name or stem) and ``rv_kms``, optional
    ``rv_err_kms``: the apparent topocentric RV, ``lambda_obs = lambda_rest (1 + rv/c)``
    (i.e. ``c z`` of the target as seen from the observatory). Every exposure needs a row."""

    orders: tuple[int, ...] | None = None
    """Orders used for the RV; leave out for all orders of the instrument (orders
    whose CCF fails the quality checks are dropped)."""
    telluric_mask_molecules: tuple[str, ...] = ("h2o", "o2")
    """Molecules whose ``bands_nm`` are masked in the CCF."""
    mask_activity: bool = True
    """Also mask ``masks.activity_windows_nm``."""
    pad_kms: float = 60.0
    """Half-width added to every mask window [km/s]."""
    ccf_v_lo: float = -60.0
    """Lower end of the CCF velocity grid [km/s]: the absolute topocentric RV, so the
    grid must contain the target's apparent RV."""
    ccf_v_hi: float = 60.0
    """Upper end of the CCF velocity grid [km/s]."""
    ccf_dv: float = 1.0
    """Step of the CCF velocity grid [km/s]."""
    plot_ccf: bool = False
    """Also draw each exposure's CCF diagnostics (``Radial_Velocities/figure/ccf``) when
    the run writes figures; the RV time series is drawn whenever figures are on."""
    ccf_plot_max_orders: int | None = None
    """Per-order CCFs drawn in a CCF figure: the highest-weight ones; ``None`` = all."""
    user_mask_nm: Windows = ()
    """Further windows masked in the CCF (air nm, topocentric)."""
    user_mask_file: str | None = None
    """Plain-text file of ``lo_nm hi_nm`` windows masked in the CCF."""

    def validate(self) -> None:
        """Known molecules, valid CCF grid and windows; ``user_rv_file`` iff ``user_file``."""
        for molecule in self.telluric_mask_molecules:
            if molecule not in MOLECULE_NAMES:
                self.fail("telluric_mask_molecules", f"unknown molecule {molecule!r}")
        if self.pad_kms < 0:
            self.fail("pad_kms", f"must be >= 0, got {self.pad_kms}")
        if not self.ccf_v_lo < self.ccf_v_hi:
            self.fail("ccf_v_lo", "must be smaller than ccf_v_hi")
        if self.ccf_dv <= 0:
            self.fail("ccf_dv", f"must be > 0, got {self.ccf_dv}")
        if self.ccf_plot_max_orders is not None and self.ccf_plot_max_orders < 1:
            self.fail("ccf_plot_max_orders", "must be >= 1")
        check_windows(self, "user_mask_nm", self.user_mask_nm)
        if self.source == "user_file" and self.user_rv_file is None:
            self.fail("user_rv_file", "required when source = 'user_file'")
        if self.source != "user_file" and self.user_rv_file is not None:
            self.fail("user_rv_file", "set source = 'user_file' to use it")
