"""Fit results: one order (:class:`OrderFitResult`) and a joint-order fit
(:class:`JointOrdersFitResult`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gtellcorr.fitting.wave_shift import corrected_wavelengths

__all__ = ["JointOrdersFitResult", "OrderFitResult"]

Array = NDArray[np.float64]


@dataclass
class OrderFitResult:
    """Best-fit telluric model, correction and parameters of one order (the field
    names are the column and extension names of the products)."""

    order: int
    """Echelle order."""
    wave_input: Array
    """Delivered wavelength grid [nm] (labelling)."""
    wave_fit: Array
    """Air + topocentric grid [nm]."""
    flux_obs: Array
    """Observed flux, divided by its median positive value."""
    err: Array
    """1-sigma flux uncertainty, scaled like the flux."""
    tell_model: Array
    """Fitted telluric transmission on the data grid."""
    flux_corrected: Array
    """``flux_obs / tell_model`` on all pixels (no floor, no clipping)."""
    err_corrected: Array | None = None
    """``err / tell_model``; diverges in saturated cores, flagging them."""
    continuum: Array | None = None
    """Fitted continuum ``C``."""
    stellar_model: Array | None = None
    """Stellar model ``M`` (``template`` treatment)."""
    exclusion: NDArray[np.bool_] | None = None
    """True where the pixel was excluded from the fit."""
    relcol: dict[str, float] = field(default_factory=dict)
    """Relative column per molecule."""
    relcol_err: dict[str, float] = field(default_factory=dict)
    """1-sigma errors of ``relcol``."""
    resolution: float | None = None
    """Fitted (or fixed) resolving power."""
    resolution_err: float | None = None
    """1-sigma error of R (``None`` if fixed)."""
    lsf_kind: str = "gauss"
    lsf_shape: float | None = None
    """LSF shape (``None`` for ``gauss``)."""
    lsf_shape_err: float | None = None
    wave_shift: tuple[float, ...] | None = None
    """Chebyshev coefficients [km/s] of the wavelength-solution correction (``None``:
    no correction; :mod:`gtellcorr.fitting.wave_shift`)."""
    wave_shift_err: tuple[float, ...] | None = None
    stellar_treatment: str = "mask"
    converged: bool | None = None
    """Whether the optimiser converged (``None``: not fitted). In a joint fit, that of
    the joint fit."""
    n_eval: int | None = None
    """Residual evaluations of the optimiser (``None``: not fitted)."""
    chi2: float = float("nan")
    """Chi-square of this order (single-order fit: of the fit).

    In a joint fit ``chi2_reduced`` and ``dof_mask_removed`` are the joint fit's
    (equal to ``chi2_reduced_joint`` / ``dof_mask_removed_joint``), while ``chi2``
    is this order's (equal to ``chi2_order``)."""
    chi2_reduced: float = float("nan")
    dof_mask_removed: int | None = None
    """Kept, valid pixels minus fitted and projected parameters."""
    chi2_order: float = float("nan")
    """Chi-square of this order alone."""
    chi2_reduced_order: float = float("nan")
    dof_mask_removed_order: int | None = None
    chi2_joint: float = float("nan")
    chi2_reduced_joint: float = float("nan")
    dof_mask_removed_joint: int | None = None
    chi2_clean: float = float("nan")
    status: str = "ok"
    """``ok`` / ``joint_ok``, or why not (``not_converged:<message>``,
    ``joint_failed:<message>``, ``no_fit_molecules``, ``corrected_not_fitted`` ...)."""

    @property
    def wave_fit_corr(self) -> Array | None:
        """``wave_fit`` corrected on the telluric lines (``None`` without a correction)."""
        if self.wave_shift is None:
            return None
        return corrected_wavelengths(self.wave_fit, self.wave_input, self.wave_shift)[0]

    @property
    def wave_input_corr(self) -> Array | None:
        """``wave_input`` corrected on the telluric lines (``None`` without a correction)."""
        if self.wave_shift is None:
            return None
        return corrected_wavelengths(self.wave_fit, self.wave_input, self.wave_shift)[1]


@dataclass
class JointOrdersFitResult:
    """Joint-order fit of one exposure."""

    order_results: list[OrderFitResult]
    relcol: dict[str, float]
    relcol_err: dict[str, float]
    chi2: float
    chi2_reduced: float
    dof_mask_removed: int
    n_eval: int
    success: bool
    message: str
    elapsed_s: float
    parameters: Mapping[str, float] = field(default_factory=dict)
    """Every fitted parameter by theta name (``relcol_h2o``, ``order078_resolution`` …)."""
    parameter_errors: Mapping[str, float] = field(default_factory=dict)
