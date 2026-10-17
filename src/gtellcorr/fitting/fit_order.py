"""Single-order and joint-order telluric fits.

The engine takes prepared inputs only: the spectrum, its fit spec, the optical-depth cache, and
(from :mod:`gtellcorr.preprocessing`) the stellar model and exclusion mask. It
reads no files and no configuration.

Masks: ``exclusion`` is True where a pixel is excluded (stellar lines, activity
windows, bad regions, strong telluric cores); the fit keeps the complement.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from gtellcorr.fitting.blocks import ModelState, ParameterBlock
from gtellcorr.fitting.continuum import fit_continuum
from gtellcorr.fitting.fit_spec import FitSpec
from gtellcorr.fitting.results import JointOrdersFitResult, OrderFitResult
from gtellcorr.fitting.solver import (
    SpectrumLike,
    build_residual,
    covariance_errors,
    evaluate_order,
    inverse_variance,
    joint_layout,
    lines_only,
    minimize,
    pack,
    theta_scale,
    unpack,
)
from gtellcorr.fitting.stellar_shift import ShiftableStellarModel, stellar_on_data
from gtellcorr.telluric.order_cache import OrderCache

__all__ = ["JointOrderInput", "fit_joint_orders", "fit_single_order"]

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
CacheSource = OrderCache | Callable[[], OrderCache]


def _corrected(flux: Array, err: Array, telluric: Array) -> tuple[Array, Array]:
    """``flux / T`` and ``err / T`` on all pixels (NaN / inf where T <= 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.where(telluric > 0.0, flux / telluric, np.nan),
            np.where(telluric > 0.0, err / telluric, np.inf),
        )


def _output_continuum(
    spectrum: SpectrumLike, spec: FitSpec, evaluated: Any, *, keep: Any, weight: Array
) -> Array:
    """The continuum stored with the result: the fit's own, or (``continuum_output_niter``
    differing from ``continuum_niter``) a sigma-clipped refit against the fitted model."""
    if int(spec.continuum_output_niter) == int(spec.continuum_niter):
        return np.asarray(evaluated.continuum, float)
    return np.asarray(
        fit_continuum(
            np.asarray(spectrum.wave_fit, float),
            np.asarray(spectrum.flux, float),
            evaluated.multiplicative_model,
            order=int(spec.continuum_order),
            sigma_clip=(float(spec.continuum_sigma_clip[0]), float(spec.continuum_sigma_clip[1])),
            max_iter=int(spec.continuum_output_niter),
            mask=np.asarray(keep, bool),
            weight=weight,
        ),
        float,
    )


def _name_errors(errors: Array | None, names: Sequence[str]) -> dict[str, float]:
    """Parameter errors by theta name (empty if the covariance was singular)."""
    if errors is None:
        return {}
    return {name: float(e) for e, name in zip(errors, names, strict=True)}


def _stellar_input(stellar_model: Any | None, spec: FitSpec) -> Any | None:
    """The stellar model as the fit uses it: kept shiftable when the spec moves the
    template (fitted or fixed ``stellar_shift``), else as an array."""
    if stellar_model is None:
        if spec.fit_stellar_shift:
            raise ValueError("fit_stellar_shift needs a stellar model")
        return None
    if spec.fit_stellar_shift or spec.stellar_shift is not None:
        if not isinstance(stellar_model, ShiftableStellarModel):
            raise ValueError("a stellar shift needs a ShiftableStellarModel, not a fixed array")
        return stellar_model
    return np.asarray(stellar_model, float)


def _shift(state: ModelState) -> tuple[float, ...] | None:
    """The correction coefficients of *state* as floats (``None``: no correction)."""
    return None if state.wave_shift is None else tuple(float(v) for v in state.wave_shift)


def _shift_err(spec: FitSpec, errors: dict[str, float], prefix: str) -> tuple[float, ...] | None:
    """Errors of the fitted correction coefficients (shared ones carry no order prefix)."""
    if not spec.fit_wave_shift:
        return None
    if spec.wave_shift_scope == "exposure":
        prefix = ""
    names = [f"{prefix}wave_shift_c{k}" for k in range(int(spec.wave_shift_degree) + 1)]
    if not all(n in errors for n in names):
        return None
    return tuple(errors[n] for n in names)


# ---------------------------------------------------------------------------
# single order
# ---------------------------------------------------------------------------
def fit_single_order(
    spectrum: SpectrumLike,
    spec: FitSpec,
    cache: CacheSource,
    *,
    stellar_model: Any | None = None,
    exclusion: Any | None = None,
    stellar_treatment: str = "mask",
    blocks: Sequence[ParameterBlock] | None = None,
) -> OrderFitResult:
    """Fit one order.

    Parameters
    ----------
    spectrum
        One order in the air + topocentric fit frame.
    spec
        From :func:`~gtellcorr.fitting.fit_spec.build_fit_spec`.
    cache
        The order's optical-depth cache, or a zero-argument callable building it.
        A callable is only called once the order is known to be fittable, so
        orders without molecules or pixels never pay for an LBLRTM run.
    stellar_model
        Stellar template on ``wave_fit`` (``template`` treatment), a
        :class:`~gtellcorr.fitting.stellar_shift.ShiftableStellarModel` when the spec
        moves it; ``None`` fits ``C * T`` (``mask`` treatment).
    exclusion
        Boolean mask, True = excluded from the fit.
    stellar_treatment
        Recorded in the result.

    An order without fitted molecules, or with fewer usable pixels than free
    parameters plus continuum coefficients plus one, is returned uncorrected
    (``T = 1``) with status ``no_fit_molecules`` or ``insufficient_fit_pixels``.
    """
    order = int(spec.order) if spec.order is not None else -1
    x = np.asarray(spectrum.wave_fit, float)
    y = np.asarray(spectrum.flux, float)
    e = np.asarray(spectrum.err, float)
    if not spec.molecules:
        return OrderFitResult(
            order=order,
            wave_input=spectrum.wave_input,
            wave_fit=x,
            flux_obs=y,
            err=e,
            tell_model=np.ones_like(y),
            flux_corrected=y.copy(),
            err_corrected=e.copy(),
            status="no_fit_molecules",
        )

    fit_keep = np.ones(x.size, dtype=bool)
    excluded = None
    weight_mask = None
    if exclusion is not None:
        excluded = np.asarray(exclusion, dtype=bool)
        if excluded.ndim != 1 or excluded.size != x.size:
            raise ValueError("exclusion mask must be one-dimensional and match spectrum.wave_fit")
        fit_keep = ~excluded
        weight_mask = fit_keep.astype(float)
        excluded = excluded.copy()
    stellar = _stellar_input(stellar_model, spec)

    theta0, lo, hi, names = pack(spec, blocks)
    n_projected = int(spec.continuum_order) + 1
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0) & fit_keep
    n_effective = int(np.count_nonzero(valid))
    min_required = int(theta0.size) + n_projected + 1
    if n_effective < min_required:
        return OrderFitResult(
            order=order,
            wave_input=np.asarray(spectrum.wave_input, float),
            wave_fit=x,
            flux_obs=y,
            err=e,
            tell_model=np.ones_like(y),
            flux_corrected=y.copy(),
            err_corrected=e.copy(),
            continuum=None,
            stellar_model=stellar_on_data(stellar, None),
            exclusion=excluded,
            relcol={m: np.nan for m in spec.molecules},
            lsf_kind=spec.lsf_kind,
            stellar_treatment=stellar_treatment,
            status=(
                f"insufficient_fit_pixels: n_effective={n_effective}, required>={min_required}"
            ),
        )

    if not isinstance(cache, OrderCache):
        log.info("building or reusing the optical-depth cache of order %03d", order)
        cache = cache()
    residual = build_residual(
        spectrum, cache, spec, mask=weight_mask, stellar_model=stellar, blocks=blocks
    )
    opt = minimize(
        residual,
        theta0,
        lo=lo,
        hi=hi,
        n_effective=n_effective,
        n_projected=n_projected,
        x_scale=theta_scale(theta0),
    )

    state = unpack(opt.theta, spec, blocks)
    evaluated = evaluate_order(
        spectrum,
        cache,
        spec,
        state,
        weight=inverse_variance(e),
        keep_mask=fit_keep,
        stellar_model=stellar,
    )
    telluric, continuum = lines_only(
        evaluated,
        _output_continuum(spectrum, spec, evaluated, keep=fit_keep, weight=inverse_variance(e)),
    )
    corrected, err_corrected = _corrected(y, e, telluric)
    errors = _name_errors(opt.theta_err, names)
    return OrderFitResult(
        order=order,
        wave_input=np.asarray(spectrum.wave_input, float),
        wave_fit=x,
        flux_obs=y,
        err=e,
        tell_model=telluric,
        flux_corrected=corrected,
        err_corrected=err_corrected,
        continuum=continuum,
        stellar_model=stellar_on_data(stellar, state.stellar_shift),
        exclusion=excluded,
        relcol={m: float(state.scales[m]) for m in spec.molecules},
        relcol_err={n[len("relcol_") :]: v for n, v in errors.items() if n.startswith("relcol_")},
        resolution=float(state.resolution),
        resolution_err=errors.get("resolution"),
        lsf_kind=spec.lsf_kind,
        lsf_shape=float(state.lsf_shape) if state.lsf_shape is not None else None,
        lsf_shape_err=errors.get("lsf_shape"),
        wave_shift=_shift(state),
        wave_shift_err=_shift_err(spec, errors, ""),
        stellar_shift=state.stellar_shift,
        stellar_shift_err=errors.get("stellar_shift"),
        stellar_treatment=stellar_treatment,
        converged=bool(opt.success),
        n_eval=int(opt.n_eval),
        chi2=opt.chi2,
        chi2_reduced=opt.chi2_reduced,
        dof_mask_removed=opt.dof_mask_removed,
        status="ok" if opt.success else f"not_converged:{opt.message}",
    )


# ---------------------------------------------------------------------------
# joint orders
# ---------------------------------------------------------------------------
@dataclass
class JointOrderInput:
    """Prepared input of one order in a joint fit (:meth:`prepare`).

    ``keep_pixels`` is True for pixels used in the fit (valid data and not
    excluded); ``weight`` is the inverse variance.
    """

    order: int
    """Echelle order (``-1`` if the spec has none)."""
    spectrum: SpectrumLike
    cache: OrderCache
    spec: FitSpec
    stellar_model: Any | None
    """An array, or a :class:`ShiftableStellarModel` when the spec moves the template."""
    exclusion: NDArray[np.bool_] | None
    weight: Array
    keep_pixels: NDArray[np.bool_]

    @classmethod
    def prepare(
        cls,
        spectrum: SpectrumLike,
        spec: FitSpec,
        cache: OrderCache,
        *,
        stellar_model: Any | None = None,
        exclusion: Any | None = None,
    ) -> JointOrderInput:
        """Inverse-variance weights, and kept pixels: finite wavelength, flux and
        error, positive error, not excluded."""
        x = np.asarray(spectrum.wave_fit, float)
        flux = np.asarray(spectrum.flux, float)
        err = np.asarray(spectrum.err, float)
        keep = np.isfinite(x) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
        if exclusion is not None:
            keep &= ~np.asarray(exclusion, bool)
        return cls(
            order=int(spec.order) if spec.order is not None else -1,
            spectrum=spectrum,
            cache=cache,
            spec=spec,
            stellar_model=_stellar_input(stellar_model, spec),
            exclusion=None if exclusion is None else np.asarray(exclusion, bool),
            weight=inverse_variance(err),
            keep_pixels=keep,
        )


def _evaluate(item: JointOrderInput, state: ModelState) -> Any:
    """:func:`evaluate_order` of one joint-fit order."""
    return evaluate_order(
        item.spectrum,
        item.cache,
        item.spec,
        state,
        weight=item.weight,
        keep_mask=item.keep_pixels,
        stellar_model=item.stellar_model,
    )


def fit_joint_orders(
    order_inputs: Sequence[JointOrderInput],
    *,
    stellar_treatment: str = "mask",
    blocks: Sequence[ParameterBlock] | None = None,
) -> JointOrdersFitResult:
    """Fit one exposure with exposure-scope parameters (relcols) shared across orders.

    Order-scope parameters (resolution, LSF shape, per-order wavelength correction) and
    the projected continuum stay independent per order. The objective is the
    concatenation of the per-order residuals. Every order needs fitted molecules
    and more kept pixels than continuum coefficients plus one (``ValueError``).

    Each order result has status ``joint_ok``, or ``joint_failed:<message>`` if the
    optimiser stopped without converging (also in :attr:`JointOrdersFitResult.success`).
    """
    if not order_inputs:
        raise ValueError("fit_joint_orders requires at least one prepared order")
    for item in order_inputs:
        if not item.spec.molecules:
            raise ValueError(f"order {item.order}: fit spec has no molecules")
        if np.count_nonzero(item.keep_pixels) <= int(item.spec.continuum_order) + 2:
            raise ValueError(f"order {item.order}: insufficient kept pixels for joint fit")

    layout = joint_layout({item.order: item.spec for item in order_inputs}, blocks)

    def residual(theta: Array) -> Array:
        """The per-order residuals for *theta*, concatenated."""
        states = layout.states(theta)
        chunks = [_evaluate(item, states[item.order]).residual for item in order_inputs]
        return np.concatenate(chunks) if chunks else np.empty(0, float)

    started = time.time()
    opt = least_squares(
        residual,
        layout.theta0,
        bounds=(layout.lower, layout.upper),
        method="trf",
        x_scale=theta_scale(layout.theta0),
    )
    elapsed = time.time() - started

    n_effective = sum(int(np.count_nonzero(item.keep_pixels)) for item in order_inputs)
    n_projected = sum(int(item.spec.continuum_order) + 1 for item in order_inputs)
    dof = max(n_effective - len(layout.theta0) - n_projected, 1)
    chi2 = float(np.sum(opt.fun**2))
    chi2_reduced = chi2 / dof
    errors = _name_errors(covariance_errors(opt, dof), layout.names)
    parameters = {name: float(v) for name, v in zip(layout.names, opt.x, strict=True)}
    relcol = {n[len("relcol_") :]: v for n, v in parameters.items() if n.startswith("relcol_")}
    relcol_err = {n[len("relcol_") :]: v for n, v in errors.items() if n.startswith("relcol_")}

    states = layout.states(opt.x)
    results = [
        _joint_order_result(
            item,
            states[item.order],
            relcol_err,
            errors,
            layout_names=layout.names,
            chi2_joint=chi2,
            chi2_reduced_joint=chi2_reduced,
            dof_joint=dof,
            stellar_treatment=stellar_treatment,
            converged=bool(opt.success),
            n_eval=int(opt.nfev),
            message=str(opt.message),
        )
        for item in order_inputs
    ]
    return JointOrdersFitResult(
        order_results=results,
        relcol=relcol,
        relcol_err=relcol_err,
        chi2=chi2,
        chi2_reduced=float(chi2_reduced),
        dof_mask_removed=int(dof),
        n_eval=int(opt.nfev),
        success=bool(opt.success),
        message=str(opt.message),
        elapsed_s=float(elapsed),
        parameters=parameters,
        parameter_errors=errors,
    )


def _joint_order_result(
    item: JointOrderInput,
    state: ModelState,
    relcol_err: dict[str, float],
    errors: dict[str, float],
    *,
    layout_names: Sequence[str],
    chi2_joint: float,
    chi2_reduced_joint: float,
    dof_joint: int,
    stellar_treatment: str,
    converged: bool,
    n_eval: int,
    message: str,
) -> OrderFitResult:
    """The result of one order of a joint fit.

    ``chi2`` / ``chi2_order`` are this order's; ``chi2_reduced`` and
    ``dof_mask_removed`` are the joint fit's (as ``chi2_reduced_joint`` /
    ``dof_mask_removed_joint``). The order's own dof subtracts its continuum
    coefficients and order-scope parameters, not the shared ones.
    """
    evaluated = _evaluate(item, state)
    spectrum = item.spectrum
    telluric, continuum = lines_only(
        evaluated,
        _output_continuum(
            spectrum, item.spec, evaluated, keep=item.keep_pixels, weight=item.weight
        ),
    )
    corrected, err_corrected = _corrected(spectrum.flux, spectrum.err, telluric)
    prefix = f"order{item.order:03d}_"
    n_local = sum(1 for name in layout_names if name.startswith(prefix))
    n_effective = int(np.count_nonzero(item.keep_pixels))
    dof_order = max(n_effective - (int(item.spec.continuum_order) + 1) - n_local, 1)
    chi2_order = float(np.sum(evaluated.residual**2))
    return OrderFitResult(
        order=item.order,
        wave_input=spectrum.wave_input,
        wave_fit=spectrum.wave_fit,
        flux_obs=spectrum.flux,
        err=spectrum.err,
        tell_model=telluric,
        flux_corrected=corrected,
        err_corrected=err_corrected,
        continuum=continuum,
        stellar_model=stellar_on_data(item.stellar_model, state.stellar_shift),
        exclusion=item.exclusion,
        relcol={m: float(state.scales[m]) for m in item.spec.molecules},
        relcol_err={m: relcol_err[m] for m in item.spec.molecules if m in relcol_err},
        resolution=float(state.resolution),
        resolution_err=errors.get(prefix + "resolution"),
        lsf_kind=item.spec.lsf_kind,
        lsf_shape=float(state.lsf_shape) if state.lsf_shape is not None else None,
        lsf_shape_err=errors.get(prefix + "lsf_shape"),
        wave_shift=_shift(state),
        wave_shift_err=_shift_err(item.spec, errors, prefix),
        stellar_shift=state.stellar_shift,
        stellar_shift_err=errors.get("stellar_shift") if item.spec.fit_stellar_shift else None,
        stellar_treatment=str(stellar_treatment),
        converged=converged,
        n_eval=n_eval,
        chi2=chi2_order,
        chi2_reduced=float(chi2_reduced_joint),
        dof_mask_removed=int(dof_joint),
        chi2_order=chi2_order,
        chi2_reduced_order=float(chi2_order / dof_order),
        dof_mask_removed_order=int(dof_order),
        chi2_joint=float(chi2_joint),
        chi2_reduced_joint=float(chi2_reduced_joint),
        dof_mask_removed_joint=int(dof_joint),
        status="joint_ok" if converged else f"joint_failed:{message}",
    )
