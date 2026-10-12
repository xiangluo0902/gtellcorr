"""Residual, theta layout and optimizer of the telluric fit.

Theta is assembled from the parameter blocks (:mod:`gtellcorr.fitting.blocks`): a
single-order fit concatenates the active blocks (:func:`pack` / :func:`unpack`), a
joint fit puts the exposure-scope values first and then each order's
(:func:`joint_layout`).

Per theta evaluation: transmission from the optical-depth cache, LSF
convolution, interpolation to the data grid, then the continuum is projected out
(variable projection) and the weighted residual returned. The fitted LSF acts
on the telluric transmission only: the model is ``C * M * (K_R conv T)``, not
``K_R conv (M * T)`` (the stellar template keeps its preprocessing resolution). A
fitted wavelength offset (:mod:`gtellcorr.fitting.wave_shift`) likewise moves only
``K_R conv T`` before it is interpolated to the data grid.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from gtellcorr.fitting.blocks import ModelState, ParameterBlock, default_blocks
from gtellcorr.fitting.continuum import fit_continuum
from gtellcorr.fitting.fit_spec import FitSpec
from gtellcorr.fitting.lsf_grid import apply_lsf_fixed_grid
from gtellcorr.fitting.wave_shift import shifted_grid
from gtellcorr.telluric.order_cache import OrderCache, transmittance

__all__ = [
    "JointLayout",
    "OptResult",
    "OrderEvaluation",
    "SpectrumLike",
    "build_residual",
    "evaluate_order",
    "inverse_variance",
    "joint_layout",
    "minimize",
    "pack",
    "theta_scale",
    "unpack",
]

Array = NDArray[np.float64]


class SpectrumLike(Protocol):
    """What the engine reads from a spectrum (e.g. :class:`~gtellcorr.data.OrderSpectrum`)."""

    @property
    def wave_fit(self) -> Array:
        """Air + topocentric wavelengths [nm]."""
        ...

    @property
    def wave_input(self) -> Array:
        """Delivered wavelengths [nm] of the same pixels."""
        ...

    @property
    def flux(self) -> Array:
        """Flux."""
        ...

    @property
    def err(self) -> Array:
        """1-sigma flux uncertainty."""
        ...


def _blocks(blocks: Sequence[ParameterBlock] | None) -> Sequence[ParameterBlock]:
    """*blocks*, or the built-in ones."""
    return default_blocks() if blocks is None else blocks


# ---------------------------------------------------------------------------
# theta layout
# ---------------------------------------------------------------------------
def pack(
    spec: FitSpec, blocks: Sequence[ParameterBlock] | None = None
) -> tuple[Array, Array, Array, list[str]]:
    """Fit spec -> ``(theta0, lo, hi, names)`` of a single-order fit."""
    theta0: list[Array] = []
    lo: list[Array] = []
    hi: list[Array] = []
    names: list[str] = []
    for block in _blocks(blocks):
        if not block.active(spec):
            continue
        theta0.append(block.initial(spec))
        low, high = block.bounds(spec)
        lo.append(low)
        hi.append(high)
        names.extend(block.names(spec))
    if not names:
        empty = np.empty(0, float)
        return empty, empty.copy(), empty.copy(), names
    return np.concatenate(theta0), np.concatenate(lo), np.concatenate(hi), names


def unpack(
    theta: Array, spec: FitSpec, blocks: Sequence[ParameterBlock] | None = None
) -> ModelState:
    """Inverse of :func:`pack`: theta -> the order's model state."""
    state = ModelState.initial(spec)
    index = 0
    for block in _blocks(blocks):
        if not block.active(spec):
            continue
        size = len(block.names(spec))
        state = block.apply(np.asarray(theta[index : index + size], float), spec, state)
        index += size
    return state


@dataclass(frozen=True)
class JointLayout:
    """Theta of a joint fit: shared (exposure-scope) values, then per-order values."""

    theta0: Array
    lower: Array
    upper: Array
    names: list[str]
    """Block names for shared values (``relcol_h2o``), ``order<NNN>_<name>`` per
    order (``order084_resolution``)."""
    specs: Mapping[int, FitSpec]
    """Fit spec per order, in fit order."""
    blocks: Sequence[ParameterBlock]
    """The blocks theta is built from."""

    def states(self, theta: Array) -> dict[int, ModelState]:
        """Model state of every order for *theta*."""
        values = dict(zip(self.names, (float(v) for v in theta), strict=True))
        states = {}
        for order, spec in self.specs.items():
            state = ModelState.initial(spec)
            for block in self.blocks:
                if not block.active(spec):
                    continue
                prefix = "" if block.scope == "exposure" else f"order{order:03d}_"
                picked = [values[prefix + name] for name in block.names(spec)]
                state = block.apply(np.asarray(picked, float), spec, state)
            states[order] = state
        return states


def joint_layout(
    specs: Mapping[int, FitSpec], blocks: Sequence[ParameterBlock] | None = None
) -> JointLayout:
    """Theta of a joint fit over *specs* (order -> spec, in fit order).

    A shared parameter starts at the median of its per-order initial values and is
    bounded by the intersection of the per-order bounds (``ValueError`` if they do
    not overlap).
    """
    chosen = _blocks(blocks)
    theta0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for block in chosen:
        if block.scope != "exposure":
            continue
        shared: dict[str, list[tuple[float, float, float]]] = {}
        for spec in specs.values():
            if not block.active(spec):
                continue
            lo, hi = block.bounds(spec)
            for name, v0, a, b in zip(block.names(spec), block.initial(spec), lo, hi, strict=True):
                shared.setdefault(name, []).append((float(v0), float(a), float(b)))
        for name, entries in shared.items():
            lo_ = float(max(e[1] for e in entries))
            hi_ = float(min(e[2] for e in entries))
            if lo_ > hi_:
                raise ValueError(f"joint bounds do not overlap for {name!r}: {lo_} > {hi_}")
            theta0.append(float(np.median([e[0] for e in entries])))
            lower.append(lo_)
            upper.append(hi_)
            names.append(name)
    for order, spec in specs.items():
        for block in chosen:
            if block.scope != "order" or not block.active(spec):
                continue
            lo, hi = block.bounds(spec)
            for name, v0, a, b in zip(block.names(spec), block.initial(spec), lo, hi, strict=True):
                theta0.append(float(v0))
                lower.append(float(a))
                upper.append(float(b))
                names.append(f"order{order:03d}_{name}")
    return JointLayout(
        theta0=np.asarray(theta0, float),
        lower=np.asarray(lower, float),
        upper=np.asarray(upper, float),
        names=names,
        specs=dict(specs),
        blocks=tuple(chosen),
    )


# ---------------------------------------------------------------------------
# model evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderEvaluation:
    """Forward model and residual of one order for one model state."""

    telluric_on_data: Array
    """LSF-convolved transmission on the data grid."""
    multiplicative_model: Array
    """``M * T`` (or ``T`` without a stellar model)."""
    continuum: Array
    """Projected continuum ``C``."""
    residual: Array
    """``(flux - C * model) * sqrt(weight)``, zero on excluded pixels."""


def inverse_variance(err: Any) -> Array:
    """``1 / err**2`` where the error is finite and positive, else 0."""
    err = np.asarray(err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(err) & (err > 0), 1.0 / err**2, 0.0)


def evaluate_order(
    spectrum: SpectrumLike,
    cache: OrderCache,
    spec: FitSpec,
    state: ModelState,
    *,
    weight: Any,
    keep_mask: Any,
    stellar_model: Any | None = None,
) -> OrderEvaluation:
    """Model, projected continuum and weighted residual of one order.

    ``keep_mask`` is True for pixels used in the fit; ``weight`` is the inverse
    variance on ``spectrum.wave_fit``. The continuum is fitted on the kept pixels
    only; excluded pixels get a zero residual.
    """
    x = np.asarray(spectrum.wave_fit, float)
    y = np.asarray(spectrum.flux, float)
    weight = np.asarray(weight, float)
    keep = np.asarray(keep_mask, bool)
    stellar = None if stellar_model is None else np.asarray(stellar_model, float)

    shape = state.lsf_shape if state.lsf_shape is not None else 2.0
    wl, trans = transmittance(cache, dict(state.scales))
    grid_r, support_r = spec.lsf_grid()
    convolved = apply_lsf_fixed_grid(
        wl,
        trans,
        float(state.resolution),
        kind=spec.lsf_kind,
        shape=shape,
        grid_resolution=grid_r,
        support_resolution=support_r,
    )
    at = x if state.wave_shift is None else shifted_grid(x, state.wave_shift)
    on_data: Array = np.asarray(np.interp(at, wl, convolved), float)
    model = on_data if stellar is None else stellar * on_data
    continuum = fit_continuum(
        x,
        y,
        model,
        order=int(spec.continuum_order),
        sigma_clip=(float(spec.continuum_sigma_clip[0]), float(spec.continuum_sigma_clip[1])),
        max_iter=int(spec.continuum_niter),
        mask=keep,
        weight=weight,
    )
    sqrt_w = np.sqrt(np.clip(weight, 0.0, None))
    residual = (y - continuum * model) * sqrt_w
    residual = np.where(keep & np.isfinite(residual), residual, 0.0)
    return OrderEvaluation(
        telluric_on_data=on_data,
        multiplicative_model=model,
        continuum=continuum,
        residual=residual,
    )


def build_residual(
    spectrum: SpectrumLike,
    cache: OrderCache,
    spec: FitSpec,
    *,
    weight: Any | None = None,
    mask: Any | None = None,
    stellar_model: Any | None = None,
    blocks: Sequence[ParameterBlock] | None = None,
) -> Callable[[Array], Array]:
    """``residual(theta)`` of a single-order fit.

    ``weight`` defaults to the inverse variance of ``spectrum.err``; ``mask`` is a
    keep weight (> 0 = keep), ``None`` keeps every pixel.
    """
    y = np.asarray(spectrum.flux, float)
    w = inverse_variance(spectrum.err) if weight is None else np.asarray(weight, float)
    m = np.ones_like(y) if mask is None else np.asarray(mask, float)
    keep = m > 0
    chosen = _blocks(blocks)

    def residual(theta: Array) -> Array:
        """Weighted residual of the order for *theta*."""
        state = unpack(theta, spec, chosen)
        return evaluate_order(
            spectrum, cache, spec, state, weight=w, keep_mask=keep, stellar_model=stellar_model
        ).residual

    return residual


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    """Outcome of the bounded least-squares solve."""

    theta: Array
    theta_err: Array | None
    """1-sigma from the Gauss-Newton covariance (``None`` if singular)."""
    cost: float
    """``least_squares`` cost (half the chi-square)."""
    chi2: float
    chi2_reduced: float
    """``chi2 / dof_mask_removed``."""
    n_resid: int
    """Length of the residual vector (excluded pixels included)."""
    n_par: int
    """Number of free parameters."""
    dof_mask_removed: int
    """Kept pixels minus free parameters minus continuum coefficients."""
    n_eval: int
    """Residual evaluations."""
    success: bool
    message: str


def theta_scale(theta0: Array) -> Array | float:
    """``x_scale`` for ``least_squares``: ``|theta0|`` (1 where zero)."""
    magnitude = np.abs(np.asarray(theta0, float))
    return np.where(magnitude > 0, magnitude, 1.0)


def covariance_errors(result: Any, dof: int) -> Array | None:
    """1-sigma errors from ``(J^T J)^-1 * chi2 / dof``; ``None`` if not invertible."""
    try:
        jac = result.jac
        chi2 = float(np.sum(result.fun**2))
        cov = np.linalg.inv(jac.T @ jac) * (chi2 / max(int(dof), 1))
        return np.asarray(np.sqrt(np.clip(np.diag(cov), 0.0, None)), float)
    except (np.linalg.LinAlgError, ValueError, TypeError, AttributeError):
        return None


def minimize(
    residual: Callable[[Array], Array],
    theta0: Array,
    *,
    lo: Array,
    hi: Array,
    n_effective: int,
    n_projected: int = 0,
    method: str = "trf",
    x_scale: Array | float = 1.0,
) -> OptResult:
    """Minimise *residual* within ``[lo, hi]`` (``scipy.optimize.least_squares``).

    ``n_effective`` counts the pixels that enter the residual after all masks;
    ``n_projected`` the coefficients solved inside each evaluation (continuum).
    Reduced chi2 and the covariance use ``n_effective - n_par - n_projected``.
    ``x_scale`` is passed to ``least_squares`` (see :func:`theta_scale`).
    """
    result = least_squares(
        residual,
        np.asarray(theta0, float),
        bounds=(np.asarray(lo, float), np.asarray(hi, float)),
        method=method,
        x_scale=x_scale,
    )
    chi2 = float(np.sum(result.fun**2))
    n_par = int(np.asarray(theta0).size)
    dof = max(int(n_effective) - n_par - int(n_projected), 1)
    return OptResult(
        theta=np.asarray(result.x, float),
        theta_err=covariance_errors(result, dof),
        cost=float(result.cost),
        chi2=chi2,
        chi2_reduced=chi2 / dof,
        n_resid=int(result.fun.size),
        n_par=n_par,
        dof_mask_removed=dof,
        n_eval=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
