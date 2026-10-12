"""LSF convolution on a fixed velocity grid: the convolution of the fit.

The fixed-R convolution (:func:`gtellcorr.fitting.lsf.apply_lsf`) builds its
velocity grid with a step of FWHM/10. The kernel is then the same *in samples* for
every R, and R only enters through the integer number of grid points: the model is
a staircase in R, so a finite-difference Jacobian sees no dependence on R and the
resolving power is effectively not fitted.

Here the grid step and the kernel support are fixed by ``grid_resolution`` and
``support_resolution`` (the upper and lower R bounds of a fit), and the kernel is
evaluated at physical velocities for the requested R. For fixed grid parameters
the result is a smooth function of R and of the shape parameter. The kernels are
those of :mod:`gtellcorr.fitting.lsf` (same FWHM definition, same truncation in
FWHM units at the widest R); the kernel velocities use the grid's actual step.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gtellcorr.fitting.lsf import _C_KMS, _sgauss_kernel, _voigt_kernel

__all__ = ["DEFAULT_OVERSAMPLE", "apply_lsf_fixed_grid"]

Array = NDArray[np.float64]
Kernel = Callable[[Array, float, float], Array]

#: Grid points per FWHM at ``grid_resolution``.
DEFAULT_OVERSAMPLE = 10.0

#: Kernel and truncation half-width in FWHM, per kind.
_KINDS: dict[str, tuple[Kernel, float]] = {
    "gauss": (_sgauss_kernel, 10.0),
    "sgauss": (_sgauss_kernel, 10.0),
    "voigt": (_voigt_kernel, 30.0),
}


def _check_shape(kind: str, shape: float) -> float:
    """The shape parameter of *kind* (2 for ``gauss``), checked."""
    if kind == "gauss":
        return 2.0
    if kind == "sgauss":
        if not float(shape) > 0.0:
            raise ValueError(f"sgauss exponent p must be > 0, got {shape!r}")
        return float(shape)
    if kind == "voigt":
        if not 0.0 <= float(shape) < 1.0:
            raise ValueError(f"voigt Lorentzian weight eta must be in [0, 1), got {shape!r}")
        return float(shape)
    raise ValueError(f"unknown lsf kind {kind!r}; use 'gauss', 'sgauss' or 'voigt'")


def apply_lsf_fixed_grid(
    wave_nm: ArrayLike,
    trans: ArrayLike,
    resolution: float,
    *,
    kind: str = "gauss",
    shape: float = 2.0,
    grid_resolution: float | None = None,
    support_resolution: float | None = None,
    oversample: float = DEFAULT_OVERSAMPLE,
) -> Array:
    """Convolve ``trans`` with the LSF of resolving power ``resolution``.

    Parameters
    ----------
    wave_nm, trans
        Increasing wavelengths [nm] and the transmission on them.
    resolution
        R = lambda / FWHM.
    kind, shape
        ``gauss``; ``sgauss`` (shape = exponent p); ``voigt`` (shape = Lorentzian
        weight eta in [0, 1)).
    grid_resolution
        R that sets the grid step (``oversample`` points per FWHM at this R). Use the
        upper R bound of a fit; default ``resolution``.
    support_resolution
        R that sets the kernel support (the kind's truncation width at this R).
        Use the lower R bound of a fit; default ``resolution``.

    Returns the broadened transmission on ``wave_nm``.
    """
    p = _check_shape(kind, shape)
    kernel, half_fwhm = _KINDS[kind]
    wave = np.asarray(wave_nm, float)
    t = np.asarray(trans, float)
    r = float(resolution)
    r_grid = r if grid_resolution is None else float(grid_resolution)
    r_support = r if support_resolution is None else float(support_resolution)
    if not (r > 0 and r_grid > 0 and r_support > 0):
        raise ValueError("resolving powers must be positive")

    lo, hi = float(wave[0]), float(wave[-1])
    span_v = math.log(hi / lo) * _C_KMS
    n = max(int(span_v / (_C_KMS / r_grid / float(oversample))) + 1, 2)
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    dv = span_v / (n - 1)  # actual step of the grid
    on_grid = np.interp(grid, wave, t)
    half = max(math.ceil(half_fwhm * _C_KMS / r_support / dv), 1)
    velocities = np.asarray(np.arange(-half, half + 1) * dv, dtype=np.float64)
    weights = kernel(velocities, _C_KMS / r, p)
    weights = weights / weights.sum()
    convolved = np.convolve(np.pad(on_grid, half, mode="edge"), weights, mode="valid")
    return np.asarray(np.interp(wave, grid, convolved), float)
