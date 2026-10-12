"""Line-spread function kernels and a convolution at a fixed resolving power.

Convolve the high-resolution telluric transmittance with the instrument LSF at
resolving power R.  Kinds:
  ``gauss``  single Gaussian (R a free ``theta`` element, fitted jointly).
  ``sgauss`` super-Gaussian ``exp(-(|v|/a)^p)`` — the SHAPE exponent ``p`` adds a
             flatter/heavier core+wing than a Gaussian (p=2 is the Gaussian).  A
             real spectrograph LSF is rarely a pure Gaussian; the residual SHAPE
             mismatch (not width, which R already fits) drives the spikes left in
             the telluric-corrected spectrum at deep line cores.

ALL kinds convolve in velocity space (uniform ln-λ), where a constant resolving
power R is a single constant-width kernel — the physically correct treatment on
the wavenumber-uniform LBLRTM grid (a constant-Δλ kernel would drift in velocity
width by ~3% across an order). The gauss kind is the super-Gaussian with p=2.

:func:`apply_lsf` builds its grid with a step of FWHM/10, so R enters only through
the integer number of grid points: fine at a fixed R (the strong-telluric mask),
but a staircase in R. Fits therefore use
:func:`gtellcorr.fitting.lsf_grid.apply_lsf_fixed_grid` with the same kernels.
"""

from __future__ import annotations

import numpy as np

_C_KMS = 299792.458


def _sgauss_kernel(v, fwhm_v, p):
    """Super-Gaussian ``exp(-(|v|/a)^p)`` with FWHM = ``fwhm_v`` (p=2 → Gaussian)."""
    a = fwhm_v / (2.0 * np.log(2.0) ** (1.0 / p))  # FWHM(exp(-(v/a)^p)) = fwhm_v
    return np.exp(-((np.abs(v) / a) ** p))


def _voigt_kernel(v, fwhm_v, eta):
    """Pseudo-Voigt in Whiting-form mixing: a Gaussian core + a Lorentzian-wing
    component, mixed by ``eta`` as a FREE shape parameter (0 → Gaussian, →1 → more
    Lorentzian wings); both components have FWHM = ``fwhm_v``.

    NB this is NOT molecfit's default kernel — molecfit default (kern_mode=0) is a
    SEPARATE Gaussian ⊗ Lorentzian convolution (two independent FWHMs), and its
    alternate Whiting path DERIVES the mix from the component widths rather than
    fitting a free η.  Here η is a single free shape knob at fixed total FWHM."""
    xv = v / fwhm_v
    xv2 = xv * xv
    xv225 = np.abs(xv) ** 2.25
    return (
        (1.0 - eta) * np.exp(-2.772 * xv2)  # 2.772 = 4 ln2 → FWHM = fwhm_v
        + eta / (1.0 + 4.0 * xv2)
        + 0.016 * (1.0 - eta) * eta * (np.exp(-0.4 * xv225) - 10.0 / (10.0 + xv225))
    )


def _velocity_convolve(
    wave_nm, trans, resolution, kern_fn, shape, *, oversample=10.0, half_fwhm=6.0
):
    """Convolve ``trans`` with a velocity-space LSF ``kern_fn(v, fwhm_v, shape)`` of
    FWHM ``c/resolution``.  Done on a uniform ln-λ (velocity) grid so a constant R
    is a single constant-width kernel; edges are edge-padded (transmittance≈1)."""
    lo, hi = float(wave_nm[0]), float(wave_nm[-1])
    fwhm_v = _C_KMS / resolution
    dv = fwhm_v / oversample
    n = max(int(np.log(hi / lo) * _C_KMS / dv) + 1, 2)
    wv = np.exp(np.linspace(np.log(lo), np.log(hi), n))  # uniform in ln-λ = in velocity
    tv = np.interp(wv, wave_nm, trans)
    half = max(int(half_fwhm * fwhm_v / dv), 1)
    kv = np.arange(-half, half + 1) * dv
    kern = kern_fn(kv, fwhm_v, shape)
    kern /= kern.sum()
    conv = np.convolve(np.pad(tv, half, mode="edge"), kern, mode="valid")
    return np.interp(wave_nm, wv, conv)


def apply_lsf(
    wave_nm: np.ndarray,
    trans: np.ndarray,
    resolution: float,
    *,
    kind: str = "gauss",
    shape: float = 2.0,
) -> np.ndarray:
    """Convolve ``trans`` with the instrument LSF at resolving power
    ``resolution`` (R = λ/Δλ).  Returns the broadened transmittance on the same
    ``wave_nm`` grid.  ``kind`` and its ``shape`` parameter:
      'gauss'  — single Gaussian (shape ignored).
      'sgauss' — super-Gaussian, ``shape`` = exponent p (p=2 ≈ Gaussian).
      'voigt'  — pseudo-Voigt (Gaussian core + Lorentzian wings), ``shape`` = the
                 Lorentzian weight η ∈ [0, 1) (η=0 ≈ Gaussian).
    """
    wave_nm = np.asarray(wave_nm, float)
    trans = np.asarray(trans, float)
    if kind == "gauss":
        # constant-R Gaussian in velocity space (= super-Gaussian p=2)
        return _velocity_convolve(
            wave_nm, trans, float(resolution), _sgauss_kernel, 2.0, half_fwhm=10.0
        )
    if kind == "sgauss":
        if not float(shape) > 0.0:
            raise ValueError(f"sgauss exponent p must be > 0, got {shape!r}")
        # super-Gaussian wings decay fast; 10×FWHM covers p down to ~1.3
        return _velocity_convolve(
            wave_nm, trans, float(resolution), _sgauss_kernel, float(shape), half_fwhm=10.0
        )
    if kind == "voigt":
        if not (0.0 <= float(shape) < 1.0):
            raise ValueError(f"voigt Lorentzian weight η must be in [0, 1), got {shape!r}")
        # Lorentzian wings decay ~1/v², so truncate FAR out (30×FWHM) to keep them
        return _velocity_convolve(
            wave_nm, trans, float(resolution), _voigt_kernel, float(shape), half_fwhm=30.0
        )
    raise ValueError(f"unknown lsf kind {kind!r}; use 'gauss', 'sgauss' or 'voigt'")
