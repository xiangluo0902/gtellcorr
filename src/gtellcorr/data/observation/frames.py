"""Wavelength frames: vacuum <-> air conversion and the air + topocentric fit frame.

The telluric model lives in ONE frame: AIR wavelengths in the TOPOCENTRIC
(observatory) rest frame. Velocity bookkeeping (v > 0 = recession; non-relativistic)::

    observer --BERV-- SSB --Vsys-- stellar system
    topocentric -> barycentric : lambda *= (1 + BERV/c)
    barycentric -> stellar rest: lambda /= (1 + Vsys/c)

Delivered -> topocentric undoes whatever the reduction applied; a *combined*
removed velocity (e.g. PEPSI ``SSTVEL``) excludes the separate BERV/Vsys flags so
nothing is counted twice. Velocity shifts are applied in vacuum.

References: Edlen 1953, JOSA 43, 339; Birch & Downs 1994, Metrologia 31, 315;
Edlen 1966, Metrologia 2, 71; Ciddor 1996, Appl. Opt. 35, 1566.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gtellcorr.config.base import ConfigGroup
from gtellcorr.constants import C_KMS
from gtellcorr.errors import DataError

__all__ = [
    "FrameSpec",
    "VacuumAirFormula",
    "air_to_vac",
    "from_air_topocentric",
    "n_air",
    "n_air_ciddor1996",
    "n_air_edlen1966",
    "n_air_espresso",
    "to_air_topocentric",
    "vac_to_air",
]

VacuumAirFormula = Literal["espresso", "edlen1966", "ciddor1996"]
Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Refractive index of air
# ---------------------------------------------------------------------------
def n_air_espresso(wave_nm: ArrayLike, T_C: float = 15.0, P_mmHg: float = 760.0) -> Array:
    """ESPRESSO-pipeline formula: Edlen 1953 refractivity + Birch & Downs 1994 P/T term.

    Source: ``espdr-kit-3.3.19-3/espdr/espdr_utils.c:2867`` (``espdr_compute_n_air``).
    """
    lambda_A = np.asarray(wave_nm, float) * 10.0
    t = float(T_C)
    p = float(P_mmHg)
    n1 = 1e-6 * p * (1.0 + (1.049 - 0.0157 * t) * 1e-6 * p) / 720.883
    n2 = 1.0 + 0.003661 * t
    sig2 = (10000.0 / lambda_A) ** 2
    n3 = 64.328 + 29498.1 / (146.0 - sig2) + 255.4 / (41.0 - sig2)
    result: Array = (n1 / n2) * n3 + 1.0
    return result


def n_air_edlen1966(wave_nm: ArrayLike) -> Array:
    """Edlen 1966 standard-air formula (molecfit, ``mf_spectrum.c:1442``)."""
    sig2 = 1.0 / (np.asarray(wave_nm, float) / 1000.0) ** 2
    result: Array = 1.0 + 1e-8 * (8342.13 + 2406030.0 / (130.0 - sig2) + 15997.0 / (38.9 - sig2))
    return result


def n_air_ciddor1996(
    wave_nm: ArrayLike,
    T_C: float = 15.0,
    P_Pa: float = 101325.0,
    h2o_partial_Pa: float = 0.0,
    CO2_ppm: float = 450.0,
) -> Array:
    """Ciddor 1996 real-air formula (valid ~230-1700 nm), BIPM-81 compressibility."""
    sig2 = (1000.0 / np.asarray(wave_nm, float)) ** 2

    k0, k1, k2, k3 = 238.0185, 5792105.0, 57.362, 167917.0
    naxs_m1 = 1e-8 * (k1 / (k0 - sig2) + k3 / (k2 - sig2))
    naxs_m1 = naxs_m1 * (1.0 + 0.534e-6 * (CO2_ppm - 450.0))

    w0, w1, w2, w3 = 295.235, 2.6422, -0.032380, 0.004028
    nws_m1 = 1e-8 * 1.022 * (w0 + w1 * sig2 + w2 * sig2**2 + w3 * sig2**3)

    a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
    b0, b1 = 5.707e-6, -2.051e-8
    c0, c1 = 1.9898e-4, -2.376e-6
    d_const = 1.83e-11
    e_const = -0.765e-8

    def _Z(P_: float, T_K_: float, t_: float, x_: float) -> float:
        return (
            1.0
            - (P_ / T_K_)
            * (a0 + a1 * t_ + a2 * t_**2 + (b0 + b1 * t_) * x_ + (c0 + c1 * t_) * x_**2)
            + (P_ / T_K_) ** 2 * (d_const + e_const * x_**2)
        )

    R = 8.314510
    M_w = 0.018015
    M_a = 1e-3 * (28.9635 + 12.011e-6 * (CO2_ppm - 400.0))

    T_K = T_C + 273.15
    x_w = h2o_partial_Pa / P_Pa if P_Pa > 0 else 0.0
    Z = _Z(P_Pa, T_K, T_C, x_w)
    rho_a = P_Pa * M_a * (1.0 - x_w * (1.0 - M_w / M_a)) / (Z * R * T_K)
    rho_w = P_Pa * M_w * x_w / (Z * R * T_K)

    T_K_axs = 288.15
    Z_axs = _Z(101325.0, T_K_axs, 15.0, 0.0)
    rho_axs = 101325.0 * M_a / (Z_axs * R * T_K_axs)

    T_K_ws = 293.15
    Z_ws = _Z(1333.0, T_K_ws, 20.0, 1.0)
    rho_ws = 1333.0 * M_w / (Z_ws * R * T_K_ws)

    n_minus_1 = (rho_a / rho_axs) * naxs_m1 + (rho_w / rho_ws) * nws_m1
    result: Array = 1.0 + n_minus_1
    return result


_FORMULAS: Mapping[str, Callable[..., Array]] = {
    "espresso": n_air_espresso,
    "edlen1966": n_air_edlen1966,
    "ciddor1996": n_air_ciddor1996,
}


def n_air(wave_nm: ArrayLike, formula: VacuumAirFormula = "espresso", **kwargs: Any) -> Array:
    """Refractive index of air with the chosen *formula* (kwargs go to the formula)."""
    func = _FORMULAS.get(str(formula).strip().lower())
    if func is None:
        raise DataError(
            f"unknown vacuum/air formula {formula!r}; available: {', '.join(sorted(_FORMULAS))}"
        )
    return func(wave_nm, **kwargs)


def vac_to_air(
    wave_vac_nm: ArrayLike, formula: VacuumAirFormula = "espresso", **kwargs: Any
) -> Array:
    """Vacuum -> air: lambda_air = lambda_vac / n(lambda_vac)."""
    w = np.asarray(wave_vac_nm, float)
    result: Array = w / n_air(w, formula, **kwargs)
    return result


def air_to_vac(
    wave_air_nm: ArrayLike, formula: VacuumAirFormula = "espresso", **kwargs: Any
) -> Array:
    """Air -> vacuum, one-step inverse: lambda_vac ~= lambda_air * n(lambda_air).

    Round-trip error O((n-1)^2), about 0.5 m/s at 720 nm.
    """
    w = np.asarray(wave_air_nm, float)
    result: Array = w * n_air(w, formula, **kwargs)
    return result


# ---------------------------------------------------------------------------
# Delivered-frame declaration
# ---------------------------------------------------------------------------
Velocity = str | float
"""A velocity given as a number or as the name of a FITS header keyword."""


@dataclass(frozen=True, kw_only=True)
class FrameSpec(ConfigGroup):
    """How an instrument delivers its wavelengths: medium and removed velocities."""

    wavelength_medium: Literal["air", "vacuum"]
    """Medium of the delivered wavelengths."""
    removed_velocity_mps: tuple[Velocity, ...] = ()
    """Combined velocities [m/s] removed by the reduction (numbers or header keys);
    if given, the BERV/Vsys flags below are ignored."""
    berv_corrected: bool = False
    """The reduction moved the wavelengths to the barycentric frame (BERV removed)."""
    berv_kms: Velocity | None = None
    """BERV [km/s] or the header keyword holding it."""
    vsys_corrected: bool = False
    """The reduction also moved the wavelengths to the stellar rest frame (Vsys removed)."""
    vsys_kms: Velocity | None = None
    """Systemic velocity [km/s] or the header keyword holding it."""

    def validate(self) -> None:
        """A removed BERV or Vsys needs its value (or header keyword)."""
        if self.berv_corrected and self.berv_kms is None:
            self.fail("berv_kms", "required when berv_corrected is true")
        if self.vsys_corrected and self.vsys_kms is None:
            self.fail("vsys_kms", "required when vsys_corrected is true")


def _resolve_velocity(value: Velocity | None, header: Any, what: str) -> float:
    """A velocity given as a number, or read from the header keyword it names."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        if header is None:
            raise DataError(f"{what}: header keyword {value!r} given but no FITS header supplied")
        if value not in header:
            raise DataError(f"{what}: keyword {value!r} not in FITS header")
        return float(header[value])
    raise DataError(f"{what}: expected a number or a header keyword, got {value!r}")


def _removed_velocity_mps(frame: FrameSpec, header: Any) -> float:
    """Sum of the combined velocities the reduction removed [m/s]."""
    total = 0.0
    for value in frame.removed_velocity_mps:
        total += _resolve_velocity(value, header, "removed_velocity_mps")
    return total


def to_air_topocentric(
    wave_nm: ArrayLike,
    frame: FrameSpec,
    header: Any = None,
    *,
    formula: VacuumAirFormula = "espresso",
) -> Array:
    """Delivered wavelengths [nm] -> air + topocentric (the telluric-model frame)."""
    w = np.asarray(wave_nm, float)
    if frame.wavelength_medium == "air":
        w = air_to_vac(w, formula)
    if frame.removed_velocity_mps:
        v_mps = _removed_velocity_mps(frame, header)
        return vac_to_air(w * (1.0 + v_mps / 1000.0 / C_KMS), formula)
    if frame.vsys_corrected:
        w = w * (1.0 + _resolve_velocity(frame.vsys_kms, header, "Vsys") / C_KMS)
    if frame.berv_corrected:
        w = w / (1.0 + _resolve_velocity(frame.berv_kms, header, "BERV") / C_KMS)
    return vac_to_air(w, formula)


def from_air_topocentric(
    wave_nm: ArrayLike,
    frame: FrameSpec,
    header: Any = None,
    *,
    formula: VacuumAirFormula = "espresso",
) -> Array:
    """Inverse of :func:`to_air_topocentric`: air + topocentric -> delivered frame."""
    w = air_to_vac(np.asarray(wave_nm, float), formula)
    if frame.removed_velocity_mps:
        w = w / (1.0 + _removed_velocity_mps(frame, header) / 1000.0 / C_KMS)
        return vac_to_air(w, formula) if frame.wavelength_medium == "air" else w
    if frame.berv_corrected:
        w = w * (1.0 + _resolve_velocity(frame.berv_kms, header, "BERV") / C_KMS)
    if frame.vsys_corrected:
        w = w / (1.0 + _resolve_velocity(frame.vsys_kms, header, "Vsys") / C_KMS)
    if frame.wavelength_medium == "air":
        return vac_to_air(w, formula)
    return w
