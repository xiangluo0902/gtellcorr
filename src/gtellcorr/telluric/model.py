"""The telluric forward model: verified LBLRTM resources + optical-depth cache.

:class:`TelluricModel` is the explicit object behind every optical-depth
calculation. It holds everything a run binds for the order caches
(:func:`~gtellcorr.telluric.order_cache.use_lblrtm_forward_context`): the verified
resource bundle, the tau-cache directory, the timeout and wavelength margin, and
the validated MIPAS reference atmosphere that supplies the eleven fixed molecules.
"""

from __future__ import annotations

import errno
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gtellcorr.errors import GtellcorrError, ResourceError
from gtellcorr.settings import Settings
from gtellcorr.telluric import order_cache as _oc
from gtellcorr.telluric.config import ContinuumMode, TelluricConfig
from gtellcorr.telluric.geometry import Geometry

if TYPE_CHECKING:
    from gtellcorr.data.atmosphere.config import AtmosphereConfig
    from gtellcorr.telluric.resources import VerifiedResources

__all__ = ["LBLRTM_PROFILE_FIELDS", "TelluricModel"]

_STEP1_FIELDS = ("z_km", "P_hPa", "T_K", "H2O_ppmv")
#: Fields of a complete 12-molecule LBLRTM profile.
LBLRTM_PROFILE_FIELDS = (
    *_STEP1_FIELDS,
    "CO2_ppmv", "O3_ppmv", "N2O_ppmv", "CO_ppmv", "CH4_ppmv",
    "O2_ppmv", "NO_ppmv", "SO2_ppmv", "NO2_ppmv", "NH3_ppmv", "HNO3_ppmv",
)  # fmt: skip


def _innermost(exc: BaseException) -> str:
    """Message of the exception that started a chain of re-raised ones."""
    while exc.__cause__ is not None or exc.__context__ is not None:
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return str(exc) or type(exc).__name__


def _workers_hint(exc: BaseException, workers: int | None) -> str:
    """Advice when an LBLRTM run timed out or its work directory filled the disk,
    which several runs at the same time make more likely."""
    if workers == 1:
        return ""
    chain: list[BaseException] = []
    cause: BaseException | None = exc
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    if any(isinstance(e, subprocess.TimeoutExpired) for e in chain):
        problem = "share the CPUs (each run is slower)"
    elif any(isinstance(e, OSError) and e.errno == errno.ENOSPC for e in chain):
        problem = "each need about 1 GB in TMPDIR"
    else:
        return ""
    return (
        f" LBLRTM runs made at the same time {problem}: set lblrtm_workers in the "
        f"settings (or GTELLCORR_LBLRTM_WORKERS) to a smaller number, raise "
        f"lblrtm_timeout_s, or point TMPDIR to a larger disk."
    )


class TelluricModel:
    """LBLRTM optical depths for fitting and forward modelling.

    Parameters
    ----------
    resources
        From :func:`gtellcorr.telluric.resources.verify`.
    reference_atmosphere
        The validated MIPAS reference
        (:func:`gtellcorr.data.atmosphere.reference_atmosphere`).
    tau_cache
        Directory of the optical-depth cache (``Settings.tau_cache``).
    config
        Wavelength margin around each order.
    timeout_s
        Wall-clock limit of one LBLRTM run (``Settings.lblrtm_timeout_s``).
    lblrtm_workers
        LBLRTM runs of a cache made at the same time (``Settings.lblrtm_workers``;
        ``None``: automatic).
    vacuum_air_formula
        Vacuum-to-air formula of the cache grid; use that of the observations
        (``frames.vacuum_air_formula``).
    verbose
        Echo the raw LBLRTM output.
    """

    def __init__(
        self,
        resources: VerifiedResources,
        reference_atmosphere: Mapping[str, Any],
        *,
        tau_cache: str | Path,
        config: TelluricConfig | None = None,
        timeout_s: float = 600.0,
        lblrtm_workers: int | None = None,
        verbose: bool = False,
        vacuum_air_formula: str = "espresso",
    ) -> None:
        self.resources = resources
        self.reference_atmosphere = reference_atmosphere
        self.tau_cache = Path(tau_cache).expanduser().resolve()
        self.config = config or TelluricConfig()
        self.timeout_s = float(timeout_s)
        self.lblrtm_workers = lblrtm_workers
        self.verbose = bool(verbose)
        self.vacuum_air_formula = str(vacuum_air_formula)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        config: TelluricConfig | None = None,
        *,
        atmosphere: AtmosphereConfig | None = None,
        verbose: bool = False,
        vacuum_air_formula: str = "espresso",
    ) -> TelluricModel:
        """Verify the configured bundle and load the reference atmosphere."""
        from gtellcorr.data.atmosphere import reference_atmosphere
        from gtellcorr.telluric.resources import verify

        s = settings or Settings.load()
        assert s.tau_cache is not None  # resolved by Settings
        return cls(
            verify(s),
            reference_atmosphere(atmosphere, s),
            tau_cache=s.tau_cache,
            config=config,
            timeout_s=s.lblrtm_timeout_s,
            lblrtm_workers=s.lblrtm_workers,
            verbose=verbose,
            vacuum_air_formula=vacuum_air_formula,
        )

    def __repr__(self) -> str:
        return (
            f"TelluricModel(bundle={str(self.resources.bundle)!r}, "
            f"tau_cache={str(self.tau_cache)!r}, wave_margin_nm={self.config.wave_margin_nm}, "
            f"vacuum_air_formula={self.vacuum_air_formula!r})"
        )

    def with_continuum_mode(self, continuum_mode: ContinuumMode) -> TelluricModel:
        """This model with another ``config.continuum_mode``: same LBLRTM bundle,
        reference atmosphere, cache directory and run settings."""
        return TelluricModel(
            self.resources,
            self.reference_atmosphere,
            tau_cache=self.tau_cache,
            config=replace(self.config, continuum_mode=continuum_mode),
            timeout_s=self.timeout_s,
            lblrtm_workers=self.lblrtm_workers,
            verbose=self.verbose,
            vacuum_air_formula=self.vacuum_air_formula,
        )

    # -- profile -------------------------------------------------------------
    def lblrtm_profile(self, full_profile: Mapping[str, Any]) -> dict[str, Any]:
        """Add the eleven fixed molecules (from the reference atmosphere) to a full profile.

        The input ``z_km``, ``P_hPa``, ``T_K`` and ``H2O_ppmv`` arrays are checked to be
        unchanged byte for byte.
        """
        from gtellcorr.data.atmosphere.full_profile import add_reference_molecular_profiles

        before = {}
        for name in _STEP1_FIELDS:
            values = np.asarray(full_profile[name])
            before[name] = (values.shape, values.dtype.str, values.tobytes(order="C"))
        complete = add_reference_molecular_profiles(dict(full_profile), self.reference_atmosphere)
        if set(complete) != set(LBLRTM_PROFILE_FIELDS):
            missing = sorted(set(LBLRTM_PROFILE_FIELDS) - set(complete))
            unexpected = sorted(set(complete) - set(LBLRTM_PROFILE_FIELDS))
            raise ValueError(
                "the completed profile does not hold the fields of the 12 LBLRTM molecules; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, expected in before.items():
            values = np.asarray(complete[name])
            if (values.shape, values.dtype.str, values.tobytes(order="C")) != expected:
                raise ValueError(f"fixed-molecule completion changed the {name} bytes")
        return dict(complete)

    # -- optical-depth cache -------------------------------------------------
    @contextmanager
    def activate(self) -> Iterator[TelluricModel]:
        """Bind this model for the order-cache functions inside the block."""
        with _oc.use_lblrtm_forward_context(
            resources=self.resources.resources,
            cache_directory=self.tau_cache,
            timeout_s=self.timeout_s,
            wave_margin_nm=self.config.wave_margin_nm,
            lblrtm_verbose=self.verbose,
            reference_atmosphere=self.reference_atmosphere,
            vacuum_air_formula=self.vacuum_air_formula,
            continuum_mode=self.config.continuum_mode,
            lblrtm_workers=self.lblrtm_workers,
        ):
            yield self

    def build_order_cache(
        self,
        profile: Mapping[str, Any],
        wave_range_nm: tuple[float, float],
        *,
        geometry: Geometry,
        fit_molecules: Sequence[str],
    ) -> _oc.OrderCache:
        """Build (one LBLRTM run per molecule) or reuse the optical-depth cache of one order.

        ``profile`` is a complete LBLRTM profile (:meth:`lblrtm_profile`);
        ``wave_range_nm`` the order's air wavelength range, padded by
        ``config.wave_margin_nm`` on each side. Raises
        :class:`~gtellcorr.errors.ResourceError` if the optical depths can be neither
        computed nor read; the message ends with the original reason.
        """
        try:
            with self.activate():
                return _oc.build_order_cache(
                    dict(profile),
                    wave_range_nm,
                    geometry=geometry,
                    fit_molecules=list(fit_molecules),
                )
        except GtellcorrError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            lo, hi = (float(w) for w in wave_range_nm)
            raise ResourceError(
                f"LBLRTM optical depths for {lo:.2f}-{hi:.2f} nm could not be computed or "
                f"read. Check the LBLRTM bundle with 'gtellcorr resources verify'; if the "
                f"optical-depth cache in {self.tau_cache} is damaged, delete it and run "
                f"again.{_workers_hint(exc, self.lblrtm_workers)} Reason: {_innermost(exc)}"
            ) from exc

    def order_cache_identity(
        self,
        profile: Mapping[str, Any],
        wave_range_nm: tuple[float, float],
        *,
        geometry: Geometry,
        fit_molecules: Sequence[str],
    ) -> Any:
        """``(cache_key, metadata)`` of an order cache, without running LBLRTM."""
        with self.activate():
            return _oc.expected_order_cache_identity(
                dict(profile),
                wave_range_nm,
                geometry=geometry,
                fit_molecules=list(fit_molecules),
            )
