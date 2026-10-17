"""The telluric-correction pipeline: explicit stages from raw spectra to results.

A run goes through three stages, each of which can also be called on its own:

1. :meth:`Pipeline.prepare` - the run directory layout, the verified LBLRTM
   resources, the stellar products (template, RV table, stellar-line masks) and
   the strong-telluric mask;
2. :meth:`Pipeline.fit_exposure` - for one exposure: the atmospheric profile at
   the time of observation, the spectral orders, the single-order or joint-order
   fit, the correction of every output order and the result FITS file;
3. :meth:`Pipeline.run` - every exposure (an earlier result is reused when its
   inputs match), then ``summary.csv``, the optional telluric PCA, the
   figures and ``run_manifest.json``.

``gtc.run(observation, config)`` runs all three. Results are written below the
current working directory unless ``output_dir`` is given.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gtellcorr._version import __version__
from gtellcorr.data.products.fits_schema import read_provenance, write_exposure_fits
from gtellcorr.data.products.layout import PRODUCT_RV, PRODUCT_TELLURIC, RunLayout, date_token
from gtellcorr.data.products.summary import SUMMARY_COLUMNS, joint_row, single_row, skipped_row
from gtellcorr.errors import ConfigError, DataError, FitError
from gtellcorr.fitting.blocks import ModelState
from gtellcorr.fitting.fit_order import JointOrderInput, fit_joint_orders, fit_single_order
from gtellcorr.fitting.fit_spec import FitSpec, build_fit_spec, select_fit_orders_for_molecules
from gtellcorr.fitting.results import OrderFitResult
from gtellcorr.fitting.solver import evaluate_order, inverse_variance, lines_only
from gtellcorr.fitting.stellar_shift import stellar_on_data
from gtellcorr.settings import Settings
from gtellcorr.telluric.geometry import Geometry
from gtellcorr.telluric.order_cache import OrderCache

if TYPE_CHECKING:
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.observation import Exposure, Observation
    from gtellcorr.data.products.result import Result
    from gtellcorr.preprocessing.prepare import StellarProducts
    from gtellcorr.telluric.model import TelluricModel

__all__ = ["Pipeline", "Progress", "config_hash", "run"]

log = logging.getLogger("gtellcorr")
Reuse = Literal["matching", "any", "never"]

_PRODUCT_NAMES = {"template": "template", "rv": "RV table", "linemask": "line masks"}
_STATUS_WORDS = {"exists": "reused", "user_file": "from the user's file"}


def _describe_status(status: Mapping[str, str]) -> str:
    """``{"template": "reused", "rv": "generated"}`` -> ``template reused, RV table generated``."""
    return ", ".join(
        f"{_PRODUCT_NAMES.get(k, k)} {_STATUS_WORDS.get(v, v)}" for k, v in status.items()
    )


@dataclass(frozen=True)
class Progress:
    """Passed to the progress callback after each stage step."""

    stage: str
    """Stage name: ``prepare``, ``fit`` or ``pca``."""
    message: str
    """What just happened, e.g. ``"<exposure>: fitting"``."""
    done: int = 0
    """Exposures finished so far (``fit`` stage), else 0."""
    total: int = 0
    """Number of exposures in the run (``fit`` stage), else 0."""


def config_hash(config: FitConfig) -> str:
    """SHA-256 of the fit configuration's parameter values.

    The configuration is written as TOML without comments or header, so documentation changes
    leave the hash, and hence product reuse, unaffected.
    """
    from gtellcorr.config.toml_format import dumps_group

    text = dumps_group("fit", config, comments=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _Loaded:
    """Orders of one exposure: fit-capable spectra, raw placeholders, read errors."""

    spectra: dict[int, Any] = field(default_factory=dict)
    raw: dict[int, Any] = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)

    def grids(self) -> dict[int, Any]:
        """The delivered pixel grid of each read order that has one."""
        orders = {**self.raw, **self.spectra}
        return {o: s.grid for o, s in orders.items() if getattr(s, "grid", None) is not None}


class Pipeline:
    """Telluric correction of one :class:`~gtellcorr.data.observation.Observation`.

    Parameters
    ----------
    observation
        Exposures to correct (instrument, target, exposures).
    config
        The fit configuration.
    output_dir
        Root of the run directories (default: the current working directory).
    settings
        Machine-local settings (default: :meth:`Settings.load`).
    reuse
        ``matching`` (default): reuse an exposure's result only if it was written with
        the same gtellcorr version, configuration, LBLRTM resources and input file;
        ``any``: reuse any existing result file; ``never``: always fit again.
    apply_stellar_treatment
        ``False`` skips the stellar front end (no template, RV or stellar-line masks);
        only the strong telluric line cores are excluded from the fit. Useful for
        featureless targets.
    plot
        Write the per-order fit figures, the RV figures and the run-level figures.
    telluric_model
        A ready :class:`~gtellcorr.telluric.model.TelluricModel` with the run's
        ``frames.vacuum_air_formula`` (default: built from
        ``settings`` and the configuration on first use).
    celestrak
        Space-weather reference file for the upper-atmosphere model (default: the
        table in ``<resource_root>/atmosphere``, installed by
        ``gtellcorr resources fetch``).
    rv_seed
        Seed of the MCMC sampler of the RV measurement, so repeated runs give
        identical RVs; ``None`` uses NumPy's global random generator.
    progress
        Optional callback receiving :class:`Progress` events.
    """

    def __init__(
        self,
        observation: Observation,
        config: FitConfig,
        *,
        output_dir: str | Path | None = None,
        settings: Settings | None = None,
        reuse: Reuse = "matching",
        apply_stellar_treatment: bool = True,
        plot: bool = True,
        telluric_model: TelluricModel | None = None,
        celestrak: Any = None,
        rv_seed: int | None = 0,
        progress: Callable[[Progress], None] | None = None,
    ) -> None:
        if not observation.exposures:
            raise DataError("the observation has no exposures")
        if observation.target is None:
            raise DataError(
                "the observation needs a target (at least its name, e.g. gtc.Star(name=...))"
            )
        self.observation = observation
        self.config = config
        self.settings = settings or Settings.load()
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self.reuse = reuse
        self.apply_stellar_treatment = apply_stellar_treatment
        self.plot = plot
        self.rv_seed = rv_seed
        self.celestrak = celestrak
        formula = config.frames.vacuum_air_formula
        given = getattr(telluric_model, "vacuum_air_formula", formula)
        if given != formula:
            raise ConfigError(
                f"telluric_model uses vacuum_air_formula {given!r}, but the observations use "
                f"{formula!r} (frames.vacuum_air_formula); build the model with the same formula"
            )
        self._model = telluric_model
        self._progress = progress
        spec = observation.instrument
        self.layout = RunLayout.for_run(
            self.output_dir,
            target=observation.target.name,
            site_token=spec.site_token,
            instrument=spec.name,
            date=date_token(e.time_utc for e in observation.exposures),
        )
        self.stellar: StellarProducts | None = None
        self.strong_windows: list[tuple[float, float]] = []
        self.config_hash = config_hash(config)
        self._prepared = False

    # -- helpers --------------------------------------------------------------
    def _emit(self, stage: str, message: str, done: int = 0, total: int = 0) -> None:
        """Log a progress message and pass it to the progress callback."""
        log.info("%s: %s", stage, message)
        if self._progress is not None:
            self._progress(Progress(stage, message, done, total))

    @property
    def model(self) -> TelluricModel:
        """The telluric forward model; built (and its resources verified) on first use."""
        if self._model is None:
            from gtellcorr.telluric.model import TelluricModel

            self._model = TelluricModel.from_settings(
                self.settings,
                self.config.telluric,
                atmosphere=self.config.atmosphere,
                vacuum_air_formula=self.config.frames.vacuum_air_formula,
            )
        return self._model

    @property
    def data_dir(self) -> Path:
        """Directory of the result FITS files and ``summary.csv``."""
        return self.layout.product_dir(PRODUCT_TELLURIC, "data")

    @property
    def figure_dir(self) -> Path:
        """Directory of the per-order fit figures."""
        return self.layout.product_dir(PRODUCT_TELLURIC, "figure")

    def product_path(self, exposure: Exposure) -> Path:
        """Path of the result FITS file of *exposure*."""
        suffix = (
            "single_order_corrected"
            if self.config.relcol_mode == "single_order"
            else "joint_corrected_all_orders"
        )
        return self.data_dir / f"{exposure.path.stem}_{suffix}.fits"

    def _provenance(self, exposure: Exposure) -> dict[str, str]:
        """Header cards that decide whether a result can be reused (``reuse = "matching"``)."""
        return {
            "GTC_VERS": __version__,
            "CFG_HASH": self.config_hash,
            "RES_FP": self.model.resources.local_fingerprint,
            "IN_HASH": _file_sha256(exposure.path),
        }

    def reusable(self, exposure: Exposure) -> bool:
        """Whether the exposure's existing result may be reused (see ``reuse``)."""
        path = self.product_path(exposure)
        if self.reuse == "never" or not path.exists():
            return False
        if self.reuse == "any":
            return True
        return read_provenance(path) == self._provenance(exposure)

    # -- stage 1 ------------------------------------------------------------------
    def prepare(self) -> None:
        """Resources, stellar products and the strong-telluric mask (idempotent)."""
        if self._prepared:
            return
        from gtellcorr.preprocessing.telluric_mask import (
            ensure_rv_telluric_windows,
            ensure_strong_telluric_mask,
            run_line_transmission,
        )
        from gtellcorr.preprocessing.treatments import stellar_treatments

        self._emit(
            "prepare", f"{len(self.observation.exposures)} exposures; output in {self.layout.root}"
        )
        _ = self.model  # verify the LBLRTM resources before anything is written
        if self.apply_stellar_treatment:
            rv_windows = None
            rv = self.config.rv
            lm = self.config.masks.stellar_linemask
            rv_model = rv.source == "measure" and rv.telluric_mask == "model"
            lines_model = (
                self.config.stellar == "mask"
                and lm.user_file is None
                and lm.from_spectrum == "master"
                and lm.blend != "off"
            )
            transmission = None
            if rv_model or lines_model:
                transmission = run_line_transmission(
                    self.model,
                    self.observation.exposures,
                    instrument=self.observation.instrument,
                    fit=self.config,
                    settings=self.settings,
                    celestrak=self.celestrak,
                )
            if rv_model:
                rv_windows = ensure_rv_telluric_windows(
                    self.model,
                    self.observation.exposures,
                    instrument=self.observation.instrument,
                    fit=self.config,
                    settings=self.settings,
                    transmission=transmission,
                )
            treatment = stellar_treatments.get(self.config.stellar)()
            self.stellar = treatment.prepare(
                self.observation,
                self.config,
                self.settings,
                self.layout,
                rv_seed=self.rv_seed,
                rv_telluric_windows=rv_windows,
                telluric_transmission=transmission if lines_model else None,
            )
            self._emit("prepare", f"stellar products: {_describe_status(self.stellar.status)}")
            if self.plot:
                self._plot_rv()
        strong = ensure_strong_telluric_mask(
            self.model,
            self.observation.exposures,
            instrument=self.observation.instrument,
            fit=self.config,
            settings=self.settings,
            out_dir=self.layout.mask_dir,
            stem=self.layout.stem,
            celestrak=self.celestrak,
        )
        if strong is not None:
            self.strong_windows = strong.windows
            if self.stellar is not None:
                self.stellar.strong_windows = strong.windows
            status = _STATUS_WORDS.get(strong.status, strong.status)
            self._emit("prepare", f"strong telluric mask {status}")
        self._prepared = True

    # -- stage 2 ------------------------------------------------------------------
    def _profile(self, exposure: Exposure) -> dict[str, Any]:
        """The 0-120 km atmospheric profile of *exposure*, as LBLRTM input."""
        from gtellcorr.data.atmosphere import full_profile_for_exposure

        full = full_profile_for_exposure(
            exposure,
            self.observation.instrument,
            config=self.config.atmosphere,
            settings=self.settings,
            celestrak=self.celestrak,
        )
        return self.model.lblrtm_profile({k: full[k] for k in ("z_km", "T_K", "P_hPa", "H2O_ppmv")})

    def _stellar_order(self, exposure: Exposure, wave: Any) -> tuple[Any, Any]:
        """Stellar model (or ``None``) and exclusion mask of one order of *exposure*."""
        from gtellcorr.preprocessing.stellar_mask import fit_exclusion
        from gtellcorr.preprocessing.treatments import stellar_treatments

        if self.stellar is None:
            return None, fit_exclusion(wave, strong_windows=self.strong_windows)
        treatment = stellar_treatments.get(self.config.stellar)()
        model, exclusion = treatment.order_model(self.stellar, exposure.path.name, wave)
        return model, exclusion

    def _load(self, exposure: Exposure, orders: Sequence[int]) -> _Loaded:
        """Read *orders* of *exposure*; unreadable orders are recorded, not raised."""
        from gtellcorr.data.observation import readers

        spec = self.observation.instrument
        reader = readers.get(spec.spectral.reader)
        formula = self.config.frames.vacuum_air_formula
        loaded = _Loaded()
        for order in orders:
            try:
                s = reader.load_order(exposure.path, order, spec.spectral, formula)
            except DataError as exc:
                loaded.errors[order] = exc
                continue
            if s.wave_fit.size > 0:
                loaded.spectra[order] = s
            else:
                loaded.raw[order] = reader.load_order_raw(
                    exposure.path, order, spec.spectral, formula
                )
        return loaded

    def _spec(self, spectrum: Any, order: int, exposure: Exposure) -> FitSpec:
        """Fit specification of one order (molecules, bounds, starting R); no template
        velocity without a stellar model (front end off)."""
        from dataclasses import replace

        spec = build_fit_spec(
            (float(spectrum.wave_fit.min()), float(spectrum.wave_fit.max())),
            config=self.config,
            instrument=self.observation.instrument,
            order=order,
            resolution=exposure.resolution,  # header R (e.g. PEPSI fibre), else the default
        )
        if spec.fit_stellar_shift and self.stellar is None:
            spec = replace(spec, fit_stellar_shift=False)
        return spec

    def _treatment_name(self) -> str:
        """Stellar treatment actually applied (``mask`` when the front end is off)."""
        return self.config.stellar if self.stellar is not None else "mask"

    def _r_source(self, fitted: str) -> str:
        """Where an order's resolving power came from, as written in the results.

        *fitted* is the label used when R is fitted (``fitted_order`` or
        ``median_fitted_orders``); otherwise R came from the run configuration, the
        exposure header or the instrument definition.
        """
        if self.config.lsf.fit_resolution:
            return fitted
        if self.config.lsf.resolution is not None:
            return "run_config"
        return (
            "instrument_header"
            if self.observation.instrument.resolution.keys
            else "instrument_config"
        )

    def _shift_source(self, fitted: str) -> str:
        """Where an order's wavelength-solution correction came from (``not_fitted`` when off)."""
        return fitted if self.config.wavelength.fit else "not_fitted"

    def fit_exposure(
        self, exposure: Exposure
    ) -> tuple[list[OrderFitResult], list[tuple[bool, list[Any]]], Path | None]:
        """Fit and correct one exposure; returns ``(results, summary rows, FITS path)``."""
        self.prepare()
        profile = self._profile(exposure)
        geometry = Geometry.from_exposure(exposure)
        if self.config.relcol_mode == "single_order":
            results, rows, metadata, fit_summary, grids = self._fit_single(
                exposure, profile, geometry
            )
        else:
            results, rows, metadata, fit_summary, grids = self._fit_joint(
                exposure, profile, geometry
            )
        if not results:
            return results, rows, None
        path = write_exposure_fits(
            self.product_path(exposure),
            results,
            exposure=exposure.path.name,
            instrument=self.observation.instrument.name.split("/", 1)[-1],
            correction_mode=self.config.relcol_mode,
            fit_summary=fit_summary,
            order_metadata=metadata,
            run_metadata=self._run_metadata(exposure, results, metadata),
            provenance=self._provenance(exposure),
            grids=grids,
        )
        if self.plot:
            self._plot_orders(path, exposure, results, metadata)
        return results, rows, path

    def _fit_orders(self) -> list[int]:
        """Orders fitted in single-order mode: configured, else the instrument's
        molecule anchor orders, else every order."""
        chosen = self.config.orders.fit_orders
        if chosen is not None:
            return [int(o) for o in chosen]
        anchors = self.observation.instrument.anchor_orders
        return (
            sorted(set().union(*anchors.values()))
            if anchors
            else list(self.observation.instrument.orders)
        )

    def _fit_single(
        self, exposure: Exposure, profile: Mapping[str, Any], geometry: Geometry
    ) -> tuple[
        list[OrderFitResult],
        list[tuple[bool, list[Any]]],
        dict[int, dict[str, Any]],
        dict[str, Any],
        dict[int, Any],
    ]:
        """Fit every fit order of *exposure* on its own (``relcol_mode = "single_order"``).

        Returns the order results, the ``summary.csv`` rows, per-order metadata for
        the result file, the exposure-level fit summary and the delivered pixel grids.
        """
        results: list[OrderFitResult] = []
        rows: list[tuple[bool, list[Any]]] = []
        metadata: dict[int, dict[str, Any]] = {}
        name = exposure.path.name
        t_iso = exposure.time_utc.isoformat()
        loaded = self._load(exposure, self._fit_orders())
        for order in self._fit_orders():
            if order not in loaded.spectra:
                error = loaded.errors.get(order)
                rows.append((False, skipped_row(name, order, t_iso, error=error)))
                continue
            s = loaded.spectra[order]
            spec = self._spec(s, order, exposure)
            stellar, exclusion = self._stellar_order(exposure, s.wave_fit)
            wave_range = (float(s.wave_fit.min()), float(s.wave_fit.max()))

            def cache(
                wave_range: tuple[float, float] = wave_range, spec: FitSpec = spec
            ) -> OrderCache:
                """Build this order's optical-depth cache only if the order is fitted."""
                return self.model.build_order_cache(
                    profile, wave_range, geometry=geometry, fit_molecules=spec.molecules
                )

            started = time.time()
            result = fit_single_order(
                s,
                spec,
                cache,
                stellar_model=stellar,
                exclusion=exclusion,
                stellar_treatment=self._treatment_name(),
            )
            elapsed = time.time() - started
            if result.status == "no_fit_molecules" or result.resolution is None:
                rows.append((False, single_row(name, result, t_iso, elapsed, fitted=False)))
                continue
            results.append(result)
            metadata[order] = {
                "used_in_fit": True,
                "resolution_value_source": self._r_source("fitted_order"),
                "lsf_shape_value_source": "fitted_order"
                if self.config.lsf.fit_shape
                else "fitting_config",
                "wavesol_source": self._shift_source("fitted_order"),
            }
            rows.append((True, single_row(name, result, t_iso, elapsed, fitted=True)))
        return results, rows, metadata, {"relcol_fit_mode": "single_order"}, loaded.grids()

    def _fit_joint(
        self, exposure: Exposure, profile: Mapping[str, Any], geometry: Geometry
    ) -> tuple[
        list[OrderFitResult],
        list[tuple[bool, list[Any]]],
        dict[int, dict[str, Any]],
        dict[str, Any],
        dict[int, Any],
    ]:
        """Fit the molecule columns jointly over several orders, then correct all orders.

        The fit orders (configured, or chosen by their molecular bands) share one set
        of molecule columns; resolving power, LSF shape and wavelength correction stay
        per order. Every other readable order is then corrected with the shared
        columns and the median R, shape and wavelength correction of the fitted orders. Orders
        without usable wavelengths are kept as all-NaN placeholders, so the result
        file always holds every order of the exposure.
        """
        spec_inst = self.observation.instrument
        name = exposure.path.name
        candidates = (
            [int(o) for o in self.config.orders.output_orders]
            if self.config.orders.output_orders is not None
            else list(spec_inst.orders)
        )
        loaded = self._load(exposure, candidates)
        if not loaded.spectra and not loaded.raw:
            raise DataError(f"{spec_inst.name}: no readable orders in {name}")
        ranges = {
            o: (float(s.wave_fit.min()), float(s.wave_fit.max())) for o, s in loaded.spectra.items()
        }
        if self.config.orders.fit_orders is None:
            fit_orders = select_fit_orders_for_molecules(
                ranges, config=self.config, instrument=spec_inst
            )
        else:
            fit_orders = [int(o) for o in self.config.orders.fit_orders]
        missing = [o for o in fit_orders if o not in loaded.spectra]
        if missing:
            raise DataError(
                f"{spec_inst.name}: requested joint fit orders not readable in {name}: {missing}"
            )

        inputs = []
        continuum_order = int(self.config.continuum.order)
        for order in fit_orders:
            s = loaded.spectra[order]
            spec = self._spec(s, order, exposure)
            if not spec.molecules:
                continue
            stellar, exclusion = self._stellar_order(exposure, s.wave_fit)
            item = JointOrderInput.prepare(
                s,
                spec,
                cache=None,  # type: ignore[arg-type]  # set below, after the pixel check
                stellar_model=stellar,
                exclusion=exclusion,
            )
            if np.count_nonzero(item.keep_pixels) <= continuum_order + 2:
                continue
            self._emit("fit", f"{exposure.path.stem}: optical depths of order {order:03d}")
            item.cache = self.model.build_order_cache(
                profile, ranges[order], geometry=geometry, fit_molecules=spec.molecules
            )
            inputs.append(item)
        if not inputs:
            raise FitError(f"{spec_inst.name}: no fit-capable orders for {name}")

        started = time.time()
        joint = fit_joint_orders(inputs, stellar_treatment=self._treatment_name())
        elapsed = time.time() - started
        fitted = {int(r.order): r for r in joint.order_results}
        fit_r = bool(self.config.lsf.fit_resolution)
        fit_shape = bool(self.config.lsf.fit_shape)
        median_r = _median(joint.order_results, "resolution")
        median_shape = _median(joint.order_results, "lsf_shape")
        median_shift = _median_shift(joint.order_results)
        if fit_r and median_r is None:
            raise FitError("joint fit produced no finite fitted-order resolution")
        if fit_shape and median_shape is None:
            raise FitError("joint fit produced no finite fitted-order lsf_shape")

        results: list[OrderFitResult] = []
        metadata: dict[int, dict[str, Any]] = {}
        for order in sorted(set(loaded.spectra) | set(loaded.raw)):
            if order in loaded.raw:
                results.append(_placeholder(order, loaded.raw[order], self._treatment_name()))
                metadata[order] = {
                    "used_in_fit": False,
                    "resolution_value_source": "not_applicable",
                    "lsf_shape_value_source": "not_applicable",
                    "wavesol_source": "not_applicable",
                }
                continue
            if order in fitted:
                results.append(fitted[order])
                metadata[order] = {
                    "used_in_fit": True,
                    "resolution_value_source": self._r_source("fitted_order"),
                    "lsf_shape_value_source": "fitted_order" if fit_shape else "fitting_config",
                    "wavesol_source": self._shift_source("fitted_order"),
                }
                continue
            s = loaded.spectra[order]
            base = self._spec(s, order, exposure)
            resolution = median_r if fit_r else float(base.resolution_init)
            shape = median_shape if fit_shape else base.lsf_shape
            results.append(
                self._correct_with_shared(
                    exposure,
                    order,
                    s,
                    base,
                    joint,
                    resolution,
                    shape,
                    profile,
                    geometry,
                    wave_shift=median_shift,
                )
            )
            metadata[order] = {
                "used_in_fit": False,
                "resolution_value_source": self._r_source("median_fitted_orders"),
                "lsf_shape_value_source": "median_fitted_orders" if fit_shape else "fitting_config",
                "wavesol_source": self._shift_source("median_fitted_orders"),
            }
        rows = [
            (True, joint_row(name, r, exposure.time_utc.isoformat(), elapsed))
            for r in joint.order_results
        ]
        fit_summary = {
            "relcol_fit_mode": "joint_orders",
            "relcol": dict(joint.relcol),
            "relcol_err": dict(joint.relcol_err),
            "chi2": joint.chi2,
            "chi2_reduced": joint.chi2_reduced,
            "dof_mask_removed": joint.dof_mask_removed,
            "converged": joint.success,
            "n_eval": joint.n_eval,
            "optimizer_message": joint.message,
            "fit_orders": [int(r.order) for r in joint.order_results],
            "output_orders": sorted(int(o) for o in loaded.spectra),
        }
        columns = ", ".join(f"{m.upper()} {v:.3f}" for m, v in joint.relcol.items())
        if "stellar_shift" in joint.parameters:
            columns += f"; template velocity {1000 * joint.parameters['stellar_shift']:+.1f} m/s"
        outcome = "" if joint.success else " (not converged)"
        self._emit("fit", f"{exposure.path.stem}: relative columns {columns}{outcome}")
        return results, rows, metadata, fit_summary, loaded.grids()

    def _correct_with_shared(
        self,
        exposure: Exposure,
        order: int,
        s: Any,
        base: FitSpec,
        joint: Any,
        resolution: float | None,
        shape: float | None,
        profile: Mapping[str, Any],
        geometry: Geometry,
        *,
        wave_shift: tuple[float, ...] | None = None,
    ) -> OrderFitResult:
        """Correct an output order with the exposure's shared relcols (not fitted).

        R, LSF shape and wavelength correction: the medians of the fitted orders; the
        template velocity: the joint fit's (shared by the exposure).
        """
        from dataclasses import replace

        stellar, exclusion = self._stellar_order(exposure, s.wave_fit)
        treatment = self._treatment_name()
        if not joint.relcol:
            return _identity(order, s, stellar, exclusion, treatment)
        shared = joint.parameters.get("stellar_shift") if stellar is not None else None
        stellar_shift = None if shared is None else float(shared)
        assert resolution is not None  # the fitted median, or the fixed R
        molecules = tuple(joint.relcol)
        spec = replace(
            base,
            molecules=molecules,
            relcol_init={m: float(joint.relcol[m]) for m in molecules},
            relcol_bounds={
                m: (
                    float(self.config.molecules[m].bounds[0]),
                    float(self.config.molecules[m].bounds[1]),
                )
                for m in molecules
            },
            fit_resolution=False,
            resolution_init=float(resolution),
            fit_lsf_shape=False,
            lsf_shape=shape,
            fit_wave_shift=False,
            wave_shift=wave_shift,
            fit_stellar_shift=False,
            stellar_shift=stellar_shift,
        )
        cache = self.model.build_order_cache(
            profile,
            (float(s.wave_fit.min()), float(s.wave_fit.max())),
            geometry=geometry,
            fit_molecules=molecules,
        )
        err = np.asarray(s.err, float)
        keep = np.isfinite(s.wave_fit) & np.isfinite(s.flux) & np.isfinite(err) & (err > 0)
        if exclusion is not None:
            keep &= ~np.asarray(exclusion, bool)
        state = ModelState(
            scales=dict(spec.relcol_init),
            resolution=float(resolution),
            lsf_shape=shape,
            wave_shift=wave_shift,
            stellar_shift=stellar_shift,
        )
        evaluated = evaluate_order(
            s,
            cache,
            spec,
            state,
            weight=inverse_variance(err),
            keep_mask=keep,
            stellar_model=stellar,
        )
        telluric, continuum = lines_only(evaluated, evaluated.continuum)
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = np.where(telluric > 0.0, s.flux / telluric, np.nan)
            err_corrected = np.where(telluric > 0.0, err / telluric, np.inf)
        return OrderFitResult(
            order=int(order),
            wave_input=s.wave_input,
            wave_fit=s.wave_fit,
            flux_obs=s.flux,
            err=err,
            tell_model=telluric,
            flux_corrected=corrected,
            err_corrected=err_corrected,
            continuum=continuum,
            stellar_model=stellar_on_data(stellar, stellar_shift),
            exclusion=exclusion,
            relcol={m: float(joint.relcol[m]) for m in molecules},
            relcol_err={m: float(joint.relcol_err.get(m, np.nan)) for m in molecules},
            resolution=float(resolution),
            lsf_kind=spec.lsf_kind,
            lsf_shape=float(shape) if shape is not None else None,
            wave_shift=wave_shift,
            stellar_shift=stellar_shift,
            stellar_shift_err=(
                joint.parameter_errors.get("stellar_shift") if stellar_shift is not None else None
            ),
            stellar_treatment=treatment,
            status="corrected_not_fitted",
        )

    def _run_metadata(
        self,
        exposure: Exposure,
        results: Sequence[OrderFitResult],
        metadata: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Run-level key/value table of the result file (inputs, resources, versions)."""
        model = self.model
        return {
            "target": self.layout.target,
            "instrument": self.observation.instrument.name,
            "site_token": self.observation.instrument.site_token,
            "exposure": exposure.path.name,
            "time_utc": exposure.time_utc.isoformat(),
            "airmass": f"{float(exposure.airmass):.8g}",
            "airmass_source": exposure.airmass_source,
            # paths relative to the output directory or the run directory, resources
            # outside them by name: a result file names no directory of this machine
            "run_root": self.layout.root.relative_to(self.layout.root.parents[2]).as_posix(),
            "stellar_treatment": self._treatment_name(),
            "relcol_fit_mode": self.config.relcol_mode,
            "fit_orders": json.dumps(
                [int(r.order) for r in results if metadata.get(int(r.order), {}).get("used_in_fit")]
            ),
            "output_orders": json.dumps([int(r.order) for r in results]),
            "rv_table": (
                _within(self.stellar.rv_table, self.layout.root) if self.stellar is not None else ""
            ),
            "lower_backend": self.config.atmosphere.lower_backend,
            "upper_source": self.config.atmosphere.upper_source,
            "gtellcorr_version": __version__,
            "config_sha256": self.config_hash,
            "lblrtm_bundle": Path(model.resources.bundle).name,
            "lblrtm_local_fingerprint": model.resources.local_fingerprint,
            "lblrtm_setup": model.resources.setup_id,
            "lblrtm_timeout_s": f"{model.timeout_s:.8g}",
            "lblrtm_wave_margin_nm": f"{model.config.wave_margin_nm:.8g}",
            "tau_cache": Path(model.tau_cache).name,
            "rv_seed": "" if self.rv_seed is None else str(self.rv_seed),
        }

    def _plot_orders(
        self,
        path: Path,
        exposure: Exposure,
        results: Sequence[OrderFitResult],
        metadata: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """One fit figure per fitted order of *exposure*."""
        import matplotlib.pyplot as plt

        from gtellcorr.plotting import plot_order_fit

        for result in results:
            if not metadata.get(int(result.order), {}).get("used_in_fit"):
                continue
            fig, _ = plot_order_fit(
                str(path), order=int(result.order),
                out=self.figure_dir / f"{exposure.path.stem}_ord{int(result.order):03d}.png",
            )  # fmt: skip
            plt.close(fig)

    def _plot_rv(self) -> None:
        """RV figures in ``Radial_Velocities/figure``: the RV time series and, for
        measured RVs, master vs per-order RV and (``rv.plot_ccf``) each exposure's CCF."""
        import matplotlib.pyplot as plt

        from gtellcorr.plotting import (
            plot_ccf_diagnostics,
            plot_rv_master_vs_perorder,
            plot_rv_timeseries,
        )

        assert self.stellar is not None
        figure_dir = self.layout.product_dir(PRODUCT_RV, "figure")
        label = f"{self.layout.target} {self.layout.instrument}"
        table = self.stellar.rv_table
        fig, _ = plot_rv_timeseries(
            table, out=figure_dir / "rv_timeseries.png", title=f"{label} per-exposure RV"
        )
        plt.close(fig)
        if self.stellar.status.get("rv") == "user_file":
            return  # no CCFs or per-order RVs behind user RVs
        fig, _ = plot_rv_master_vs_perorder(
            table,
            out=figure_dir / "rv_master_vs_perorder.png",
            title=f"{label}: master CCF vs per-order RV",
        )
        plt.close(fig)
        if not self.config.rv.plot_ccf:
            return
        for exposure in self.observation.exposures:
            obs_time = exposure.time_utc.strftime("%Y%m%dT%H%M%S")
            ccf = self.layout.ccf_path(obs_time, exposure.path)
            if ccf.exists():
                fig, _ = plot_ccf_diagnostics(
                    ccf,
                    out=figure_dir / "ccf" / f"{exposure.path.stem}_ccf.png",
                    title=f"{exposure.path.name} - {label} RV CCF",
                    max_orders=self.config.rv.ccf_plot_max_orders,
                )
                plt.close(fig)

    # -- stage 3 ------------------------------------------------------------------
    def run(self) -> Result:
        """Correct every exposure and write the run-level products."""
        from gtellcorr.data.products.result import load_result, summary_rows_from_fits

        self.prepare()
        exposures = self.observation.exposures
        rows: list[list[Any]] = []
        products = 0
        for i, exposure in enumerate(exposures, start=1):
            if self.reusable(exposure):
                self._emit("fit", f"{exposure.path.stem}: result reused", i, len(exposures))
                rows.extend(
                    summary_rows_from_fits(
                        self.product_path(exposure), exposure.time_utc.isoformat()
                    )
                )
                products += 1
                continue
            self._emit("fit", f"{exposure.path.stem}: fitting", i, len(exposures))
            _, exposure_rows, path = self.fit_exposure(exposure)
            if path is not None:
                products += 1
                rows.extend(row for _, row in exposure_rows)
            else:
                rows.extend(row for ok, row in exposure_rows if not ok)
        summary = self.data_dir / "summary.csv"
        with open(summary, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(SUMMARY_COLUMNS)
            writer.writerows(rows)
        if products == 0:
            raise FitError(f"no exposure-level result was produced or reused; see {summary}")
        post = self.config.postprocessing
        if post.telluric_pca:
            from gtellcorr.data.products.pca import compute_telluric_pca_for_run

            pca = compute_telluric_pca_for_run(
                self.data_dir, n_components=post.telluric_pca_components
            )
            self._emit("pca", f"{sum(r['status'] == 'ok' for r in pca)} orders updated")
        figures = sorted(str(p) for p in self.figure_dir.glob("*.png")) if self.plot else []
        self._write_manifest(summary, figures)
        return load_result(self.layout.root)

    def _write_manifest(self, summary: Path, figures: Sequence[str]) -> None:
        """Write ``run_config.toml`` and ``run_manifest.json`` in the run directory."""
        from gtellcorr.config.toml_io import dumps_config

        (self.layout.root / "run_config.toml").write_text(dumps_config(self.config))
        manifest = {
            "gtellcorr_version": __version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instrument": self.observation.instrument.name,
            "target": self.layout.target,
            "exposures": [e.path.name for e in self.observation.exposures],
            "config_sha256": self.config_hash,
            "lblrtm_local_fingerprint": self.model.resources.local_fingerprint,
            "summary_path": _within(summary, self.layout.root),
            "figure_count": len(figures),
            "figures": list(figures),
        }
        (self.layout.root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def result(self) -> Result:
        """The run's results as currently on disk (see :func:`~gtellcorr.load_result`)."""
        from gtellcorr.data.products.result import load_result

        return load_result(self.layout.root)


# ---------------------------------------------------------------------------
def _within(path: Path, root: Path) -> str:
    """*path* relative to *root* (POSIX form); its name if it lies outside *root*."""
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _median(results: Sequence[OrderFitResult], attribute: str) -> float | None:
    """Median of a numeric result attribute over the orders where it is finite."""
    values = []
    for r in results:
        value = getattr(r, attribute)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.nanmedian(values)) if values else None


def _median_shift(results: Sequence[OrderFitResult]) -> tuple[float, ...] | None:
    """Coefficient-wise median of the fitted wavelength corrections (``None`` if none)."""
    shifts = [r.wave_shift for r in results if r.wave_shift is not None]
    if not shifts:
        return None
    n = max(len(c) for c in shifts)
    padded = np.array([list(c) + [0.0] * (n - len(c)) for c in shifts], float)
    return tuple(float(v) for v in np.nanmedian(padded, axis=0))


def _identity(order: int, s: Any, stellar: Any, exclusion: Any, treatment: str) -> OrderFitResult:
    """Result for an order left uncorrected (no molecule to fit): transmission 1."""
    return OrderFitResult(
        order=int(order),
        wave_input=s.wave_input,
        wave_fit=s.wave_fit,
        flux_obs=s.flux,
        err=s.err,
        tell_model=np.ones_like(s.flux),
        flux_corrected=s.flux.copy(),
        err_corrected=s.err.copy(),
        continuum=np.ones_like(s.flux),
        stellar_model=stellar_on_data(stellar, None),
        exclusion=np.zeros_like(s.flux, dtype=bool) if exclusion is None else exclusion,
        lsf_kind="",
        stellar_treatment=treatment,
        status="no_fit_molecules",
    )


def _placeholder(order: int, raw: Any, treatment: str) -> OrderFitResult:
    """Row for an order without usable wavelengths: NaN products, all excluded."""
    return OrderFitResult(
        order=int(order),
        wave_input=raw.wave_input,
        wave_fit=raw.wave_fit,
        flux_obs=raw.flux,
        err=raw.err,
        tell_model=np.full_like(raw.flux, np.nan, dtype=float),
        flux_corrected=np.full_like(raw.flux, np.nan, dtype=float),
        err_corrected=np.full_like(raw.err, np.nan, dtype=float),
        continuum=np.full_like(raw.flux, np.nan, dtype=float),
        stellar_model=None,
        exclusion=np.ones_like(raw.flux, dtype=bool),
        lsf_kind="",
        stellar_treatment=treatment,
        status=raw.status or "raw_only",
    )


def run(
    observation: Observation,
    config: FitConfig,
    *,
    output_dir: str | Path | None = None,
    **options: Any,
) -> Result:
    """Correct ``observation`` with ``config`` (see :class:`Pipeline` for the options)."""
    return Pipeline(observation, config, output_dir=output_dir, **options).run()
