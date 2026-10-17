"""Stellar preparation of a run: template, RV table, stellar-line masks.

Each product is reused when it already exists in the run directory (names and
layout in :mod:`gtellcorr.data.products.layout`) and can be reused, and otherwise
generated or refused according to ``FitConfig.on_missing_stellar_data``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gtellcorr.data.observation.frames import vac_to_air
from gtellcorr.data.observation.instrument import start_resolution
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.preprocessing import rv as rvmod
from gtellcorr.preprocessing.stellar_mask import (
    build_stellar_linemask,
    missing_exposure_linemasks,
    write_exposure_linemasks,
)
from gtellcorr.preprocessing.synthesis import (
    Synthesis,
    broaden_template,
    generate_template,
    load_template,
    template_filename,
)
from gtellcorr.preprocessing.windows import C_KMS, Window, read_windows

if TYPE_CHECKING:
    from collections.abc import Callable

    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.data.observation.observation import Observation
    from gtellcorr.data.products.layout import RunLayout
    from gtellcorr.preprocessing.telluric_mask import LineTransmission
    from gtellcorr.settings import Settings

__all__ = [
    "StellarProducts",
    "ensure_linemasks",
    "ensure_rv_table",
    "ensure_template",
    "load_stellar_template",
    "prepare_stellar",
    "template_path",
]

log = logging.getLogger(__name__)
Array = NDArray[np.float64]

#: Prefix of the rest-window source recorded in masks built from ``stellar_linemask.user_file``.
USER_WINDOWS_SOURCE = "user_file:"
#: The RV a per-exposure mask was shifted by, as written in its header.
_MASK_RV = re.compile(r"apparent stellar RV = ([+-]?\d+(?:\.\d+)?) km/s")


@dataclass
class StellarProducts:
    """What the per-order stellar treatment needs, prepared once per run."""

    layout: RunLayout
    template_path: Path
    rv_table: Path
    rv_kms: Mapping[str, float]
    """Apparent topocentric RV per exposure file name."""
    activity_windows: list[Window]
    """``masks.activity_windows_nm`` (air nm, not shifted)."""
    bad_regions: list[Window]
    """``masks.bad_regions_nm`` (air nm, not shifted)."""
    strong_windows: list[Window] = field(default_factory=list)
    """Strong-telluric windows, set by the pipeline once the strong mask exists."""
    status: dict[str, str] = field(default_factory=dict)
    """Per product: ``reused``, ``generated`` or ``user_file``."""
    template_frame: str = "rest"
    """``rest``: placed by the exposure RV; ``topocentric``: never shifted."""
    _template: tuple[Array, Array] | None = None

    @property
    def template(self) -> tuple[Array, Array]:
        """The template as the run uses it (set by :func:`prepare_stellar`; otherwise
        read from ``template_path`` as written, without medium or resolution
        conversion)."""
        if self._template is None:
            self._template = load_template(self.template_path)
        return self._template

    def exposure_rv(self, exposure: str) -> float:
        """RV of an exposure: the first RV-table row whose file name equals its name, has
        the same stem, or starts with its stem. The rows are tried in table order, so a
        prefix match earlier in the table wins over an exact match later on."""
        base = Path(exposure).name
        stem = Path(base).stem
        for name, rv in self.rv_kms.items():
            if name == base or Path(name).stem == stem or name.startswith(stem):
                return rv
        raise KeyError(f"no RV-table row for {base!r} in {self.rv_table.name}")

    def placement_rv(self, exposure: str) -> float:
        """Velocity the template is shifted by for *exposure*: its RV, or 0 for a
        ``topocentric`` template (already at the observed positions)."""
        return 0.0 if self.template_frame == "topocentric" else self.exposure_rv(exposure)


def _synthesis(observation: Observation, config: FitConfig, settings: Settings) -> Synthesis:
    """Synthesis inputs from the target, ``stellar_model`` and the settings."""
    if observation.target is None:
        raise ConfigError("the stellar template needs the target's parameters (data.target)")
    return Synthesis.from_config(observation.target, config.stellar_model, settings)


def template_path(
    observation: Observation, config: FitConfig, settings: Settings, layout: RunLayout
) -> Path:
    """``stellar_model.template_file`` if set, else the synthesised template in the run's
    template directory, named by :func:`~gtellcorr.preprocessing.synthesis.template_filename`
    at the instrument's wavelength range and the start R of the fit
    (:func:`~gtellcorr.data.observation.instrument.start_resolution`)."""
    if config.stellar_model.template_file is not None:
        return Path(config.stellar_model.template_file).expanduser()
    spec = observation.instrument
    if spec.spectral.wavelength_range_nm is None:
        raise ConfigError(
            f"{spec.name}: spectral.wavelength_range_nm is needed for the stellar template"
        )
    name = template_filename(
        _synthesis(observation, config, settings),
        wave_range_nm=tuple(spec.spectral.wavelength_range_nm),  # type: ignore[arg-type]
        resolution=_start_resolution(observation, config),
    )
    return layout.template_dir / name


def load_stellar_template(
    path: str | Path, observation: Observation, config: FitConfig
) -> tuple[Array, Array]:
    """The template as the run uses it.

    A user-supplied ``stellar_model.template_file`` is converted to air
    (``template_medium = "vacuum"``) and broadened from ``template_resolution`` to the
    start R of the fit (:func:`~gtellcorr.data.observation.instrument.start_resolution`);
    a synthesised template is used as written (it was synthesised at that R).
    """
    wave, flux = load_template(path)
    sm = config.stellar_model
    if sm.template_file is None:
        return wave, flux
    if sm.template_medium == "vacuum":
        wave = np.asarray(vac_to_air(wave, config.frames.vacuum_air_formula), float)
    if sm.template_resolution is not None:
        flux = broaden_template(
            wave,
            flux,
            from_resolution=float(sm.template_resolution),
            to_resolution=_start_resolution(observation, config),
        )
    return wave, flux


def _start_resolution(observation: Observation, config: FitConfig) -> float:
    """The R the stellar template is broadened to (the start R of the fit)."""
    return start_resolution(observation.exposures, observation.instrument, config.lsf.resolution)


def ensure_template(
    observation: Observation,
    config: FitConfig,
    settings: Settings,
    layout: RunLayout,
    *,
    workers: int = 12,
) -> tuple[Path, str]:
    """Return ``(template path, "reused" | "generated")``.

    An existing file at :func:`template_path` is reused (see
    :func:`~gtellcorr.preprocessing.synthesis.generate_template` for what a reuse does
    not check). A missing synthesised template is generated if
    ``on_missing_stellar_data = "generate"``; otherwise, and always for a missing
    ``stellar_model.template_file``, :class:`FileNotFoundError`.
    """
    path = template_path(observation, config, settings, layout)
    if path.exists():
        return path, "reused"
    if config.stellar_model.template_file is not None:
        # never synthesise into the user's path: the template would then be taken
        # for the user's and converted by template_medium / template_resolution
        raise FileNotFoundError(f"stellar_model.template_file not found: {path}")
    policy = config.on_missing_stellar_data
    if policy != "generate":
        # Both treatments need the template: the RV measurement cross-correlates with it.
        raise FileNotFoundError(
            f"stellar template {path.name} not found (on_missing_stellar_data = {policy!r}); "
            "set it to 'generate' or provide stellar_model.template_file"
        )
    spec = observation.instrument
    generate_template(
        _synthesis(observation, config, settings),
        path,
        wave_range_nm=tuple(spec.spectral.wavelength_range_nm),  # type: ignore[arg-type]
        resolution=_start_resolution(observation, config),
        workers=workers,
    )
    return path, "generated"


def ensure_rv_table(
    observation: Observation,
    config: FitConfig,
    layout: RunLayout,
    template: tuple[Any, Any],
    *,
    seed: int | None = 0,
    telluric_windows: Sequence[Window] | None = None,
) -> tuple[Path, str]:
    """The run's RV table: from ``rv.user_rv_file``, else reused if it covers every
    exposure, was measured and was measured with the same settings (:func:`_rv_settings`;
    a table without a settings record is reused only if the settings are those of the
    earlier versions: band windows and a fixed CCF grid), else measured.

    *telluric_windows* are the model windows of ``rv.telluric_mask = "model"``
    (:func:`~gtellcorr.preprocessing.telluric_mask.ensure_rv_telluric_windows`)."""
    marker = _rv_source_marker(layout)
    if config.rv.source == "user_file":
        assert config.rv.user_rv_file is not None  # RVConfig.validate
        path = rvmod.write_user_rv_table(
            observation.exposures, config.rv.user_rv_file, layout, instrument=observation.instrument
        )
        marker.write_text(f"user_file {Path(config.rv.user_rv_file).expanduser().resolve()}\n")
        return path, "user_file"
    names = [e.path.name for e in observation.exposures]
    vsini = None if observation.target is None else observation.target.vsini_kms
    windows = rvmod.rv_mask_windows(
        config.rv, config.molecules, config.masks.activity_windows_nm, telluric_windows
    )
    settings = _rv_settings(config, template, windows, vsini, model=telluric_windows is not None)
    record = _rv_settings_record(layout)
    if (
        not marker.exists()
        and layout.rv_table.exists()
        and rvmod.rv_table_covers(layout.rv_table, names)
    ):
        stale = _stale_rv_table(record, settings, config)
        if stale is None:
            return layout.rv_table, "reused"
        log.info("measuring the RVs again: %s", stale)
    path = rvmod.measure_rv_table(
        observation.exposures,
        template,
        instrument=observation.instrument,
        layout=layout,
        config=config.rv,
        mask_windows=windows,
        formula=config.frames.vacuum_air_formula,
        seed=seed,
        vsini_kms=vsini,
    )
    marker.unlink(missing_ok=True)
    record.write_text(json.dumps(settings, indent=1) + "\n")
    return path, "generated"


def _rv_source_marker(layout: RunLayout) -> Path:
    """Present while the run's RV table was written from a user file (not measured)."""
    return layout.rv_table.with_name(f"{layout.rv_table.stem}_source.txt")


def _rv_settings_record(layout: RunLayout) -> Path:
    """Settings the run's RV table was measured with (:func:`_rv_settings`)."""
    return layout.rv_table.with_name(f"{layout.rv_table.stem}_settings.json")


def _rv_settings(
    config: FitConfig,
    template: tuple[Any, Any],
    windows: Sequence[Window],
    vsini_kms: float | None,
    *,
    model: bool,
) -> dict[str, Any]:
    """What a measured RV table depends on besides the spectra: the template, the masked
    windows, the CCF grid and the orders (as read back from the JSON record)."""
    rv = config.rv
    tw, tf = (np.asarray(a, dtype=np.float64) for a in template)
    lo, hi = rvmod.ccf_grid_limits(rv, vsini_kms)
    settings = {
        "telluric_mask": "model" if model else "bands",
        "windows_sha256": hashlib.sha256(
            np.round(np.asarray(windows, float), 6).tobytes()
        ).hexdigest(),
        "template_sha256": hashlib.sha256(tw.tobytes() + tf.tobytes()).hexdigest(),
        "ccf_grid_kms": [lo, hi, float(rv.ccf_dv)],
        "orders": None if rv.orders is None else [int(o) for o in rv.orders],
    }
    loaded: dict[str, Any] = json.loads(json.dumps(settings))
    return loaded


def _stale_rv_table(record: Path, settings: Mapping[str, Any], config: FitConfig) -> str | None:
    """Why the existing RV table cannot be reused, or ``None``."""
    try:
        recorded = json.loads(record.read_text())
    except (OSError, ValueError):
        # tables of earlier versions: band windows and the fixed grid
        legacy = settings["telluric_mask"] == "bands" and settings["ccf_grid_kms"][:2] == [
            float(config.rv.ccf_v_lo),
            float(config.rv.ccf_v_hi),
        ]
        return None if legacy else f"there is no record of how it was measured ({record.name})"
    changed = sorted(k for k in set(recorded) | set(settings) if recorded.get(k) != settings.get(k))
    if changed:
        return f"it was measured with other settings ({', '.join(changed)})"
    return None


def _linemask_settings(
    config: FitConfig, products: StellarProducts, transmission: LineTransmission | None = None
) -> dict[str, Any]:
    """What the per-exposure stellar-line masks depend on, besides the RVs."""
    lm = config.masks.stellar_linemask
    if lm.user_file is not None:
        path = Path(lm.user_file).expanduser()
        settings: dict[str, Any] = {
            "source": "user_file",
            "file": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    elif lm.from_spectrum == "template":
        settings = {
            "source": "template",
            "template": products.template_path.name,
            "template_frame": products.template_frame,
            "depth_below": lm.depth_below,
            "pad_kms": lm.pad_kms,
            "merge_gap_nm": lm.merge_gap_nm,
        }
    else:
        sm = config.stellar_model
        settings = {
            "source": "master",
            "line_elements": list(sm.line_elements),
            "line_min_depth": sm.line_min_depth,
            "ispec_linelist": sm.ispec.linelist,
            "blend": lm.blend,
        }
        if lm.blend != "off":
            if transmission is None:
                raise DataError(
                    f"masks.stellar_linemask.blend = {lm.blend!r} needs the run's telluric "
                    "model: run the Pipeline, or set blend = 'off'"
                )
            settings |= {
                "blend_ratio": lm.blend_ratio,
                "telluric_depth": lm.telluric_depth,
                "telluric_model_sha256": transmission.sha256(),
            }
    loaded: dict[str, Any] = json.loads(json.dumps(settings))  # as read back from the record
    return loaded


def _recorded_linemask_settings(layout: RunLayout) -> dict[str, Any] | None:
    """The settings recorded with the existing masks, or ``None``."""
    try:
        recorded: dict[str, Any] = json.loads(layout.linemask_settings.read_text())
    except (OSError, ValueError):
        return None
    return recorded


def _stale_linemasks(rv_path: Path, layout: RunLayout, settings: Mapping[str, Any]) -> str | None:
    """Why the existing per-exposure masks cannot be reused, or ``None``.

    A mask is stale when the settings recorded with the masks differ from *settings*
    (or were not recorded), or when the RV it was shifted by (its header, 4
    decimals) differs from the run's RV table.
    """
    recorded = _recorded_linemask_settings(layout)
    if recorded is None:
        return f"there is no record of how they were built ({layout.linemask_settings.name})"
    changed = sorted(k for k in set(recorded) | set(settings) if recorded.get(k) != settings.get(k))
    if changed:
        what = ", ".join(f"{k} {recorded.get(k)!r} -> {settings.get(k)!r}" for k in changed)
        return f"they were built with other settings ({what})"
    for row in rvmod.read_rv_table(rv_path):
        mask = layout.exposure_linemask(row["file"])
        if not mask.exists():
            continue
        with open(mask) as fh:
            head = fh.read(2000)
        found = _MASK_RV.search(head)
        rv = float(row["rv_kms"])
        if found is not None and abs(float(found.group(1)) - rv) > 1e-4:
            return (
                f"{mask.name} was shifted by RV {float(found.group(1)):+.4f} km/s, "
                f"the RV table now gives {rv:+.4f} km/s"
            )
    return None


def ensure_linemasks(
    observation: Observation,
    config: FitConfig,
    settings: Settings,
    layout: RunLayout,
    products: StellarProducts,
    telluric_transmission: LineTransmission | None = None,
) -> str:
    """Per-exposure stellar-line masks (``mask`` treatment): ``user_file``, ``reused``
    or ``generated``.

    The windows come from ``stellar_linemask.user_file``, or from the spectrum chosen
    by ``stellar_linemask.from_spectrum``; lines measured on the master are judged
    against *telluric_transmission* (the run's model, needed unless ``blend =
    "off"``). The settings the masks were built with are recorded beside them
    (``RunLayout.linemask_settings``); masks are reused only if those settings and
    the RVs they were placed by are unchanged.
    """
    lm = config.masks.stellar_linemask
    wanted = _linemask_settings(config, products, telluric_transmission)
    rows = rvmod.read_rv_table(products.rv_table)
    if lm.user_file is not None:
        # user rest-frame windows are cheap to place: rewritten every run
        user_path = Path(lm.user_file).expanduser()
        write_exposure_linemasks(
            rows, read_windows(user_path), f"{USER_WINDOWS_SOURCE}{user_path.name}", layout
        )
        layout.linemask_settings.write_text(json.dumps(wanted, indent=1))
        return "user_file"
    missing = missing_exposure_linemasks(products.rv_table, layout)
    stale = None if missing else _stale_linemasks(products.rv_table, layout, wanted)
    if not missing and stale is None:
        return "reused"
    if config.on_missing_stellar_data == "error":
        if missing:
            raise FileNotFoundError(
                "per-exposure stellar-line masks not found (on_missing_stellar_data = 'error')"
            )
        raise DataError(
            f"the existing stellar-line masks cannot be reused: {stale}; set "
            "on_missing_stellar_data = 'generate' to rebuild them, or delete them"
        )
    if lm.from_spectrum == "template":
        if config.stellar_model.template_file is not None:
            log.warning(
                "stellar-line mask from the user-supplied template %s: an observed spectrum "
                "without telluric correction is not suitable, as its telluric lines would be "
                "masked as stellar lines",
                products.template_path.name,
            )
        write_exposure_linemasks(
            rows,
            build_stellar_linemask(products.template, lm),
            f"template:{products.template_path.name}",
            layout,
            shift=products.template_frame != "topocentric",
        )
    else:
        reuse_rest = layout.linemask_rest.exists() and _recorded_linemask_settings(layout) == wanted
        if not reuse_rest:
            _measure_master_lines(
                observation, config, settings, layout, rows, telluric_transmission
            )
        write_exposure_linemasks(
            rows,
            read_windows(layout.linemask_rest),
            f"measured:{layout.linemask_rest.name}",
            layout,
        )
    layout.linemask_settings.write_text(json.dumps(wanted, indent=1))
    return "generated"


def _master_telluric(
    transmission: LineTransmission, rows: Sequence[Mapping[str, str]]
) -> Callable[[Array], Array]:
    """The model transmission as it appears in the rest-frame master: the mean over the
    RV table's exposures of the model at ``lambda_rest * (1 + rv / c)`` and each
    exposure's airmass (the model's own where the table has none)."""

    def airmass(row: Mapping[str, str]) -> float:
        try:
            value = float(row.get("airmass") or "nan")
        except ValueError:
            value = float("nan")
        return value if np.isfinite(value) and value >= 1.0 else transmission.airmass

    shifts = [(float(r["rv_kms"]), airmass(r)) for r in rows]

    def at(wave_rest: Array) -> Array:
        wave = np.asarray(wave_rest, float)
        models = [transmission.at(wave * (1.0 + rv / C_KMS), airmass=am) for rv, am in shifts]
        return np.asarray(np.mean(models, axis=0), float)

    return at


def _measure_master_lines(
    observation: Observation,
    config: FitConfig,
    settings: Settings,
    layout: RunLayout,
    rows: list[dict[str, str]],
    transmission: LineTransmission | None,
) -> None:
    """Measure the stellar lines on the master spectrum and write the rest-frame windows
    and the line catalogue.

    The master is built only if its file is missing; an existing master is reused
    as it is, even if the RVs or the ``master_spectrum`` settings have changed.
    """
    lm = config.masks.stellar_linemask
    from gtellcorr.preprocessing.master_spectrum import (
        build_master_spectrum,
        load_master_spectrum,
        save_master_spectrum,
    )
    from gtellcorr.preprocessing.stellar_lines import measure_stellar_lines, write_stellar_lines

    sm = config.stellar_model
    if not layout.master_npz.exists():
        master = build_master_spectrum(
            rows,
            directory=observation.directory,
            instrument=observation.instrument,
            continuum=config.master_spectrum.continuum,
            broad_windows=config.master_spectrum.broad_line_mask_nm,
            transit=observation.transit,
            formula=config.frames.vacuum_air_formula,
            ispec_dir=settings.ispec_dir,
        )
        save_master_spectrum(master["orders"], layout.master_npz, layout.master_s1d)
    catalogue, windows = measure_stellar_lines(
        load_master_spectrum(layout.master_npz),
        elements=sm.line_elements,
        min_depth=sm.line_min_depth,
        ispec_dir=settings.ispec_dir,
        linelist=sm.ispec.linelist,
        telluric=None if transmission is None else _master_telluric(transmission, rows),
        blend=lm.blend,
        blend_ratio=lm.blend_ratio,
        telluric_depth=lm.telluric_depth,
    )
    write_stellar_lines(
        catalogue,
        windows,
        catalog_path=layout.lines_catalog,
        rest_path=layout.linemask_rest,
        target=layout.target,
        elements=sm.line_elements,
        min_depth=sm.line_min_depth,
        master_path=layout.master_npz,
    )


def prepare_stellar(
    observation: Observation,
    config: FitConfig,
    settings: Settings,
    layout: RunLayout,
    *,
    rv_seed: int | None = 0,
    workers: int = 12,
    rv_telluric_windows: Sequence[Window] | None = None,
    telluric_transmission: LineTransmission | None = None,
) -> StellarProducts:
    """Template and RV table (both treatments), plus line masks for ``mask``.

    *rv_seed* seeds the RV measurement; *workers* is the number of processes for a
    template synthesis; *rv_telluric_windows* are the telluric model windows of the
    RV measurement (:func:`ensure_rv_table`); *telluric_transmission* is the run's
    telluric model the measured stellar lines are judged against
    (:func:`ensure_linemasks`). ``strong_windows`` is left empty for the pipeline to
    fill.
    """
    path, template_status = ensure_template(observation, config, settings, layout, workers=workers)
    template = load_stellar_template(path, observation, config)
    rv_path, rv_status = ensure_rv_table(
        observation, config, layout, template, seed=rv_seed, telluric_windows=rv_telluric_windows
    )
    rows = rvmod.read_rv_table(rv_path)
    products = StellarProducts(
        layout=layout,
        template_path=path,
        rv_table=rv_path,
        rv_kms={r["file"]: float(r["rv_kms"]) for r in rows},
        activity_windows=[(float(a), float(b)) for a, b in config.masks.activity_windows_nm],
        bad_regions=[(float(a), float(b)) for a, b in config.masks.bad_regions_nm],
        status={"template": template_status, "rv": rv_status},
        template_frame=config.stellar_model.template_frame,
        _template=template,
    )
    if config.stellar == "mask":
        products.status["linemask"] = ensure_linemasks(
            observation, config, settings, layout, products, telluric_transmission
        )
    return products
