from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gtellcorr.data.observation.observation import Exposure
from gtellcorr.data.observation.site import Site
from gtellcorr.errors import ConfigError
from gtellcorr.settings import Settings
from gtellcorr.telluric import Geometry, TelluricModel, transmission
from gtellcorr.telluric.order_cache import build_order_cache, use_lblrtm_forward_context


def test_geometry_from_exposure(tmp_path: Path) -> None:
    exposure = Exposure(
        path=tmp_path / "x.fits",
        time_utc=datetime(2022, 2, 13, tzinfo=UTC),
        site=Site(lat_deg=31.9, lon_deg=-111.6, alt_m=2091.0),
        site_sources={},
        airmass=1.93,
        airmass_source="fits_header",
        weather=None,
        resolution=110000.0,
        object="x",
    )
    assert Geometry.from_exposure(exposure) == Geometry(
        observatory=(31.9, -111.6, 2.091), airmass=1.93
    )


def test_settings_tau_cache(tmp_path: Path) -> None:
    assert (
        Settings(resource_root=tmp_path, cache_dir=tmp_path).tau_cache
        == tmp_path / "lblrtm_tau_cache"
    )
    custom = Settings.load(
        environ={
            "GTELLCORR_TAU_CACHE": str(tmp_path / "t"),
            "GTELLCORR_SETTINGS": str(tmp_path / "no"),
        }
    )
    assert custom.tau_cache == tmp_path / "t"


def test_ported_order_cache_has_no_hidden_defaults() -> None:
    with pytest.raises(RuntimeError, match=r"inside TelluricModel\.activate\(\)"):
        build_order_cache(
            {}, (600.0, 601.0), geometry=Geometry((0.0, 0.0, 0.0), 1.0), fit_molecules=["h2o"]
        )
    with (
        pytest.raises(TypeError, match="ResolvedLblrtmResourceFiles"),
        use_lblrtm_forward_context(resources=object(), cache_directory="/tmp"),
    ):
        pass


def test_forward_context_requires_a_known_continuum_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import gtellcorr.telluric.order_cache as oc

    class _Resources:
        def require_fully_validated(self) -> None:
            pass

    monkeypatch.setattr(oc, "ResolvedLblrtmResourceFiles", _Resources)
    common = {
        "resources": _Resources(),
        "cache_directory": "/tmp",
        "timeout_s": 10.0,
        "wave_margin_nm": 1.0,
        "lblrtm_verbose": False,
        "vacuum_air_formula": "espresso",
    }
    with (
        pytest.raises(ValueError, match="continuum_mode must be given"),
        use_lblrtm_forward_context(**common),
    ):
        pass
    with (
        pytest.raises(ValueError, match="continuum_mode must be 'full', 'scaled' or 'none'"),
        use_lblrtm_forward_context(**common, continuum_mode="partial"),
    ):
        pass
    with use_lblrtm_forward_context(**common, continuum_mode="none"):
        assert oc._require_lblrtm_forward_context().continuum_mode == "none"
    with use_lblrtm_forward_context(**common, continuum_mode="scaled"):
        assert oc._require_lblrtm_forward_context().continuum_mode == "scaled"


class _Model:
    """Stands in for TelluricModel; argument checks happen before it is used."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"model.{name} must not be reached")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"wave_nm": [600.0]}, ValueError, "strictly increasing"),
        ({"wave_nm": [601.0, 600.0]}, ValueError, "strictly increasing"),
        ({"relcol": {}}, ValueError, "at least one"),
        ({"relcol": {"h20": 1.0}}, ConfigError, "cannot be scaled"),
    ],
)
def test_transmission_argument_checks(kwargs: dict[str, Any], error: type, message: str) -> None:
    args: dict[str, Any] = {"wave_nm": np.linspace(600.0, 601.0, 5), **kwargs}
    with pytest.raises(error, match=message):
        transmission(
            _Model(), {}, args.pop("wave_nm"), geometry=Geometry((0.0, 0.0, 0.0), 1.0), **args
        )  # type: ignore[arg-type]


def test_model_repr_and_profile_contract(tmp_path: Path) -> None:
    class Verified:
        bundle = tmp_path / "bundle"
        resources = None

    reference = {"reference_provenance": {"source_sha256": "0" * 64}}
    model = TelluricModel(Verified(), reference, tau_cache=tmp_path / "tau")  # type: ignore[arg-type]
    assert "tau" in repr(model) and model.config.wave_margin_nm == 20.0
    with pytest.raises(KeyError):
        model.lblrtm_profile({"z_km": np.arange(3.0)})


@pytest.mark.parametrize("formula", ["espresso", "ciddor1996"])
def test_order_cache_uses_the_runs_vacuum_air_formula(formula: str, tmp_path: Path) -> None:
    from gtellcorr.telluric import order_cache as oc

    class Resources:
        def require_fully_validated(self) -> None:
            pass

    context = oc._LblrtmForwardContext(
        resources=Resources(),  # type: ignore[arg-type]
        cache_directory=tmp_path, timeout_s=10.0, wave_margin_nm=20.0, lblrtm_verbose=False,
        reference_atmosphere=None, vacuum_air_formula=formula,
    )  # fmt: skip
    token = oc._LBLRTM_FORWARD_CONTEXT.set(context)
    try:
        profile = {
            "z_km": np.array([0.0, 10.0]), "P_hPa": np.array([1000.0, 260.0]),
            "T_K": np.array([288.0, 223.0]),
        }  # fmt: skip
        arguments, _, _ = oc._prepare_order_cache_request(
            profile, (600.0, 601.0), geometry=Geometry((19.0, -155.0, 2.0), 1.2),
            fit_molecules=["h2o"], wave_margin_nm=None, lblrtm_verbose=None,
        )  # fmt: skip
    finally:
        oc._LBLRTM_FORWARD_CONTEXT.reset(token)
    assert arguments["wavelength_conversion_formula"] == formula
    assert arguments["airmass"] == 1.2 and arguments["wave_margin_nm"] == 0.0


def test_model_vacuum_air_formula(tmp_path: Path) -> None:
    class Verified:
        bundle = tmp_path / "bundle"
        resources = None

    reference = {"reference_provenance": {"source_sha256": "0" * 64}}
    model = TelluricModel(Verified(), reference, tau_cache=tmp_path / "tau")  # type: ignore[arg-type]
    assert model.vacuum_air_formula == "espresso" and "'espresso'" in repr(model)
    other = TelluricModel(
        Verified(), reference, tau_cache=tmp_path / "tau", vacuum_air_formula="ciddor1996"
    )  # type: ignore[arg-type]
    assert other.vacuum_air_formula == "ciddor1996"


def test_lblrtm_runner_logs_instead_of_printing() -> None:
    import ast
    import inspect
    import logging

    from gtellcorr.telluric.lblrtm import lblrtm_runner

    tree = ast.parse(inspect.getsource(lblrtm_runner))
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and hasattr(n.func, "id")]
    assert "print" not in calls
    assert lblrtm_runner._level(True) == logging.INFO
    assert lblrtm_runner._level(False) == logging.DEBUG


def test_order_cache_failure_is_one_plain_resource_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    from gtellcorr.errors import ResourceError
    from gtellcorr.telluric import order_cache
    from gtellcorr.telluric.lblrtm.lblrtm_errors import OpacityBasisCacheError

    def damaged(*args: Any, **kwargs: Any) -> Any:
        try:
            raise ValueError("cache file is truncated")
        except ValueError as exc:
            raise OpacityBasisCacheError("failed to build or load the cache") from exc

    model = TelluricModel(object(), {}, tau_cache=tmp_path)  # type: ignore[arg-type]
    monkeypatch.setattr(model, "activate", contextlib.nullcontext)
    monkeypatch.setattr(order_cache, "build_order_cache", damaged)
    geometry = Geometry((0.0, 0.0, 0.0), 1.0)
    with pytest.raises(ResourceError) as info:
        model.build_order_cache({}, (650.0, 680.5), geometry=geometry, fit_molecules=["h2o"])
    message = str(info.value)
    assert "650.00-680.50 nm" in message and str(tmp_path) in message
    assert message.endswith("Reason: cache file is truncated")


def test_gtellcorr_errors_pass_through_the_order_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    from gtellcorr.telluric import order_cache

    def bad_config(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError("fit_molecules: unknown molecule 'xx'")

    model = TelluricModel(object(), {}, tau_cache=tmp_path)  # type: ignore[arg-type]
    monkeypatch.setattr(model, "activate", contextlib.nullcontext)
    monkeypatch.setattr(order_cache, "build_order_cache", bad_config)
    with pytest.raises(ConfigError, match="unknown molecule"):
        model.build_order_cache(
            {}, (650.0, 680.0), geometry=Geometry((0.0, 0.0, 0.0), 1.0), fit_molecules=["xx"]
        )


def test_with_continuum_mode_copies_everything_but_the_mode(tmp_path: Path) -> None:
    from gtellcorr.telluric.config import TelluricConfig

    class Verified:
        bundle = tmp_path / "bundle"
        resources = None

    reference = {"reference_provenance": {"source_sha256": "0" * 64}}
    model = TelluricModel(
        Verified(),  # type: ignore[arg-type]
        reference,
        tau_cache=tmp_path / "tau",
        config=TelluricConfig(wave_margin_nm=5.0, continuum_mode="scaled"),
        timeout_s=42.0,
        verbose=True,
        vacuum_air_formula="ciddor1996",
    )
    full = model.with_continuum_mode("full")
    assert full is not model and model.config.continuum_mode == "scaled"
    assert full.config == TelluricConfig(wave_margin_nm=5.0, continuum_mode="full")
    for name in ("resources", "reference_atmosphere", "tau_cache", "timeout_s", "verbose"):
        assert getattr(full, name) == getattr(model, name), name
    assert full.vacuum_air_formula == "ciddor1996"


@pytest.mark.parametrize(
    ("mode", "mask_mode"), [("scaled", "full"), ("full", "full"), ("none", "none")]
)
def test_strong_mask_uses_a_full_cache_for_scaled_runs(
    mode: str, mask_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astropy.io import fits

    import gtellcorr.data.atmosphere as atmosphere
    import gtellcorr.data.observation.instrument as instrument
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.preprocessing import telluric_mask
    from gtellcorr.telluric.config import TelluricConfig

    seen: dict[str, Any] = {}

    def generate(model: TelluricModel, *args: Any, **kwargs: Any) -> str:
        seen["mode"] = model.config.continuum_mode
        return "mask"

    profile = dict.fromkeys(("z_km", "T_K", "P_hPa", "H2O_ppmv"), np.zeros(3))
    monkeypatch.setattr(atmosphere, "full_profile_for_exposure", lambda *a, **k: profile)
    monkeypatch.setattr(instrument, "read_resolution", lambda *a, **k: 130000.0)
    monkeypatch.setattr(fits, "getheader", lambda *a, **k: {})
    monkeypatch.setattr(TelluricModel, "lblrtm_profile", lambda self, profile: {})
    monkeypatch.setattr(telluric_mask, "generate_strong_telluric_mask", generate)
    exposure = Exposure(
        path=tmp_path / "x.fits",
        time_utc=datetime(2025, 6, 19, tzinfo=UTC),
        site=Site(lat_deg=32.7, lon_deg=-109.9, alt_m=3221.0),
        site_sources={},
        airmass=1.2,
        airmass_source="fits_header",
        weather=None,
        resolution=130000.0,
        object="x",
    )
    model = TelluricModel(
        object(),  # type: ignore[arg-type]
        {},
        tau_cache=tmp_path,
        config=TelluricConfig(continuum_mode=mode),  # type: ignore[arg-type]
    )
    result = telluric_mask.ensure_strong_telluric_mask(
        model,
        [exposure],
        instrument=None,  # type: ignore[arg-type]
        fit=FitConfig(),
        settings=None,
        out_dir=tmp_path,
        stem="x",
    )
    assert result == "mask" and seen["mode"] == mask_mode
    assert model.config.continuum_mode == mode


@pytest.mark.parametrize(("lsf_resolution", "expected"), [(None, 50000.0), (80000.0, 80000.0)])
def test_run_wide_telluric_models_use_the_start_resolution(
    lsf_resolution: float | None, expected: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strong-telluric mask and the RV's telluric windows are computed at the start R of
    the fit (``lsf.resolution``, else the exposures' header R), like the stellar template."""
    from types import SimpleNamespace

    from astropy.io import fits

    import gtellcorr.data.atmosphere as atmosphere
    import gtellcorr.data.observation.instrument as instrument
    from gtellcorr.config.fit_config import FitConfig
    from gtellcorr.preprocessing import telluric_mask
    from gtellcorr.telluric.config import TelluricConfig

    seen: dict[str, float] = {}

    def generate(model: TelluricModel, *args: Any, **kwargs: Any) -> str:
        seen["strong mask"] = kwargs["resolution"]
        return "mask"

    def line_transmission(model: TelluricModel, *args: Any, **kwargs: Any) -> str:
        seen["RV windows"] = kwargs["resolution"]
        return "lines"

    profile = dict.fromkeys(("z_km", "T_K", "P_hPa", "H2O_ppmv"), np.zeros(3))
    monkeypatch.setattr(atmosphere, "full_profile_for_exposure", lambda *a, **k: profile)
    monkeypatch.setattr(instrument, "read_resolution", lambda *a, **k: 50000.0)  # the header R
    monkeypatch.setattr(fits, "getheader", lambda *a, **k: {})
    monkeypatch.setattr(TelluricModel, "lblrtm_profile", lambda self, profile: {})
    monkeypatch.setattr(telluric_mask, "generate_strong_telluric_mask", generate)
    monkeypatch.setattr(telluric_mask, "line_transmission", line_transmission)
    exposure = Exposure(
        path=tmp_path / "x.fits",
        time_utc=datetime(2025, 4, 18, tzinfo=UTC),
        site=Site(lat_deg=32.7, lon_deg=-109.9, alt_m=3221.0),
        site_sources={},
        airmass=1.2,
        airmass_source="fits_header",
        weather=None,
        resolution=50000.0,
        object="x",
    )
    model = TelluricModel(
        object(),  # type: ignore[arg-type]
        {},
        tau_cache=tmp_path,
        config=TelluricConfig(continuum_mode="full"),
    )
    spec = SimpleNamespace(
        spectral=SimpleNamespace(wavelength_range_nm=(623.0, 742.0)),
        resolution=SimpleNamespace(default=130000.0),
    )
    fit = FitConfig()
    fit = fit.replace(lsf=fit.lsf.replace(resolution=lsf_resolution))
    telluric_mask.ensure_strong_telluric_mask(
        model, [exposure], instrument=spec, fit=fit, settings=None, out_dir=tmp_path, stem="x",
    )  # type: ignore[arg-type]  # fmt: skip
    telluric_mask.run_line_transmission(
        model, [exposure], instrument=spec, fit=fit, settings=None,  # type: ignore[arg-type]
    )  # fmt: skip
    assert seen == {"strong mask": expected, "RV windows": expected}
