"""Fitting engine on a synthetic optical-depth cache (no LBLRTM)."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import numpy as np
import pytest

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation import instruments
from gtellcorr.data.observation.spectrum import OrderSpectrum
from gtellcorr.fitting import (
    LSF,
    FitSpec,
    JointOrderInput,
    ModelState,
    build_fit_spec,
    default_blocks,
    evaluate_order,
    fit_joint_orders,
    fit_single_order,
    pack,
    parameter_blocks,
    unpack,
)
from gtellcorr.fitting.config import Wavelength
from gtellcorr.fitting.solver import joint_layout
from gtellcorr.fitting.wave_shift import C_KMS, corrected_wavelengths, offset_velocity, shifted_grid
from gtellcorr.telluric.order_cache import OrderCache

NEID = instruments.get("NEID")
GRID = np.linspace(689.0, 697.0, 16001)
LINES = {"h2o": (690.2, 691.5, 693.1, 694.8), "o2": (690.8, 692.4, 695.9)}


class FakeTau:
    """Transmission exp(-sum_m column_m * tau_m) with Gaussian lines per molecule."""

    def __init__(self) -> None:
        self.tau = {
            m: sum(0.8 * np.exp(-0.5 * ((GRID - c) / 0.004) ** 2) for c in centres)
            for m, centres in LINES.items()
        }

    def transmittance(self, **columns: float) -> np.ndarray:
        return np.exp(-sum(columns[m] * self.tau[m] for m in columns))


def cache(molecules: tuple[str, ...] = ("h2o", "o2")) -> OrderCache:
    return OrderCache(
        tau_cache=FakeTau(),
        wave_nm=GRID,
        fit_molecules=molecules,
        ref_column=dict.fromkeys(molecules, 1.0),
    )


def spec_for(order: int = 84, **lsf: Any) -> FitSpec:
    cfg = FitConfig(lsf=LSF(**lsf)) if lsf else FitConfig()
    return build_fit_spec((690.0, 696.0), config=cfg, instrument=NEID, order=order)


def spectrum(
    truth: dict[str, float],
    resolution: float,
    *,
    noise: float = 1e-3,
    seed: int = 1,
    wave_shift: tuple[float, ...] | None = None,
) -> OrderSpectrum:
    x = np.linspace(690.0, 696.0, 4000)
    spec = spec_for()
    state = ModelState(scales=truth, resolution=resolution, lsf_shape=None, wave_shift=wave_shift)
    flat = OrderSpectrum(
        wave_fit=x, wave_input=x, flux=np.ones_like(x), cont=np.ones_like(x),
        err=np.full_like(x, noise), airmass=1.2,
    )  # fmt: skip
    model = evaluate_order(
        flat, cache(), spec, state, weight=np.ones_like(x), keep_mask=np.ones(x.size, bool)
    ).telluric_on_data
    continuum = 1.0 + 0.05 * (x - 693.0)
    flux = continuum * model + np.random.default_rng(seed).normal(0.0, noise, x.size)
    return dataclasses.replace(flat, flux=flux)


TRUTH = {"h2o": 0.8, "o2": 1.1}


# ---------------------------------------------------------------------------
def test_pack_unpack_layout() -> None:
    spec = spec_for(kind="sgauss", fit_shape=True)
    theta0, lo, hi, names = pack(spec)
    assert names == ["relcol_h2o", "relcol_o2", "resolution", "lsf_shape"]
    assert theta0.tolist() == [1.0, 1.0, spec.resolution_init, 2.0]
    assert (lo[2], hi[2]) == spec.resolution_bounds and (lo[3], hi[3]) == (1.2, 4.0)
    state = unpack(np.array([0.5, 0.9, 1.1e5, 3.0]), spec)
    assert state == ModelState(scales={"h2o": 0.5, "o2": 0.9}, resolution=1.1e5, lsf_shape=3.0)
    fixed = dataclasses.replace(spec, fit_resolution=False, fit_lsf_shape=False)
    assert pack(fixed)[3] == ["relcol_h2o", "relcol_o2"]
    assert unpack(np.array([0.5, 0.9]), fixed).resolution == spec.resolution_init
    empty = dataclasses.replace(fixed, molecules=())
    assert pack(empty)[0].size == 0


def test_joint_layout_shares_exposure_blocks() -> None:
    a = spec_for(84)
    b = dataclasses.replace(
        spec_for(79),
        molecules=("h2o",),
        relcol_init={"h2o": 2.0},
        relcol_bounds={"h2o": (0.1, 5.0)},
    )
    layout = joint_layout({84: a, 79: b})
    assert layout.names == [
        "relcol_h2o", "relcol_o2", "order084_resolution", "order079_resolution",
    ]  # fmt: skip
    h2o_lo = max(a.relcol_bounds["h2o"][0], 0.1)
    assert layout.theta0[0] == 1.5 and layout.lower[0] == h2o_lo  # median, intersection
    states = layout.states(np.array([0.7, 1.2, 1.0e5, 1.1e5]))
    assert states[84].scales == {"h2o": 0.7, "o2": 1.2} and states[79].scales == {"h2o": 0.7}
    assert (states[84].resolution, states[79].resolution) == (1.0e5, 1.1e5)
    disjoint = dataclasses.replace(
        b, relcol_bounds={"h2o": (200.0, 300.0)}, relcol_init={"h2o": 250.0}
    )
    with pytest.raises(ValueError, match="do not overlap"):
        joint_layout({84: a, 79: disjoint})


def test_fit_single_order_recovers_truth() -> None:
    # R fixed at its true value: the reference LSF is not differentiable in R
    # (test_reference_lsf_is_a_staircase_in_resolution)
    spec = dataclasses.replace(spec_for(), fit_resolution=False)
    data = spectrum(TRUTH, spec.resolution_init)
    calls = []

    def factory() -> OrderCache:
        calls.append(1)
        return cache()

    result = fit_single_order(data, spec, factory)
    assert calls == [1] and result.status == "ok"
    assert result.relcol["h2o"] == pytest.approx(0.8, abs=5 * result.relcol_err["h2o"])
    assert result.relcol["o2"] == pytest.approx(1.1, abs=5 * result.relcol_err["o2"])
    assert result.resolution == spec.resolution_init and result.resolution_err is None
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.1)
    assert result.dof_mask_removed == 4000 - 2 - (FitConfig().continuum.order + 1)
    np.testing.assert_allclose(result.flux_corrected * result.tell_model, data.flux)
    assert result.exclusion is None and result.stellar_model is None


def test_fit_single_order_labels_no_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    from gtellcorr.fitting import fit_order

    real = fit_order.minimize

    def stopped(*args: Any, **kwargs: Any) -> Any:
        opt = real(*args, **kwargs)
        return dataclasses.replace(opt, success=False, message="max_nfev reached")

    monkeypatch.setattr(fit_order, "minimize", stopped)
    spec = dataclasses.replace(spec_for(), fit_resolution=False)
    result = fit_single_order(spectrum(TRUTH, spec.resolution_init), spec, cache())
    assert result.status == "not_converged:max_nfev reached"


def test_fit_single_order_with_stellar_model_and_exclusion() -> None:
    data = spectrum(TRUTH, 1.1e5)
    stellar = np.ones_like(data.flux)
    exclusion = np.zeros(data.flux.size, bool)
    exclusion[:100] = True
    result = fit_single_order(
        data, spec_for(kind="sgauss", fit_shape=True), cache(), stellar_model=stellar,
        exclusion=exclusion, stellar_treatment="template",
    )  # fmt: skip
    assert result.stellar_treatment == "template" and result.lsf_kind == "sgauss"
    assert result.lsf_shape is not None and result.lsf_shape_err is not None
    assert result.exclusion is not None and result.exclusion[:100].all()
    with pytest.raises(ValueError, match="must be one-dimensional"):
        fit_single_order(data, spec_for(), cache(), exclusion=exclusion[:10])


def test_fit_single_order_skips_without_calling_the_cache() -> None:
    data = spectrum(TRUTH, 1.1e5)

    def never() -> OrderCache:
        raise AssertionError("the cache must not be built")

    result = fit_single_order(data, dataclasses.replace(spec_for(), molecules=()), never)
    assert result.status == "no_fit_molecules"
    np.testing.assert_array_equal(result.flux_corrected, data.flux)
    result = fit_single_order(data, spec_for(), never, exclusion=np.ones(data.flux.size, bool))
    assert result.status.startswith("insufficient_fit_pixels")
    assert np.isnan(result.relcol["h2o"]) and np.isnan(result.chi2)


def test_fit_joint_orders_shares_relcol() -> None:
    inputs = []
    for order, seed in ((84, 1), (85, 2)):
        spec = dataclasses.replace(spec_for(order), fit_resolution=False)
        data = spectrum(TRUTH, spec.resolution_init, seed=seed)
        inputs.append(JointOrderInput.prepare(data, spec, cache()))
    joint = fit_joint_orders(inputs)
    assert joint.success
    assert joint.relcol["h2o"] == pytest.approx(0.8, abs=5 * joint.relcol_err["h2o"])
    assert joint.relcol["o2"] == pytest.approx(1.1, abs=5 * joint.relcol_err["o2"])
    assert set(joint.parameters) == {"relcol_h2o", "relcol_o2"}
    assert joint.dof_mask_removed == 8000 - 2 - 2 * (FitConfig().continuum.order + 1)
    for result in joint.order_results:
        assert result.status == "joint_ok"
        assert result.converged is True and result.n_eval == joint.n_eval
        assert result.relcol == joint.relcol and result.chi2_joint == joint.chi2
        assert result.relcol_err == joint.relcol_err
        assert result.dof_mask_removed_order == 4000 - (FitConfig().continuum.order + 1)


def test_fit_joint_orders_labels_no_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    from gtellcorr.fitting import fit_order

    real = fit_order.least_squares

    def stopped(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        result.success, result.message = False, "max_nfev reached"
        return result

    monkeypatch.setattr(fit_order, "least_squares", stopped)
    spec = dataclasses.replace(spec_for(84), fit_resolution=False)
    data = spectrum(TRUTH, spec.resolution_init)
    joint = fit_joint_orders([JointOrderInput.prepare(data, spec, cache())])
    assert not joint.success
    (result,) = joint.order_results
    assert result.status == "joint_failed:max_nfev reached" and result.converged is False


def test_joint_fit_with_resolution_names_order_parameters() -> None:
    inputs = [
        JointOrderInput.prepare(spectrum(TRUTH, 1.1e5, seed=s), spec_for(o), cache())
        for o, s in ((84, 1), (85, 2))
    ]
    joint = fit_joint_orders(inputs)
    assert set(joint.parameters) == {
        "relcol_h2o", "relcol_o2", "order084_resolution", "order085_resolution",
    }  # fmt: skip
    for result in joint.order_results:
        assert result.dof_mask_removed_order == 4000 - (FitConfig().continuum.order + 1) - 1


def test_reference_lsf_is_a_staircase_in_resolution() -> None:
    """Why fits use the fixed-grid convolution: the velocity grid of ``apply_lsf``
    scales with the FWHM, so R only enters through the integer number of grid points.
    The transmission is piecewise constant in R and a finite-difference Jacobian sees
    no dependence on R. ``apply_lsf`` is used only at a fixed R (strong-telluric
    mask); the fixed-grid convolution is smooth in R (test_lsf_grid.py).
    """
    from gtellcorr.fitting.lsf import apply_lsf

    trans = FakeTau().transmittance(h2o=1.0)
    base = apply_lsf(GRID, trans, 104919.0)
    step = 1.5e-8 * 104919.0  # least_squares' default finite-difference step
    assert np.max(np.abs(apply_lsf(GRID, trans, 104919.0 + step) - base)) < 1e-12
    assert np.max(np.abs(apply_lsf(GRID, trans, 104919.0 + 10.0) - base)) > 1e-4


def test_fit_joint_orders_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_joint_orders([])
    data = spectrum(TRUTH, 1.1e5)
    empty = JointOrderInput.prepare(data, dataclasses.replace(spec_for(), molecules=()), cache())
    with pytest.raises(ValueError, match="no molecules"):
        fit_joint_orders([empty])
    masked = JointOrderInput.prepare(
        data, spec_for(), cache(), exclusion=np.ones(data.flux.size, bool)
    )
    with pytest.raises(ValueError, match="insufficient kept pixels"):
        fit_joint_orders([masked])


class OffsetBlock:
    """A plug-in block: one exposure-scope parameter stored in ``extra``."""

    name: ClassVar[str] = "offset"
    scope: ClassVar[str] = "exposure"

    def active(self, spec: FitSpec) -> bool:
        return True

    def names(self, spec: FitSpec) -> tuple[str, ...]:
        return ("offset",)

    def initial(self, spec: FitSpec) -> np.ndarray:
        return np.array([0.0])

    def bounds(self, spec: FitSpec) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])

    def apply(self, values: np.ndarray, spec: FitSpec, state: ModelState) -> ModelState:
        return dataclasses.replace(state, extra={**state.extra, "offset": float(values[0])})


def test_plugin_blocks_extend_theta_without_solver_changes() -> None:
    blocks = (*default_blocks(), OffsetBlock())
    spec = spec_for()
    assert pack(spec, blocks)[3][-1] == "offset"
    assert unpack(np.array([1.0, 1.0, 1.1e5, 0.25]), spec, blocks).extra == {"offset": 0.25}
    layout = joint_layout({84: spec, 85: spec_for(85)}, blocks)
    assert layout.names.count("offset") == 1  # shared across orders
    assert {"relcol", "resolution", "lsf_shape"} <= set(parameter_blocks.names())


# ---------------------------------------------------------------------------
# wavelength offset
# ---------------------------------------------------------------------------
def shift_spec(echelle: int = 84, **wavelength: Any) -> FitSpec:
    cfg = FitConfig(wavelength=Wavelength(fit=True, **wavelength))
    spec = build_fit_spec((690.0, 696.0), config=cfg, instrument=NEID, order=echelle)
    return dataclasses.replace(spec, fit_resolution=False)


def test_offset_velocity_and_grid() -> None:
    x = np.linspace(690.0, 696.0, 7)
    assert offset_velocity(x, [0.3]).tolist() == [0.3] * 7
    u = 2.0 * np.log(x / 690.0) / np.log(696.0 / 690.0) - 1.0
    np.testing.assert_allclose(offset_velocity(x, [0.1, 0.2]), 0.1 + 0.2 * u)
    # the Chebyshev argument is a velocity coordinate: u = 0 at the geometric mean
    np.testing.assert_allclose(
        offset_velocity(np.array([690.0, np.sqrt(690.0 * 696.0), 696.0]), [0.0, 1.0]),
        [-1.0, 0.0, 1.0],
        atol=1e-12,
    )
    with_nan = np.array([np.nan, 690.0, 696.0])
    np.testing.assert_allclose(offset_velocity(with_nan, [0.0, 1.0])[1:], [-1.0, 1.0])
    assert shifted_grid(x, [0.0]).tobytes() == x.tobytes()
    fit_corr, input_corr = corrected_wavelengths(x, 2.0 * x, [0.2, -0.1])
    np.testing.assert_array_equal(fit_corr, shifted_grid(x, [0.2, -0.1]))
    np.testing.assert_allclose(input_corr, 2.0 * fit_corr, rtol=1e-15)
    np.testing.assert_allclose(shifted_grid(x, [C_KMS * 0.01]), x / 1.01)


def test_positive_offset_moves_telluric_lines_to_the_red() -> None:
    spec = dataclasses.replace(spec_for(), fit_resolution=False)
    blue = spectrum(TRUTH, spec.resolution_init, noise=1e-9)
    red = spectrum(TRUTH, spec.resolution_init, noise=1e-9, wave_shift=(3.0,))
    near = np.abs(blue.wave_fit - 690.2) < 0.05
    shift = blue.wave_fit[near][np.argmin(red.flux[near])] - 690.2
    assert shift == pytest.approx(690.2 * 3.0 / C_KMS, abs=2e-3)  # one pixel is 1.5 pm


def test_zero_offset_is_bitwise_the_unshifted_model() -> None:
    spec = spec_for()
    data = spectrum(TRUTH, 1.1e5)
    args = {"weight": np.ones(4000), "keep_mask": np.ones(4000, bool)}
    plain = ModelState(scales=TRUTH, resolution=1.1e5, lsf_shape=None)
    zero = dataclasses.replace(plain, wave_shift=(0.0,))
    a = evaluate_order(data, cache(), spec, plain, **args)
    b = evaluate_order(data, cache(), spec, zero, **args)
    assert a.residual.tobytes() == b.residual.tobytes()


def test_wave_shift_layout() -> None:
    spec = dataclasses.replace(shift_spec(order=1, bounds_kms=(-0.5, 0.8)), fit_resolution=True)
    theta0, lo, hi, names = pack(spec)
    assert names == ["relcol_h2o", "relcol_o2", "resolution", "wave_shift_c0", "wave_shift_c1"]
    assert theta0[3:].tolist() == [0.0, 0.0] and lo[3:].tolist() == [-0.5, -0.5]
    assert hi[3:].tolist() == [0.8, 0.8]
    assert unpack(np.array([1.0, 1.0, 1.1e5, 0.1, 0.2]), spec).wave_shift == (0.1, 0.2)
    assert pack(spec_for())[3] == ["relcol_h2o", "relcol_o2", "resolution"]  # off by default
    joint = joint_layout({84: shift_spec(84), 85: shift_spec(85)})
    assert joint.names == [
        "relcol_h2o",
        "relcol_o2",
        "order084_wave_shift_c0",
        "order085_wave_shift_c0",
    ]
    shared = joint_layout(
        {84: shift_spec(84, scope="exposure"), 85: shift_spec(85, scope="exposure")}
    )
    assert shared.names == ["relcol_h2o", "relcol_o2", "wave_shift_c0"]
    assert {"wave_shift", "shared_wave_shift"} <= set(parameter_blocks.names())


@pytest.mark.parametrize("truth", [(0.1,), (-0.06, 0.12)])
def test_fit_single_order_recovers_wave_shift(truth: tuple[float, ...]) -> None:
    spec = shift_spec(order=len(truth) - 1)
    data = spectrum(TRUTH, spec.resolution_init, wave_shift=truth)
    result = fit_single_order(data, spec, cache())
    assert result.status == "ok" and result.wave_shift_err is not None
    for fitted, err, true in zip(result.wave_shift, result.wave_shift_err, truth, strict=True):
        assert 0 < err < 0.03
        assert fitted == pytest.approx(true, abs=5 * err)
    assert result.relcol["h2o"] == pytest.approx(0.8, abs=5 * result.relcol_err["h2o"])
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.1)
    unfitted = fit_single_order(data, dataclasses.replace(spec, fit_wave_shift=False), cache())
    assert unfitted.wave_shift is None and unfitted.chi2 > result.chi2


def test_joint_fit_per_order_and_shared_wave_shift() -> None:
    truths = {84: (0.08,), 85: (-0.05,)}
    inputs = [
        JointOrderInput.prepare(
            spectrum(TRUTH, 1.1e5, seed=o, wave_shift=truths[o]), shift_spec(o), cache()
        )
        for o in truths
    ]
    joint = fit_joint_orders(inputs)
    assert set(joint.parameters) == {
        "relcol_h2o", "relcol_o2", "order084_wave_shift_c0", "order085_wave_shift_c0",
    }  # fmt: skip
    for result in joint.order_results:
        (err,) = result.wave_shift_err
        assert result.wave_shift[0] == pytest.approx(truths[result.order][0], abs=5 * err)
        assert result.dof_mask_removed_order == 4000 - (FitConfig().continuum.order + 1) - 1
    shared_inputs = [
        JointOrderInput.prepare(
            spectrum(TRUTH, 1.1e5, seed=o, wave_shift=(0.07,)),
            shift_spec(o, scope="exposure"),
            cache(),
        )
        for o in truths
    ]
    shared = fit_joint_orders(shared_inputs)
    assert set(shared.parameters) == {"relcol_h2o", "relcol_o2", "wave_shift_c0"}
    a, b = shared.order_results
    assert a.wave_shift == b.wave_shift and a.wave_shift_err == b.wave_shift_err
    assert a.wave_shift[0] == pytest.approx(0.07, abs=5 * a.wave_shift_err[0])
    assert a.dof_mask_removed_order == 4000 - (FitConfig().continuum.order + 1)
