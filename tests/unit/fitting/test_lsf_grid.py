"""Fixed-grid LSF convolution and parameter scaling: the resolving power is fitted."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.fitting import JointOrderInput, fit_joint_orders, fit_single_order
from gtellcorr.fitting.lsf import apply_lsf
from gtellcorr.fitting.lsf_grid import apply_lsf_fixed_grid
from gtellcorr.fitting.solver import evaluate_order, theta_scale

from .test_engine import GRID, TRUTH, FakeTau, ModelState, OrderSpectrum, cache, spec_for

TRANS = FakeTau().transmittance(h2o=1.0, o2=1.0)


def _fixed(r: float, **kw: float | str) -> np.ndarray:
    return apply_lsf_fixed_grid(
        GRID, TRANS, r, grid_resolution=1.2e5, support_resolution=1.0e5, **kw
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(("kind", "shape"), [("gauss", 2.0), ("sgauss", 3.0), ("voigt", 0.3)])
def test_fixed_grid_is_smooth_in_resolution(kind: str, shape: float) -> None:
    r = 104919.0
    base = _fixed(r, kind=kind, shape=shape)
    # finite differences of very different steps agree: no staircase
    slopes = [
        np.linalg.norm(_fixed(r + h, kind=kind, shape=shape) - base) / h for h in (1e-3, 1e-1, 10.0)
    ]
    assert slopes[0] > 0
    assert slopes == pytest.approx([slopes[2]] * 3, rel=0.02)


def test_fixed_grid_agrees_with_reference_convolution() -> None:
    for r in (1.0e5, 1.1e5, 1.2e5):
        diff = np.abs(_fixed(r) - apply_lsf(GRID, TRANS, r))
        assert diff.max() < 2e-3  # same kernel; grids differ by the reference quantisation
    flat = np.ones_like(GRID)
    np.testing.assert_allclose(apply_lsf_fixed_grid(GRID, flat, 1.1e5), 1.0, rtol=1e-12)


def test_fixed_grid_argument_checks() -> None:
    with pytest.raises(ValueError, match="sgauss exponent"):
        _fixed(1.1e5, kind="sgauss", shape=0.0)
    with pytest.raises(ValueError, match="voigt"):
        _fixed(1.1e5, kind="voigt", shape=1.0)
    with pytest.raises(ValueError, match="unknown lsf kind"):
        _fixed(1.1e5, kind="svd")
    with pytest.raises(ValueError, match="positive"):
        apply_lsf_fixed_grid(GRID, TRANS, -1.0)


def test_spec_grid_and_parameter_scale() -> None:
    spec = spec_for()
    lo, hi = spec.resolution_bounds
    assert spec.lsf_grid() == (hi, lo)
    assert dataclasses.replace(spec, fit_resolution=False).lsf_grid() == (spec.resolution_init,) * 2
    assert theta_scale(np.array([1.0, 0.0, 1.1e5])).tolist() == [1.0, 1.0, 1.1e5]
    assert "convolution" not in FitConfig().lsf.to_dict() and "solver" not in FitConfig().to_dict()


def _data(noise: float, seed: int = 1, r: float = 1.08e5) -> OrderSpectrum:
    x = np.linspace(690.0, 696.0, 4000)
    flat = OrderSpectrum(
        wave_fit=x, wave_input=x, flux=np.ones_like(x), cont=np.ones_like(x),
        err=np.full_like(x, noise), airmass=1.2,
    )  # fmt: skip
    model = evaluate_order(
        flat, cache(), spec_for(), ModelState(scales=TRUTH, resolution=r, lsf_shape=None),
        weight=np.ones_like(x), keep_mask=np.ones(x.size, bool),
    ).telluric_on_data  # fmt: skip
    noisy = (1.0 + 0.05 * (x - 693.0)) * model + np.random.default_rng(seed).normal(
        0.0, noise, x.size
    )
    return dataclasses.replace(flat, flux=noisy)


def test_resolution_is_fitted() -> None:
    result = fit_single_order(_data(1e-3), spec_for(), cache())
    assert result.resolution_err is not None and result.resolution_err < 1e3
    assert result.resolution == pytest.approx(1.08e5, abs=5 * result.resolution_err)
    for m, truth in TRUTH.items():
        assert result.relcol[m] == pytest.approx(truth, abs=5 * result.relcol_err[m])
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.1)
    exact = fit_single_order(_data(1e-7), spec_for(), cache())
    assert exact.relcol["h2o"] == pytest.approx(0.8, rel=1e-5)
    assert exact.resolution == pytest.approx(1.08e5, rel=1e-5)


def test_joint_fit_with_resolution_has_errors() -> None:
    inputs = [
        JointOrderInput.prepare(_data(1e-3, seed=s), spec_for(o), cache())
        for o, s in ((84, 1), (85, 2))
    ]
    joint = fit_joint_orders(inputs)
    assert set(joint.relcol_err) == {"h2o", "o2"}
    for m, truth in TRUTH.items():
        assert joint.relcol[m] == pytest.approx(truth, abs=5 * joint.relcol_err[m])
    for result in joint.order_results:
        assert result.resolution == pytest.approx(1.08e5, abs=5 * result.resolution_err)
