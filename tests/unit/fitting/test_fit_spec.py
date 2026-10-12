from __future__ import annotations

import pytest

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.data.observation import instruments
from gtellcorr.errors import ConfigError, DataError
from gtellcorr.fitting import (
    LSF,
    MoleculeSelection,
    build_fit_spec,
    configured_fit_molecules,
    select_fit_molecules,
    select_fit_orders_for_molecules,
)
from gtellcorr.fitting.config import Wavelength
from gtellcorr.fitting.fit_spec import DEFAULT_RESOLUTION_TOL, anchor_orders, default_resolution_tol

NEID = instruments.get("NEID")
RANGE_H2O = (720.0, 730.0)  # inside an h2o band, no o2 band


def test_anchor_orders_come_from_the_instrument_unless_configured() -> None:
    cfg = FitConfig()
    pins = anchor_orders(cfg, NEID)
    assert pins["o2"] == (83, 84, 85, 92, 93) and 79 in pins["h2o"]
    cfg = cfg.replace(molecules=cfg.molecules.update(o2={"anchor_orders": [92]}))
    assert anchor_orders(cfg, NEID)["o2"] == (92,)
    cfg = cfg.replace(molecules=cfg.molecules.update(o2={"anchor_orders": []}))
    assert "o2" not in anchor_orders(cfg, NEID)  # () = no pins, select by bands


def test_pinned_molecules_are_fitted_only_in_their_orders() -> None:
    cfg = FitConfig()
    assert select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=79) == ("h2o",)
    assert select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=92) == ("o2",)
    assert select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=84) == ("h2o", "o2")
    assert select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=10) == ()
    # without an order number the wavelength bands decide
    assert select_fit_molecules((686.0, 695.0), config=cfg, instrument=NEID) == ("h2o", "o2")


def test_add_and_exclude() -> None:
    cfg = FitConfig(molecule_selection=MoleculeSelection(add=("co2",), exclude=("o2",)))
    with pytest.warns(RuntimeWarning, match="fitted in all requested orders"):
        mols = select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=84)
    assert mols == ("h2o", "co2")
    assert configured_fit_molecules(cfg) == ("h2o", "co2")
    # an added molecule with anchors follows them
    cfg = FitConfig(molecule_selection=MoleculeSelection(add=("o3",)))
    cfg = cfg.replace(molecules=cfg.molecules.update(o3={"anchor_orders": [50]}))
    assert "o3" in select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=50)
    assert "o3" not in select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=51)
    assert "o3" in select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID)
    # excluded wins over added
    cfg = FitConfig(molecule_selection=MoleculeSelection(add=("co2",), exclude=("h2o",)))
    with pytest.warns(RuntimeWarning):
        assert "h2o" not in select_fit_molecules(RANGE_H2O, config=cfg, instrument=NEID, order=79)


def test_joint_order_selection() -> None:
    cfg = FitConfig()
    ranges = {o: RANGE_H2O for o in (78, 79, 83, 92, 5)}
    assert select_fit_orders_for_molecules(ranges, config=cfg, instrument=NEID) == [78, 79, 83, 92]
    with pytest.raises(DataError, match="none are available"):
        select_fit_orders_for_molecules({5: RANGE_H2O}, config=cfg, instrument=NEID)
    unpinned = cfg.replace(molecules=cfg.molecules.update(o2={"anchor_orders": []}))
    ranges = {1: (686.0, 695.0), 2: (500.0, 510.0), 79: RANGE_H2O}
    with pytest.warns(RuntimeWarning, match="bands_nm overlap"):
        assert select_fit_orders_for_molecules(
            ranges, config=unpinned, instrument=NEID, molecules=["o2"]
        ) == [1]
    with pytest.raises(DataError, match="no overlap"):  # raised before any warning
        select_fit_orders_for_molecules(
            {2: (500.0, 510.0)}, config=unpinned, instrument=NEID, molecules=["o2"]
        )


def test_build_fit_spec_resolution_and_lsf() -> None:
    cfg = FitConfig()
    spec = build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84)
    tol = NEID.resolution.fit_tol  # lsf.resolution_tol unset: the instrument's tolerance
    assert tol == 0.15
    r0 = NEID.resolution.default
    assert spec.resolution_init == r0
    assert spec.resolution_bounds == (r0 * (1.0 - tol), r0 * (1.0 + tol))
    assert (spec.lsf_kind, spec.lsf_shape, spec.fit_lsf_shape) == ("gauss", None, False)
    assert spec.relcol_bounds == {
        "h2o": cfg.molecules["h2o"].bounds,
        "o2": cfg.molecules["o2"].bounds,
    }
    assert spec.molecules == ("h2o", "o2")
    assert not spec.fit_wave_shift and spec.wave_shift is None


def test_configured_resolution_is_the_initial_and_fixed_r() -> None:
    cfg = FitConfig().replace(lsf=LSF(resolution=1.2e5, resolution_tol=0.1))
    spec = build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84)
    assert spec.resolution_init == 1.2e5
    assert spec.resolution_bounds == pytest.approx((1.08e5, 1.32e5))
    fixed = FitConfig().replace(lsf=LSF(fit_resolution=False, resolution=1.15e5))
    spec = build_fit_spec(RANGE_H2O, config=fixed, instrument=NEID, order=84)
    assert not spec.fit_resolution and spec.lsf_grid() == (1.15e5, 1.15e5)
    # a configured R overrides the exposure's header R (argument)
    spec = build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84, resolution=1.1e5)
    assert spec.resolution_init == 1.2e5
    with pytest.raises(ConfigError):
        LSF(resolution=0.0)


def test_resolution_tol_precedence() -> None:
    explicit = FitConfig().replace(lsf=LSF(resolution_tol=0.05))
    assert default_resolution_tol(explicit, NEID) == 0.05
    assert default_resolution_tol(FitConfig(), NEID) == 0.15
    plain = NEID.replace(resolution=NEID.resolution.replace(fit_tol=None))
    assert default_resolution_tol(FitConfig(), plain) == DEFAULT_RESOLUTION_TOL == 0.08
    assert instruments.get("PEPSI").resolution.fit_tol is None


def test_build_fit_spec_wave_shift() -> None:
    cfg = FitConfig().replace(wavelength=Wavelength(fit=True, order=1, bounds_kms=(-0.5, 0.7)))
    spec = build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84)
    assert spec.fit_wave_shift and spec.wave_shift_degree == 1
    assert spec.wave_shift_scope == "order" and spec.wave_shift_bounds == (-0.5, 0.7)
    assert spec.continuum_order == cfg.continuum.order

    cfg = cfg.replace(lsf=LSF(kind="sgauss", fit_shape=True, resolution_bounds=(1e5, 1.2e5)))
    spec = build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84, resolution=1.1e5)
    assert spec.resolution_bounds == (1e5, 1.2e5) and spec.resolution_init == 1.1e5
    assert (spec.lsf_shape, spec.lsf_shape_bounds, spec.fit_lsf_shape) == (2.0, (1.2, 4.0), True)
    # gauss has no shape: fit_shape is ignored
    cfg = cfg.replace(lsf=LSF(kind="gauss", fit_shape=True))
    assert not build_fit_spec(RANGE_H2O, config=cfg, instrument=NEID, order=84).fit_lsf_shape
