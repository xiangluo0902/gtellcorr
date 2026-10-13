from __future__ import annotations

import pickle

import pytest

from gtellcorr.errors import ConfigError
from gtellcorr.fitting.config import (
    LSF,
    Continuum,
    Molecule,
    Molecules,
    MoleculeSelection,
    Wavelength,
)
from gtellcorr.telluric.molecules import ENGINE_MOLECULES, MOLECULE_NAMES


def test_reference_defaults() -> None:
    assert Continuum() == Continuum(order=5, sigma_clip=(2.0, 3.0), niter=5)
    lsf = LSF()
    # None: the instrument's resolution.fit_tol, else 0.08
    assert lsf.kind == "gauss" and lsf.fit_resolution and lsf.resolution_tol is None
    assert Wavelength() == Wavelength(fit=False, order=0, scope="order", bounds_kms=(-1.0, 1.0))
    assert dict(lsf.shape) == {"sgauss": 2.0, "voigt": 0.2}


@pytest.mark.parametrize(
    ("group", "data", "message"),
    [
        (Continuum, {"order": -1}, r"^order: must be >= 0"),
        (Continuum, {"sigma_clip": [0, 3]}, r"^sigma_clip: thresholds must be positive"),
        (Continuum, {"niter": 0}, r"^niter: must be >= 1"),
        (LSF, {"kind": "lorentz"}, r"^kind: must be one of"),
        (LSF, {"shape": {"gauss": 1.0}}, r"^shape\.gauss: unknown LSF kind"),
        (
            LSF,
            {"shape_bounds": {"voigt": [0.9, 0.1]}},
            r"^shape_bounds\.voigt: must satisfy lo < hi",
        ),
        (LSF, {"resolution_tol": 1.5}, r"^resolution_tol: must be in \(0, 1\)"),
        (LSF, {"resolution_bounds": [2e5, 1e5]}, r"^resolution_bounds: must satisfy 0 < lo < hi"),
        (Wavelength, {"scope": "exposure", "order": 1}, r"^scope: an exposure-wide correction"),
        (Wavelength, {"scope": "night"}, r"^scope: must be one of"),
        (Wavelength, {"bounds_kms": [0.1, 1.0]}, r"^bounds_kms: must satisfy lo < 0 < hi"),
        (Wavelength, {"order": -1}, r"^order: must be >= 0"),
        (Molecule, {"bounds": [0, 1]}, r"^bounds: must satisfy 0 < lo < hi"),
        (Molecule, {"relcol": 5.0, "bounds": [0.5, 2.0]}, r"^relcol: must lie within bounds"),
        (MoleculeSelection, {"add": ["oh"]}, r"^add: 'oh' cannot be fitted"),
        (MoleculeSelection, {"add": ["h2o"], "exclude": ["h2o"]}, r"^add: 'h2o' appears in both"),
    ],
)
def test_validation(group: type, data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        group.from_dict(data)


def test_default_molecule_table_matches_reference() -> None:
    mols = Molecules.default()
    assert tuple(mols) == MOLECULE_NAMES and len(mols) == 39
    assert mols.fitted == ("h2o", "o2")
    assert mols["h2o"].bounds == (0.01, 100.0)
    assert mols["o2"].bands_nm[2] == (759.0, 772.0)
    assert mols["o"].bands_nm == ()
    assert all(m.anchor_orders is None for m in mols.values())


def test_molecules_merge_partial_entries_onto_defaults() -> None:
    mols = Molecules.from_dict({"h2o": {"bounds": [0.1, 10.0]}, "co2": {"fit": True}})
    assert mols["h2o"].bounds == (0.1, 10.0)
    assert mols["h2o"].bands_nm == Molecules.default()["h2o"].bands_nm  # kept
    assert mols.fitted == ("h2o", "co2", "o2")  # LBLRTM ID order


def test_molecules_update_returns_new_table() -> None:
    base = Molecules.default()
    new = base.update(o2=Molecule(fit=False, bounds=(0.5, 1.5)), o3={"fit": True})
    assert base.fitted == ("h2o", "o2")
    assert new.fitted == ("h2o", "o3")
    assert new["o2"].bands_nm == ()  # a full Molecule replaces the entry


def test_molecules_reject_unknown_and_unfittable() -> None:
    with pytest.raises(ConfigError, match=r"^h20: unknown molecule\. Did you mean .*'h2o'"):
        Molecules.from_dict({"h20": {}})
    with pytest.raises(ConfigError, match=r"^oh\.fit: the telluric model can only fit"):
        Molecules.from_dict({"oh": {"fit": True}})
    with pytest.raises(ConfigError, match=r"^h2o\.bounds"):
        Molecules.from_dict({"h2o": {"bounds": [5, 1]}})
    with pytest.raises(ConfigError, match=r"^h2o: expected a table"):
        Molecules.from_dict({"h2o": 3})


def test_molecules_value_semantics() -> None:
    a, b = Molecules.default(), Molecules.from_dict({})
    assert a == b and hash(a) == hash(b) and a != "x"
    assert pickle.loads(pickle.dumps(a)) == a
    assert repr(a) == "Molecules(fitted=['h2o', 'o2'])"
    assert set(ENGINE_MOLECULES) <= set(a)
