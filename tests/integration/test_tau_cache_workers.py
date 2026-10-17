"""A TauCache built with several LBLRTM runs at a time equals the serial one.

A ``scaled`` cache (fitted H2O and O2: 13 LBLRTM runs, all three run roles) is
built in the O2 A band once with one run at a time and once with six; the optical
depths, the grid, the cache key, the build record and the saved archive must be the
same byte for byte. The checksums of the raw LBLRTM output files (TAPE6, TAPE12) in
the build record are left out: they differ between any two runs of the same input,
also one at a time.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gtellcorr.data.observation.frames import air_to_vac
from gtellcorr.settings import Settings
from gtellcorr.telluric.lblrtm import load_lblrtm_resource_manifest, validate_lblrtm_resource_bundle
from gtellcorr.telluric.lblrtm.tau_cache import MOLECULES, TauCache

pytestmark = pytest.mark.lblrtm

FIXTURES = Path(__file__).parents[1] / "fixtures" / "lblrtm"
FORMULA = "espresso"
WINDOW_AIR_NM = (760.0, 763.0)
FIT = ("h2o", "o2")


def _profile() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "reference_profile.npz", allow_pickle=False) as archive:
        profile = {k: np.asarray(archive[k], dtype=np.float64) for k in ("z_km", "P_hPa", "T_K")}
        for m in MOLECULES:
            profile[f"{m.upper()}_ppmv"] = np.asarray(archive[f"{m.upper()}_ppmv"], np.float64)
    return profile


def _without_output_checksums(record: Any) -> Any:
    """*record* (a build record, or a part of it) without the checksums of the raw
    LBLRTM output files."""
    if isinstance(record, Mapping):
        return {
            key: _without_output_checksums(value)
            for key, value in record.items()
            if key not in ("tape6_sha256", "tape12_sha256")
        }
    if isinstance(record, (list, tuple)):
        return [_without_output_checksums(value) for value in record]
    return record


def _build(resources: Any, workers: int) -> TauCache:
    profile = _profile()
    state = json.loads((FIXTURES / "reference_state_definition.json").read_text())
    los = state["expected_tau_cache_identity"]["line_of_sight"]
    alt = float(los["observer_altitude_km"])
    vac = np.asarray(air_to_vac(np.asarray(WINDOW_AIR_NM), formula=FORMULA))
    return TauCache(
        lowfreq=1.0e7 / float(vac[-1]),
        highfreq=1.0e7 / float(vac[0]),
        pressure=float(np.interp(alt, profile["z_km"], profile["P_hPa"])),
        temperature=float(np.interp(alt, profile["z_km"], profile["T_K"])),
        airmass=float(los["effective_airmass"]),
        lat=float(los["reference_latitude_deg"]),
        alt=alt,
        vac2air=True,
        atmosphere_profile=profile,
        resources=resources,
        wavelength_conversion_formula=FORMULA,
        continuum_mode="scaled",
        timeout_s=600.0,
    ).build(fit_molecules=FIT, workers=workers)


@pytest.fixture(scope="module")
def caches() -> dict[int, TauCache]:
    settings = Settings.load()
    assert settings.lblrtm_bundle is not None
    resources = validate_lblrtm_resource_bundle(
        load_lblrtm_resource_manifest(), settings.lblrtm_bundle
    )
    return {workers: _build(resources, workers) for workers in (1, 6)}


def test_parallel_build_equals_serial(caches: dict[int, TauCache]) -> None:
    serial, parallel = caches[1], caches[6]
    assert serial.build_provenance is not None
    assert serial.build_provenance["run_count"] == 13
    assert parallel.cache_key == serial.cache_key
    assert _without_output_checksums(parallel.build_provenance) == _without_output_checksums(
        serial.build_provenance
    )
    assert parallel.wavegrid.tobytes() == serial.wavegrid.tobytes()
    assert parallel.tau_fixed.tobytes() == serial.tau_fixed.tobytes()
    assert parallel.tau_continuum.tobytes() == serial.tau_continuum.tobytes()
    for m in FIT:
        assert parallel.tau_unit[m].tobytes() == serial.tau_unit[m].tobytes()
        assert parallel.tau_cont_beta[m].tobytes() == serial.tau_cont_beta[m].tobytes()
        assert parallel.tau_cont_gamma[m].tobytes() == serial.tau_cont_gamma[m].tobytes()


def test_parallel_build_saves_the_same_archive(caches: dict[int, TauCache], tmp_path: Path) -> None:
    arrays, meta = {}, {}
    for workers, cache in caches.items():
        path = cache.save(tmp_path / f"workers{workers}.npz")
        with zipfile.ZipFile(path) as archive:
            arrays[workers] = {
                name: archive.read(name) for name in archive.namelist() if name != "_meta.npy"
            }
        with np.load(path, allow_pickle=False) as archive:
            meta[workers] = _without_output_checksums(json.loads(archive["_meta"].tobytes()))
    assert arrays[6] == arrays[1]
    assert meta[6] == meta[1]
