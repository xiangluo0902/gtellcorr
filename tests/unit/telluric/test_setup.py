from __future__ import annotations

import copy

import pytest

from gtellcorr.errors import ResourceError
from gtellcorr.telluric import resources
from gtellcorr.telluric.lblrtm import load_lblrtm_resource_manifest
from gtellcorr.telluric.resources import check_setup, load_setup, physics_fields


def test_shipped_setup_matches_the_shipped_manifest() -> None:
    manifest = load_lblrtm_resource_manifest()
    setup = load_setup()
    assert setup["physics"] == physics_fields(manifest.data)
    assert setup["derived_from"] == manifest.path.name
    check_setup(manifest.data, setup)


def test_host_specific_bytes_are_not_part_of_the_physics() -> None:
    data = copy.deepcopy(load_lblrtm_resource_manifest().data)
    data["products"]["lblrtm_executable"]["sha256"] = "0" * 64
    data["products"]["tape3"]["sha256"] = "1" * 64  # e.g. a different build timestamp
    data["build_contract"]["lblrtm_make_target"] = "linuxGNUdbl"
    check_setup(data, load_setup())


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("products", "tape3", "scientific_sha256"), "tape3.scientific_sha256"),
        (("sources", "aer_line_file_archive", "sha256"), "sources.aer_line_file_archive.sha256"),
        (("build_contract", "lblrtm_precision"), "precision.lblrtm"),
    ],
)
def test_physics_changes_are_rejected(path: tuple[str, ...], field: str) -> None:
    data = copy.deepcopy(load_lblrtm_resource_manifest().data)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "single" if field == "precision.lblrtm" else "f" * 64
    with pytest.raises(ResourceError, match=f"differs in: .*{field}"):
        check_setup(data, load_setup())


def test_unknown_setup() -> None:
    with pytest.raises(ResourceError, match="unknown LBLRTM setup description"):
        load_setup("nope.json")


def test_names_of_version_0_9_0_still_work() -> None:
    assert resources.load_contract is load_setup and resources.check_contract is check_setup
    assert resources.scientific_identity is physics_fields


def test_comparison_records_are_optional() -> None:
    from gtellcorr.telluric.lblrtm.lblrtm_resources import validate_lblrtm_resource_manifest

    data = copy.deepcopy(dict(load_lblrtm_resource_manifest().data))
    data.pop("golden_compatibility", None)
    data.pop("resolved_discrepancies", None)
    validate_lblrtm_resource_manifest(data)
