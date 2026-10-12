from __future__ import annotations

import pytest

from gtellcorr.errors import (
    ConfigError,
    DataError,
    FitError,
    GtellcorrError,
    RegistryError,
    ResourceError,
    did_you_mean,
)


@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ConfigError, ValueError),
        (DataError, ValueError),
        (RegistryError, LookupError),
        (ResourceError, RuntimeError),
        (FitError, RuntimeError),
    ],
)
def test_errors_share_base_and_builtin(error: type[Exception], builtin: type[Exception]) -> None:
    assert issubclass(error, GtellcorrError)
    assert issubclass(error, builtin)


def test_did_you_mean_is_case_insensitive_and_keeps_spelling() -> None:
    assert did_you_mean("lbt/pepsy", ["LBT/PEPSI", "WIYN/NEID"]) == " Did you mean 'LBT/PEPSI'?"


def test_did_you_mean_returns_empty_without_match() -> None:
    assert did_you_mean("xyz", ["continuum", "lsf"]) == ""
