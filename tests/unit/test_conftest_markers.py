"""Regression test for the marker-based skipping in tests/conftest.py."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("name", ["lblrtm", "network", "regression"])
def test_parameter_named_like_a_marker_is_not_skipped(name: str) -> None:
    assert name  # this test must run, not be skipped
