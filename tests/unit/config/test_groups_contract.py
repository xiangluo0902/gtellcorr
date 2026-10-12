"""Contracts every parameter group must satisfy."""

from __future__ import annotations

import dataclasses
import types
import typing

import pytest

import gtellcorr.config.fit_config
import gtellcorr.config.run_config  # noqa: F401  (registers all groups)
from gtellcorr.config.base import ConfigGroup, field_docs


def _all_groups() -> list[type[ConfigGroup]]:
    seen: list[type[ConfigGroup]] = []
    stack = list(ConfigGroup.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls.__module__.startswith("gtellcorr.") and cls not in seen:
            seen.append(cls)
        stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda c: c.__qualname__)


GROUPS = _all_groups()


def test_all_groups_are_discovered() -> None:
    names = {g.__name__ for g in GROUPS}
    assert {"FitConfig", "RunConfig", "Continuum", "LSF", "DataConfig", "Star"} <= names


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.__name__)
def test_optional_fields_default_to_none(group: type[ConfigGroup]) -> None:
    # TOML has no null: a None value is written by omitting the key, so it can only
    # round-trip if None is also the default.
    hints = typing.get_type_hints(group)
    for f in dataclasses.fields(group):
        tp = hints[f.name]
        optional = typing.get_origin(tp) in (typing.Union, types.UnionType) and type(
            None
        ) in typing.get_args(tp)
        if optional:
            assert f.default is None, (
                f"{group.__name__}.{f.name} is optional but defaults to {f.default!r}"
            )


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.__name__)
def test_groups_are_frozen_and_keyword_only(group: type[ConfigGroup]) -> None:
    params = group.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    assert all(f.kw_only for f in dataclasses.fields(group))


def test_fit_config_fields_are_documented() -> None:
    from gtellcorr.config.fit_config import FitConfig

    docs = field_docs(FitConfig)
    for name in ("stellar", "relcol_mode", "on_missing_stellar_data"):
        assert docs[name]
