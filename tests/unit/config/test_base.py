from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pytest

from gtellcorr.config.base import ConfigGroup, FrozenDict, Windows, check_windows, field_docs
from gtellcorr.errors import ConfigError


@dataclass(frozen=True, kw_only=True)
class Inner(ConfigGroup):
    level: int = 1
    """Inner level."""

    def validate(self) -> None:
        if self.level < 0:
            self.fail("level", "must be >= 0")


@dataclass(frozen=True, kw_only=True)
class Outer(ConfigGroup):
    name: str
    """Required name."""
    ratio: float = 0.5
    flag: bool = False
    mode: Literal["a", "b"] = "a"
    pair: tuple[float, float] = (1.0, 2.0)
    many: tuple[int, ...] = ()
    windows: Windows = ()
    table: Mapping[str, float] = field(default_factory=dict)
    inner: Inner = field(default_factory=Inner)
    choice: str | Inner | None = None
    optional: float | None = None

    def validate(self) -> None:
        check_windows(self, "windows", self.windows)


def test_coercion_from_toml_like_data() -> None:
    o = Outer.from_dict(
        {
            "name": "x",
            "ratio": 2,  # int -> float
            "pair": [3, 4],  # list -> tuple of floats
            "many": [1, 2],
            "windows": [[1, 2], [3, 4.5]],
            "table": {"a": 1},
            "inner": {"level": 3},
            "choice": {"level": 2},
        }
    )
    assert o.ratio == 2.0 and isinstance(o.ratio, float)
    assert o.pair == (3.0, 4.0)
    assert o.many == (1, 2)
    assert o.windows == ((1.0, 2.0), (3.0, 4.5))
    assert isinstance(o.table, FrozenDict) and o.table["a"] == 1.0
    assert o.inner == Inner(level=3)
    assert o.choice == Inner(level=2)
    assert Outer(name="x", choice="str-value").choice == "str-value"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "x", "ratio": "high"}, r"^ratio: expected a number, got 'high'"),
        ({"name": "x", "ratio": True}, r"^ratio: expected a number"),
        ({"name": "x", "flag": 1}, r"^flag: expected true/false"),
        ({"name": "x", "many": [1.5]}, r"^many\[0\]: expected an integer"),
        ({"name": "x", "pair": [1.0]}, r"^pair: expected 2 values"),
        ({"name": "x", "pair": "12"}, r"^pair: expected a list"),
        ({"name": "x", "mode": "c"}, r"^mode: must be one of 'a', 'b', got 'c'"),
        ({"name": "x", "inner": {"level": -1}}, r"^inner\.level: must be >= 0"),
        ({"name": "x", "inner": {"levle": 1}}, r"^inner\.levle: unknown parameter.*'level'"),
        ({"name": "x", "inner": 5}, r"^inner: expected a table"),
        ({"name": "x", "windows": [[2, 1]]}, r"^windows\[0\]: window must satisfy lo < hi"),
        ({"name": "x", "table": [1]}, r"^table: expected a table"),
        ({"name": "x", "ratio": float("nan")}, r"^ratio: must not be NaN"),
        ({"name": None}, r"^name: must not be empty"),
        ({"nme": "x"}, r"^nme: unknown parameter of Outer\. Did you mean 'name'\?"),
        ({}, r"^Outer: .*missing.*'name'"),
    ],
)
def test_errors_carry_field_path(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Outer.from_dict(data)


def test_from_dict_requires_a_mapping() -> None:
    with pytest.raises(ConfigError, match="expected a table"):
        Outer.from_dict([1, 2])  # type: ignore[arg-type]


def test_to_dict_omits_none_and_round_trips() -> None:
    o = Outer.from_dict({"name": "x", "windows": [[1, 2]], "table": {"k": 1.0}})
    plain = o.to_dict()
    assert "optional" not in plain and "choice" not in plain
    assert plain["windows"] == [[1.0, 2.0]]
    assert plain["inner"] == {"level": 1}
    assert Outer.from_dict(plain) == o


def test_groups_are_immutable_hashable_and_picklable() -> None:
    o = Outer(name="x", table={"k": 1.0})
    with pytest.raises(AttributeError):
        o.name = "y"  # type: ignore[misc]
    with pytest.raises(TypeError):
        o.table["k"] = 2.0  # type: ignore[index]
    assert hash(o) == hash(Outer(name="x", table={"k": 1.0}))
    assert pickle.loads(pickle.dumps(o)) == o


def test_replace_validates_and_rejects_unknown_fields() -> None:
    o = Outer(name="x")
    assert o.replace(ratio=3).ratio == 3.0
    with pytest.raises(ConfigError, match=r"^inner\.level"):
        o.replace(inner={"level": -2})
    with pytest.raises(ConfigError, match=r"unknown parameter.*'ratio'"):
        o.replace(ration=1)


def test_frozen_dict_behaves_like_a_read_only_mapping() -> None:
    d = FrozenDict({"a": 1})
    assert d == {"a": 1} and len(d) == 1 and list(d) == ["a"]
    assert repr(d) == "FrozenDict({'a': 1})"
    assert pickle.loads(pickle.dumps(d)) == d


def test_field_docs_reads_attribute_docstrings() -> None:
    docs = field_docs(Outer)
    assert docs["name"] == "Required name."
    assert "ratio" not in docs
    assert field_docs(Inner) == {"level": "Inner level."}
