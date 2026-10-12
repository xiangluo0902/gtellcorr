"""Base machinery shared by every configuration parameter group.

A parameter group is a frozen, keyword-only dataclass deriving from
:class:`ConfigGroup`. The base class gives every group, for free:

* **type coercion and checking** driven by the field annotations — TOML lists
  become tuples, ints become floats where a float is declared, nested dicts
  become nested groups, and anything of the wrong type is rejected;
* **unknown-key rejection** with a spelling hint (``from_dict``);
* **immutability** — containers are converted to tuples and :class:`FrozenDict`;
* **TOML round-tripping** via :meth:`ConfigGroup.to_dict` (``None`` fields are
  omitted, since TOML has no null; every optional field therefore defaults to
  ``None`` — enforced by the test suite).

Every :class:`~gtellcorr.errors.ConfigError` message starts with the dotted path
of the offending field relative to the object being built, e.g.
``"continuum.order: must be >= 0, got -1"``; enclosing groups prepend their own
field names, so a full run configuration reports ``"fit.continuum.order: …"``.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import math
import textwrap
import types
import typing
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, Literal, TypeVar, Union, get_args, get_origin

from gtellcorr.errors import ConfigError, did_you_mean

__all__ = ["ConfigGroup", "ConfigValue", "FrozenDict", "Windows", "check_windows", "field_docs"]

Self = TypeVar("Self", bound="ConfigValue")
K = TypeVar("K")
V = TypeVar("V")

#: A list of ``(lo, hi)`` wavelength windows in nm.
Windows = tuple[tuple[float, float], ...]


class FrozenDict(Mapping[K, V]):
    """An immutable, hashable, picklable mapping (dictionary fields of config groups)."""

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[K, V] | None = None) -> None:
        self._data: dict[K, V] = dict(data or {})
        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (self._data,))


class ConfigValue:
    """Anything that can appear as a typed field of a group and round-trip through TOML.

    Subclasses implement both conversions; :class:`ConfigGroup` does it for every
    dataclass group, and a few special values (the molecule table) do it by hand.
    """

    @classmethod
    def from_dict(cls: type[Self], data: Mapping[str, Any]) -> Self:
        """Build the value from TOML data (tables become mappings, arrays lists)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Plain TOML data of the value, the inverse of :meth:`from_dict`."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigGroup(ConfigValue):
    """Base class of every configuration parameter group."""

    _hints_cache: ClassVar[dict[type, dict[str, Any]]] = {}

    def __post_init__(self) -> None:
        """Coerce every field to its annotated type, then run :meth:`validate`."""
        hints = _type_hints(type(self))
        for f in dataclasses.fields(self):
            value = coerce(getattr(self, f.name), hints[f.name], f.name)
            object.__setattr__(self, f.name, value)
        self.validate()

    def validate(self) -> None:
        """Check value ranges and cross-field constraints (override in subclasses).

        Raise :class:`ConfigError` via :meth:`fail` so the message carries the
        field path.
        """

    @staticmethod
    def fail(field: str, message: str) -> typing.NoReturn:
        """Raise a :class:`ConfigError` for *field* (used by :meth:`validate`)."""
        raise ConfigError(f"{field}: {message}")

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Names of the group's parameters, in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls: type[Self], data: Mapping[str, Any]) -> Self:
        """Build the group from a mapping, rejecting unknown keys."""
        if not isinstance(data, Mapping):
            raise ConfigError(f"expected a table of parameters, got {data!r}")
        valid = cls.field_names()  # type: ignore[attr-defined]
        for key in data:
            if key not in valid:
                raise ConfigError(
                    f"{key}: unknown parameter of {cls.__name__}.{did_you_mean(str(key), valid)} "
                    f"Valid parameters: {', '.join(valid)}."
                )
        try:
            return cls(**data)
        except TypeError as exc:  # missing required field
            raise ConfigError(f"{cls.__name__}: {exc}") from None

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form suitable for TOML (``None`` fields are omitted)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = to_plain(getattr(self, f.name))
            if value is not None:
                out[f.name] = value
        return out

    def replace(self: Self, **changes: Any) -> Self:
        """Return a copy with *changes* applied (validated like a new object)."""
        valid = type(self).field_names()  # type: ignore[attr-defined]
        for key in changes:
            if key not in valid:
                raise ConfigError(
                    f"{key}: unknown parameter of {type(self).__name__}.{did_you_mean(key, valid)}"
                )
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]


def check_windows(owner: ConfigGroup, field: str, windows: Windows) -> None:
    """Require every ``(lo, hi)`` window to satisfy ``lo < hi``."""
    for i, (lo, hi) in enumerate(windows):
        if not lo < hi:
            owner.fail(f"{field}[{i}]", f"window must satisfy lo < hi, got ({lo}, {hi})")


def field_docs(cls: type) -> dict[str, str]:
    """Attribute docstrings (the string literal right after a field) of a group class.

    Read from the source with :mod:`ast`, so the documentation lives next to each
    field in the code and is reused for commented TOML templates.
    """
    docs: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if klass is object or not dataclasses.is_dataclass(klass):
            continue
        try:
            source = textwrap.dedent(inspect.getsource(klass))
        except (OSError, TypeError):  # pragma: no cover - source unavailable
            continue
        body = ast.parse(source).body[0]
        assert isinstance(body, ast.ClassDef)
        statements = body.body
        for node, following in itertools.pairwise(statements):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                docs[node.target.id] = inspect.cleandoc(following.value.value)
    return docs


# ---------------------------------------------------------------------------
# Plain-data conversion
# ---------------------------------------------------------------------------
def to_plain(value: Any) -> Any:
    """Convert config values to TOML-friendly builtins (tuples → lists, groups → dicts)."""
    if isinstance(value, ConfigValue):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, tuple | list):
        return [to_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Annotation-driven coercion
# ---------------------------------------------------------------------------
def _type_hints(cls: type) -> dict[str, Any]:
    """Resolved field annotations of a group class (cached)."""
    cache = ConfigGroup._hints_cache
    if cls not in cache:
        cache[cls] = typing.get_type_hints(cls)
    return cache[cls]


def _describe(tp: Any) -> str:
    """Readable name of a type annotation, for error messages."""
    if isinstance(tp, type):
        return tp.__name__
    return str(tp).replace("typing.", "")


def coerce(value: Any, tp: Any, path: str) -> Any:
    """Convert *value* to the annotated type *tp* or raise ``ConfigError("<path>: …")``."""
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return value

    if origin is Union or origin is types.UnionType:
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: must not be empty")
        errors = []
        for arm in args:
            if arm is type(None):
                continue
            try:
                return coerce(value, arm, path)
            except ConfigError as exc:
                errors.append(str(exc))
        if len(errors) == 1:
            raise ConfigError(errors[0])
        raise ConfigError(f"{path}: expected {_describe(tp)}, got {value!r}")

    if value is None:
        raise ConfigError(f"{path}: must not be empty")

    if origin is Literal:
        for option in args:
            if value == option and type(value) is type(option):
                return option
        options = ", ".join(repr(a) for a in args)
        hint = did_you_mean(value, [str(a) for a in args]) if isinstance(value, str) else ""
        raise ConfigError(f"{path}: must be one of {options}, got {value!r}.{hint}")

    if isinstance(tp, type) and issubclass(tp, ConfigValue):
        if isinstance(value, tp):
            return value
        if isinstance(value, Mapping):
            try:
                return tp.from_dict(value)
            except ConfigError as exc:
                raise ConfigError(f"{path}.{exc}") from None
        raise ConfigError(f"{path}: expected a table of parameters, got {value!r}")

    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected true/false, got {value!r}")

    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected an integer, got {value!r}")

    if tp is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            result = float(value)
            if math.isnan(result):
                raise ConfigError(f"{path}: must not be NaN")
            return result
        raise ConfigError(f"{path}: expected a number, got {value!r}")

    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string, got {value!r}")

    if origin is tuple:
        if isinstance(value, str) or not isinstance(value, list | tuple):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} values, got {len(value)}: {value!r}")
        return tuple(
            coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True))
        )

    if origin in (Mapping, dict) or tp in (Mapping, dict):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a table, got {value!r}")
        key_tp, value_tp = args if args else (Any, Any)
        return FrozenDict(
            {
                coerce(k, key_tp, f"{path}"): coerce(v, value_tp, f"{path}.{k}")
                for k, v in value.items()
            }
        )

    raise TypeError(f"{path}: unsupported annotation {tp!r} in a config group")  # programming error
