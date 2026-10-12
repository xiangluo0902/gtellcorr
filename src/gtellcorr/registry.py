"""Generic name → object registry with entry-point plugins.

Every extension point in gtellcorr (instruments, sites, spectrum readers,
atmosphere backends, RT backends, stellar treatments, fit parameter blocks) is a
:class:`Registry`. Built-in entries register themselves on import; third-party
packages add entries through a Python entry-point group, loaded lazily the first
time the registry is queried::

    # pyproject.toml of a plugin package
    [project.entry-points."gtellcorr.instruments"]
    "CAHA/CARMENES" = "my_plugin.carmenes:SPEC"

Name lookup is case-insensitive, and aliases share the namespace of the
canonical names, so ``"LBT/PEPSI"``, ``"lbt/pepsi"`` and an alias ``"PEPSI"``
all resolve to the same entry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from gtellcorr.errors import RegistryError, did_you_mean

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of objects of one kind.

    Parameters
    ----------
    kind
        Human-readable name of what the registry holds (``"instrument"``),
        used in error messages.
    entry_point_group
        Entry-point group scanned for third-party entries, or ``None`` for a
        registry that only holds built-in entries.
    """

    def __init__(self, kind: str, *, entry_point_group: str | None = None) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._objects: dict[str, T] = {}  # folded canonical name -> object
        self._display: dict[str, str] = {}  # folded canonical name -> original spelling
        self._aliases: dict[str, str] = {}  # folded alias -> folded canonical name
        self._entry_points_loaded = entry_point_group is None

    # ------------------------------------------------------------------ writing
    def register(
        self,
        name: str,
        obj: T,
        *,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> T:
        """Register *obj* under *name* (and *aliases*); return *obj* unchanged.

        Raises :class:`~gtellcorr.errors.RegistryError` if *name* or an alias is
        already taken, unless ``replace=True`` and the clash is with the same
        canonical name.
        """
        if not name or not name.strip():
            raise RegistryError(f"{self.kind} name must be a non-empty string")
        key = name.casefold()
        alias_keys = [a.casefold() for a in aliases]
        if replace and key in self._objects:
            self.unregister(name)
        for candidate in (key, *alias_keys):
            owner = self._owner(candidate)
            if owner is not None:
                raise RegistryError(
                    f"{self.kind} name {candidate!r} is already registered "
                    f"(by {self._display[owner]!r})"
                )
        if len(set(alias_keys) | {key}) != len(alias_keys) + 1:
            raise RegistryError(f"{self.kind} {name!r}: duplicate name among aliases")
        self._objects[key] = obj
        self._display[key] = name
        for alias_key in alias_keys:
            self._aliases[alias_key] = key
        return obj

    def add(
        self, name: str, *, aliases: Iterable[str] = (), replace: bool = False
    ) -> Callable[[T], T]:
        """Decorator form of :meth:`register`."""

        def decorator(obj: T) -> T:
            """Register *obj* and return it unchanged."""
            return self.register(name, obj, aliases=aliases, replace=replace)

        return decorator

    def unregister(self, name: str) -> None:
        """Remove an entry and its aliases (mainly for tests and plugin reloads)."""
        key = self._resolve(name)
        del self._objects[key]
        del self._display[key]
        for alias_key in [a for a, target in self._aliases.items() if target == key]:
            del self._aliases[alias_key]

    # ------------------------------------------------------------------ reading
    def get(self, name: str) -> T:
        """Return the object registered under *name* or one of its aliases."""
        return self._objects[self._resolve(name)]

    def canonical_name(self, name: str) -> str:
        """Return the canonical spelling for *name* or one of its aliases."""
        return self._display[self._resolve(name)]

    def names(self) -> tuple[str, ...]:
        """Canonical names of all entries, sorted case-insensitively."""
        self._load_entry_points()
        return tuple(sorted(self._display.values(), key=str.casefold))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        self._load_entry_points()
        return self._owner(name.casefold()) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        self._load_entry_points()
        return len(self._objects)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, entries={list(self.names())})"

    # ------------------------------------------------------------------ internals
    def _owner(self, folded: str) -> str | None:
        """Canonical key that owns a folded name or alias, or ``None``."""
        if folded in self._objects:
            return folded
        return self._aliases.get(folded)

    def _resolve(self, name: str) -> str:
        """Canonical key of *name*; raises with suggestions if it is unknown."""
        self._load_entry_points()
        owner = self._owner(name.casefold()) if isinstance(name, str) else None
        if owner is None:
            known = [*self._display.values(), *self._aliases]
            hint = did_you_mean(name, known) if isinstance(name, str) else ""
            available = ", ".join(repr(n) for n in self.names()) or "none"
            raise RegistryError(f"unknown {self.kind} {name!r}.{hint} Available: {available}.")
        return owner

    def _load_entry_points(self) -> None:
        """Register the plugins of the entry-point group, once, all or none."""
        if self._entry_points_loaded:
            return
        # Mark first so a plugin that queries this registry while loading does not recurse.
        self._entry_points_loaded = True
        assert self.entry_point_group is not None
        try:
            # Load and check every plugin before registering any, so a failure leaves
            # the registry unchanged and is raised again on every later query.
            loaded: list[tuple[str, T]] = []
            seen: set[str] = set()
            for ep in entry_points(group=self.entry_point_group):
                try:
                    obj = ep.load()
                except Exception as exc:
                    raise RegistryError(
                        f"failed to load {self.kind} plugin {ep.name!r} from {ep.value!r}: {exc}"
                    ) from exc
                key = ep.name.casefold()
                owner = self._owner(key)
                if owner is not None or key in seen:
                    by = self._display[owner] if owner is not None else "another plugin"
                    raise RegistryError(
                        f"{self.kind} plugin {ep.name!r} from {ep.value!r}: name is already "
                        f"registered (by {by!r})"
                    )
                seen.add(key)
                loaded.append((ep.name, obj))
        except RegistryError:
            self._entry_points_loaded = False
            raise
        for name, obj in loaded:
            self.register(name, obj)
