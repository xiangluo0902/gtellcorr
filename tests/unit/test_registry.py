from __future__ import annotations

from dataclasses import dataclass

import pytest

import gtellcorr.registry as registry_module
from gtellcorr.errors import RegistryError
from gtellcorr.registry import Registry


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    obj: object = None
    error: Exception | None = None

    def load(self) -> object:
        if self.error is not None:
            raise self.error
        return self.obj


def test_register_and_get_is_case_insensitive() -> None:
    reg: Registry[int] = Registry("instrument")
    reg.register("LBT/PEPSI", 1)
    assert reg.get("LBT/PEPSI") == 1
    assert reg.get("lbt/pepsi") == 1
    assert reg.canonical_name("lbt/PEPSI") == "LBT/PEPSI"


def test_aliases_resolve_to_canonical_entry() -> None:
    reg: Registry[int] = Registry("instrument")
    reg.register("WIYN/NEID", 7, aliases=["NEID"])
    assert reg.get("neid") == 7
    assert reg.canonical_name("NEID") == "WIYN/NEID"
    assert "NEID" in reg
    assert reg.names() == ("WIYN/NEID",)  # aliases are not listed as names


def test_duplicate_name_or_alias_is_rejected() -> None:
    reg: Registry[int] = Registry("instrument")
    reg.register("WIYN/NEID", 1, aliases=["NEID"])
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("wiyn/neid", 2)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("KPNO/NEID", 2, aliases=["neid"])
    with pytest.raises(RegistryError, match="duplicate name"):
        reg.register("VLT/ESPRESSO", 3, aliases=["ESPRESSO", "espresso"])


def test_replace_swaps_entry_and_drops_old_aliases() -> None:
    reg: Registry[int] = Registry("site")
    reg.register("KPNO", 1, aliases=["Kitt Peak"])
    reg.register("KPNO", 2, replace=True)
    assert reg.get("KPNO") == 2
    assert "Kitt Peak" not in reg


def test_unknown_name_suggests_close_match() -> None:
    reg: Registry[int] = Registry("stellar treatment")
    reg.register("template", 1)
    reg.register("mask", 2)
    with pytest.raises(RegistryError, match=r"unknown stellar treatment 'tamplate'.*'template'"):
        reg.get("tamplate")


def test_unregister_removes_name_and_aliases() -> None:
    reg: Registry[int] = Registry("site")
    reg.register("LBT", 1, aliases=["Mt. Graham"])
    reg.unregister("mt. graham")
    assert "LBT" not in reg
    assert "Mt. Graham" not in reg
    assert len(reg) == 0


def test_decorator_registers_and_returns_object() -> None:
    reg: Registry[type] = Registry("reader")

    @reg.add("echelle_hdu")
    class EchelleReader:
        pass

    assert reg.get("echelle_hdu") is EchelleReader


def test_empty_name_is_rejected() -> None:
    with pytest.raises(RegistryError, match="non-empty"):
        Registry[int]("site").register("  ", 1)


def test_entry_points_are_loaded_lazily_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        calls.append(group)
        return [FakeEntryPoint("CAHA/CARMENES", "plugin:SPEC", obj="carmenes")]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[str] = Registry("instrument", entry_point_group="gtellcorr.instruments")
    reg.register("WIYN/NEID", "neid")  # built-ins register before plugins load
    assert calls == []
    assert reg.get("caha/carmenes") == "carmenes"
    assert reg.names() == ("CAHA/CARMENES", "WIYN/NEID")
    assert calls == ["gtellcorr.instruments"]


def test_failing_plugin_raises_with_plugin_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return [FakeEntryPoint("broken", "plugin:X", error=ImportError("no module"))]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[str] = Registry("instrument", entry_point_group="gtellcorr.instruments")
    with pytest.raises(RegistryError, match=r"plugin 'broken' from 'plugin:X'.*no module"):
        reg.names()


def test_plugin_clashing_with_builtin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return [FakeEntryPoint("wiyn/neid", "plugin:SPEC", obj="other")]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[str] = Registry("instrument", entry_point_group="gtellcorr.instruments")
    reg.register("WIYN/NEID", "neid")
    with pytest.raises(RegistryError, match="already registered"):
        reg.get("WIYN/NEID")


def test_contains_rejects_non_strings() -> None:
    reg: Registry[int] = Registry("site")
    reg.register("LBT", 1)
    assert 1 not in reg


def test_iteration_and_repr_list_canonical_names() -> None:
    reg: Registry[int] = Registry("site")
    reg.register("lbt", 1)
    reg.register("KPNO", 2)
    assert list(reg) == ["KPNO", "lbt"]
    assert repr(reg) == "Registry('site', entries=['KPNO', 'lbt'])"


def test_plugin_failure_is_raised_on_every_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return [
            FakeEntryPoint("good", "plugin:G", obj="g"),
            FakeEntryPoint("broken", "plugin:X", error=ImportError("no module")),
        ]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[str] = Registry("instrument", entry_point_group="gtellcorr.instruments")
    for _ in range(2):
        with pytest.raises(RegistryError, match="plugin 'broken'"):
            reg.names()
    assert reg._objects == {}  # nothing half-registered


def test_two_plugins_with_same_name_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return [FakeEntryPoint("X", "a:X", obj=1), FakeEntryPoint("x", "b:X", obj=2)]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[int] = Registry("instrument", entry_point_group="gtellcorr.instruments")
    with pytest.raises(RegistryError, match="by 'another plugin'"):
        reg.names()
