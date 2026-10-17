"""TOML format of gtellcorr files: schema version, migrations, and a readable writer.

This module depends on no parameter group, so every layer (configuration files,
instrument definitions, ...) can use it without import cycles. Every file starts
with ``schema_version``; when the format changes, the version is bumped and a
migration ``N -> N+1`` is appended to ``MIGRATIONS``, so files written by any
earlier release keep loading.
"""

from __future__ import annotations

import json
import logging
import math
import re
import tomllib
from collections.abc import Callable, Mapping
from typing import Any

from gtellcorr.config.base import ConfigGroup, field_docs
from gtellcorr.errors import ConfigError

__all__ = ["CURRENT_SCHEMA_VERSION", "MIGRATIONS", "dumps", "dumps_group", "loads", "migrate"]

CURRENT_SCHEMA_VERSION = 5

log = logging.getLogger(__name__)

#: PySME settings that schema version 1 accepted but never used: the options of the
#: stellar-parameter fit (not part of gtellcorr), plus ``wave_step_nm`` and
#: ``abundances``, which the synthesis did not read (it always used PySME's adaptive
#: grid and Grevesse 2007 abundances). Dropping them keeps version-1 results unchanged.
_V1_UNUSED_PYSME_KEYS = (
    "auto_mask", "mask_cont_above", "mask_line_below", "mask_bad_windows_nm",
    "mgb_wing_windows_nm", "linelist_mode", "line_select_method", "obs_continuum",
    "cscale_flag", "cscale_type", "vrad_flag", "wave_step_nm", "abundances",
)  # fmt: skip


def _v1_to_v2(doc: dict[str, Any]) -> dict[str, Any]:
    """Schema 1 -> 2: drop the PySME settings version 1 accepted but never used, and
    map ``on_missing_stellar_data = "fallback_mask"`` to ``"error"``.

    ``fallback_mask`` never fell back to the mask treatment: a missing template
    already stopped the run as with ``error`` (the RV measurement needs it).
    """
    fit = doc.get("fit", {})
    if isinstance(fit, dict) and fit.get("on_missing_stellar_data") == "fallback_mask":
        fit["on_missing_stellar_data"] = "error"
        log.info('schema 1 -> 2: on_missing_stellar_data "fallback_mask" is now "error"')
    pysme = doc.get("fit", {}).get("stellar_model", {}).get("pysme")
    if isinstance(pysme, dict):
        dropped = [key for key in _V1_UNUSED_PYSME_KEYS if pysme.pop(key, None) is not None]
        if dropped:
            log.info(
                "schema 1 -> 2: dropped unused fit.stellar_model.pysme settings %s",
                ", ".join(dropped),
            )
    return doc


def _v2_to_v3(doc: dict[str, Any]) -> dict[str, Any]:
    """Schema 2 -> 3: ``masks.strong_telluric.relcol`` is renamed ``h2o_relcol``.

    It scales only the H2O column of the mask model, unlike the fitted ``relcol`` of
    each molecule.
    """
    strong = doc.get("fit", {}).get("masks", {}).get("strong_telluric")
    if isinstance(strong, dict) and "relcol" in strong:
        strong["h2o_relcol"] = strong.pop("relcol")
        log.info("schema 2 -> 3: fit.masks.strong_telluric.relcol is now h2o_relcol")
    return doc


def _v3_to_v4(doc: dict[str, Any]) -> dict[str, Any]:
    """Schema 3 -> 4: ``lsf.convolution``, ``solver.scale_parameters`` and
    ``masks.strong_telluric.n_grid_per_nm`` are removed.

    The first two only offered the behaviour of the original implementation, in which
    the resolving power was not actually fitted: its convolution grid scaled with the
    FWHM, so the model was a staircase in R. Files with the current behaviour
    (``"fixed_grid"``, ``true``) are upgraded silently; the old values are refused.
    ``n_grid_per_nm`` had no effect (the mask model is sampled on the optical-depth
    grid) and is dropped whatever its value.
    """
    fit = doc.get("fit", {})
    if not isinstance(fit, dict):
        return doc
    strong = fit.get("masks", {}).get("strong_telluric")
    if isinstance(strong, dict) and "n_grid_per_nm" in strong:
        del strong["n_grid_per_nm"]
        log.info("schema 3 -> 4: dropped fit.masks.strong_telluric.n_grid_per_nm (no effect)")
    lsf = fit.get("lsf")
    if isinstance(lsf, dict) and "convolution" in lsf:
        if lsf["convolution"] != "fixed_grid":
            raise ConfigError(
                f"fit.lsf.convolution = {lsf['convolution']!r} is no longer supported: the "
                "original convolution did not fit the resolving power. Remove the key; R is "
                "fitted on a fixed velocity grid"
            )
        del lsf["convolution"]
        log.info("schema 3 -> 4: dropped fit.lsf.convolution (always the fixed grid)")
    solver = fit.get("solver")
    if isinstance(solver, dict) and "scale_parameters" in solver:
        if solver["scale_parameters"] is not True:
            raise ConfigError(
                "fit.solver.scale_parameters = false is no longer supported: the fit "
                "parameters are always scaled by their initial values. Remove the key"
            )
        del solver["scale_parameters"]
        log.info("schema 3 -> 4: dropped fit.solver.scale_parameters (always scaled)")
    if solver == {}:
        del fit["solver"]
    return doc


def _v4_to_v5(doc: dict[str, Any]) -> dict[str, Any]:
    """Schema 4 -> 5: ``spectral.airmass_card`` of instrument files is removed.

    The airmass of an exposure is read with ``header.airmass``, which tries several
    cards (e.g. one per ESPRESSO unit telescope); the reader's single card was not
    used, and named the wrong telescope for three of the four VLT units.
    """
    spectral = doc.get("instrument", {}).get("spectral")
    if isinstance(spectral, dict) and "airmass_card" in spectral:
        del spectral["airmass_card"]
        log.info("schema 4 -> 5: dropped instrument.spectral.airmass_card (header.airmass is used)")
    return doc


#: ``MIGRATIONS[n]`` upgrades a raw document from schema version ``n`` to ``n + 1``.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
    3: _v3_to_v4,
    4: _v4_to_v5,
}


def migrate(doc: dict[str, Any], version: int) -> dict[str, Any]:
    """Upgrade raw tables written with schema *version* to the current schema."""
    while version < CURRENT_SCHEMA_VERSION:
        doc = MIGRATIONS[version](doc)
        version += 1
    return doc


def loads(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Parse TOML text and migrate it to the current schema; return the raw tables."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML: {exc}") from exc
    version = doc.pop("schema_version", None)
    if version is None:
        raise ConfigError(f"{source}: missing 'schema_version'")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ConfigError(f"{source}: schema_version must be a positive integer, got {version!r}")
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"{source}: schema_version {version} is newer than this gtellcorr supports "
            f"({CURRENT_SCHEMA_VERSION}); upgrade gtellcorr"
        )
    return migrate(doc, version)


def dumps(tables: dict[str, Any], *, source: Any = None, header: str = "") -> str:
    """Serialise plain tables to TOML with the current ``schema_version`` first.

    A small dedicated writer (instead of a generic TOML library) keeps the files
    readable: arrays stay on one line when they fit in ``_LINE_WIDTH`` characters.
    If *source* (the config object the tables came from) is given, every field is
    preceded by its documentation as a ``#`` comment. The output is always parsed
    back and compared, so the writer cannot silently lose information.
    """
    doc = {"schema_version": CURRENT_SCHEMA_VERSION, **tables}
    lines = [f"# {line}".rstrip() for line in header.splitlines()]
    root = None if source is None else source if isinstance(source, _Root) else _Root(source)
    _emit_table(doc, [], lines, root, None, set())
    text = "\n".join(lines).strip() + "\n"
    if tomllib.loads(text) != doc:
        raise AssertionError("TOML writer round-trip mismatch")  # pragma: no cover
    return text


def dumps_group(table: str, group: ConfigGroup, *, comments: bool = True, header: str = "") -> str:
    """TOML text holding ``schema_version`` and one ``[table]`` built from *group*."""
    source = _Root({table: group}) if comments else None
    return dumps({table: group.to_dict()}, source=source, header=header)


class _Root:
    """Wraps a mapping of top-level objects so the emitter can look children up."""

    def __init__(self, children: Any) -> None:
        self.children = children


def _child(obj: Any, key: str) -> Any:
    """The config object behind table *key* of *obj* (for its documentation)."""
    if obj is None:
        return None
    if isinstance(obj, _Root):
        inner = obj.children
        return inner.get(key) if isinstance(inner, Mapping) else getattr(inner, key, None)
    if isinstance(obj, ConfigGroup):
        return getattr(obj, key, None)
    if isinstance(obj, Mapping):
        return obj.get(key)
    return None


def _doc_for(obj: Any, key: str, seen: set[type]) -> str | None:
    """Documentation of field *key* of *obj*; each group type is documented once."""
    if isinstance(obj, _Root) and isinstance(obj.children, ConfigGroup):
        obj = obj.children
    if isinstance(obj, ConfigGroup) and type(obj) not in seen:
        return field_docs(type(obj)).get(key)
    return None


def _comment(doc: str | None, lines: list[str]) -> None:
    """Append *doc* as ``#`` comment lines."""
    if doc:
        lines.extend(f"# {line}".rstrip() for line in doc.splitlines())


_LINE_WIDTH = 88
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(key: str) -> str:
    """A TOML key: bare when possible, quoted otherwise."""
    return key if _BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def _scalar(value: Any) -> str:
    """Inline TOML text of a value (numbers, strings, arrays, inline tables)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_key(k)} = {_scalar(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")  # pragma: no cover


def _emit_value(key: str, value: Any, lines: list[str]) -> None:
    """Append ``key = value``, wrapping a long array over several lines."""
    line = f"{_key(key)} = {_scalar(value)}"
    if len(line) <= _LINE_WIDTH or not isinstance(value, list) or not value:
        lines.append(line)
        return
    lines.append(f"{_key(key)} = [")
    if any(isinstance(v, list | Mapping) for v in value):
        lines.extend(f"    {_scalar(v)}," for v in value)
    else:  # pack scalars, several per line
        current = "   "
        for v in value:
            item = f" {_scalar(v)},"
            if len(current) + len(item) > _LINE_WIDTH:
                lines.append(current)
                current = "   "
            current += item
        lines.append(current)
    lines.append("]")


def _emit_table(
    table: Mapping[str, Any],
    path: list[str],
    lines: list[str],
    obj: Any,
    doc: str | None,
    seen: set[type],
) -> None:
    """Append one table: its scalar values first, then its sub-tables, recursively."""
    scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping)}
    tables = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    if path and (scalars or not tables or doc):
        lines.append("")
        _comment(doc, lines)
        lines.append("[" + ".".join(_key(p) for p in path) + "]")
    for key, value in scalars.items():
        _comment(_doc_for(obj, key, seen), lines)
        _emit_value(key, value, lines)
    child_docs = {key: _doc_for(obj, key, seen) for key in tables}
    if isinstance(obj, ConfigGroup):
        seen.add(type(obj))
    for key, value in tables.items():
        _emit_table(value, [*path, key], lines, _child(obj, key), child_docs[key], seen)
