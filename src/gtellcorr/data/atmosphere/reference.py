"""Atmospheric reference files: the RFM MIPAS profile and CelesTrak space weather.

* **MIPAS** (``rfm_mipas_midlatitude_night_v3_20010103.atm``) is a fixed file; its
  SHA-256 is fixed in the LBLRTM setup description shipped with gtellcorr.
* **CelesTrak** ``SW-All.csv`` grows every day, so it is kept as dated copies
  (``celestrak_space_weather_sw_all_YYYYMMDD.csv``). The hash of the copy in use is
  recorded in ``celestrak_space_weather.json`` next to it and in each run's record
  of its inputs, never in the scientific configuration. Historical daily values are
  final observations and do not change between copies.

Only the standard library is used for downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gtellcorr.errors import ResourceError

__all__ = [
    "CELESTRAK_URL",
    "ReferenceFile",
    "fetch_celestrak",
    "fetch_mipas",
    "latest_celestrak",
    "record_celestrak",
]

CELESTRAK_URL = "https://celestrak.org/SpaceData/SW-All.csv"
CELESTRAK_RECORD = "celestrak_space_weather.json"
_TIMEOUT_S = 120


@dataclass(frozen=True)
class ReferenceFile:
    """A local reference file and the SHA-256 it must have."""

    path: Path
    sha256: str


def _sha256(data: bytes) -> str:
    """Hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    """The content at *url* (:class:`ResourceError` on failure)."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            data: bytes = response.read()
    except OSError as exc:
        raise ResourceError(f"cannot download {url}: {exc}") from exc
    return data


def fetch_mipas(
    directory: str | os.PathLike[str], *, file_name: str, sha256: str, url: str
) -> ReferenceFile:
    """Download the MIPAS profile into *directory* unless a verified copy exists."""
    target = Path(directory) / file_name
    if target.is_file() and _sha256(target.read_bytes()) == sha256:
        return ReferenceFile(target, sha256)
    data = _download(url)
    got = _sha256(data)
    if got != sha256:
        raise ResourceError(
            f"{url}: the downloaded file is not the expected one (SHA-256 {got}, expected {sha256})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return ReferenceFile(target, sha256)


def fetch_celestrak(
    directory: str | os.PathLike[str], *, url: str = CELESTRAK_URL, today: datetime | None = None
) -> ReferenceFile:
    """Download today's CelesTrak table and record its hash."""
    data = _download(url)
    if not data.startswith(b"DATE,"):
        raise ResourceError(f"{url}: unexpected content (not a CelesTrak SW-All table)")
    stamp = (today or datetime.now(UTC)).strftime("%Y%m%d")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"celestrak_space_weather_sw_all_{stamp}.csv"
    target.write_bytes(data)
    digest = _sha256(data)
    record = {
        "file": target.name,
        "sha256": digest,
        "source_url": url,
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (folder / CELESTRAK_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return ReferenceFile(target, digest)


def record_celestrak(path: str | os.PathLike[str], *, url: str = CELESTRAK_URL) -> ReferenceFile:
    """Register an existing CelesTrak table as the one to use (writes the record)."""
    target = Path(path)
    if not target.is_file():
        raise ResourceError(f"CelesTrak table not found: {target}")
    data = target.read_bytes()
    if not data.startswith(b"DATE,"):
        raise ResourceError(f"{target}: not a CelesTrak SW-All table")
    digest = _sha256(data)
    record = {"file": target.name, "sha256": digest, "source_url": url, "fetched_utc": None}
    (target.parent / CELESTRAK_RECORD).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return ReferenceFile(target, digest)


def latest_celestrak(directory: str | os.PathLike[str]) -> ReferenceFile:
    """The recorded CelesTrak table, checked against its recorded hash."""
    folder = Path(directory)
    record_path = folder / CELESTRAK_RECORD
    if not record_path.is_file():
        raise ResourceError(
            f"no CelesTrak table recorded in {folder}; run 'gtellcorr resources fetch celestrak'"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    target = folder / record["file"]
    if not target.is_file() or _sha256(target.read_bytes()) != record["sha256"]:
        raise ResourceError(f"{target} is missing or does not match {CELESTRAK_RECORD}")
    return ReferenceFile(target, str(record["sha256"]))
