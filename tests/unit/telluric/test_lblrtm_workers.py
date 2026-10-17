"""LBLRTM runs of an optical-depth cache made at the same time: the number of CPUs
(affinity mask, cgroup quota), the automatic number of runs, the setting, and the
order and failure handling of ``TauCache._run_all``. LBLRTM itself is replaced by a
stand-in; the cache built with several runs at a time is compared with the serial
one in the ``lblrtm`` integration test ``test_tau_cache_workers.py``."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from gtellcorr.errors import ConfigError
from gtellcorr.settings import Settings
from gtellcorr.telluric.lblrtm import workers as w
from gtellcorr.telluric.lblrtm.lblrtm_errors import LblrtmExecutionError
from gtellcorr.telluric.lblrtm.tau_cache import TauCache
from gtellcorr.telluric.model import _workers_hint


def _cpu_max(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cpu.max"
    path.write_text(text)
    return path


@pytest.fixture
def sixteen_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)


# ---- available CPUs -------------------------------------------------------------------
@pytest.mark.usefixtures("sixteen_cpus")
@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        (None, 16),  # no cgroup file (macOS, cgroup v1)
        ("max 100000\n", 16),  # cgroup without a CPU limit
        ("400000 100000\n", 4),
        ("150000 100000\n", 1),  # 1.5 CPUs: whole CPUs only
        ("50000 100000\n", 1),  # less than one CPU
        ("3200000 100000\n", 16),  # quota above the affinity mask
        ("garbage\n", 16),
        ("-1 100000\n", 16),
    ],
)
def test_available_cpus_affinity_and_cgroup(
    tmp_path: Path, cpu_max: str | None, expected: int
) -> None:
    path = tmp_path / "missing" if cpu_max is None else _cpu_max(tmp_path, cpu_max)
    assert w.available_cpus(cgroup_cpu_max=path) == expected


def test_available_cpus_affinity_mask_beats_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # e.g. a Slurm job of 6 CPUs on a 128-core node
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 4, 5, 6, 7, 8}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert w.available_cpus(cgroup_cpu_max=Path("/nonexistent")) == 6


@pytest.mark.parametrize(("cpu_count", "expected"), [(10, 10), (None, 1)])
def test_available_cpus_without_affinity(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected: int
) -> None:
    # macOS: no sched_getaffinity; os.cpu_count() may return None
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    assert w.available_cpus(cgroup_cpu_max=Path("/nonexistent")) == expected


# ---- automatic number -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("cpus", "processes", "expected"),
    [(20, 1, 8), (10, 1, 8), (6, 1, 6), (10, 3, 3), (16, 4, 4), (4, 6, 1), (1, 1, 1)],
)
def test_auto_lblrtm_workers(
    monkeypatch: pytest.MonkeyPatch, cpus: int, processes: int, expected: int
) -> None:
    monkeypatch.setattr(w, "available_cpus", lambda: cpus)
    assert w.auto_lblrtm_workers(processes) == expected


def test_resolve_lblrtm_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w, "available_cpus", lambda: 10)
    assert w.resolve_lblrtm_workers(None, 13) == (
        8,
        "lblrtm_workers = auto: 10 CPUs available, at most 8",
    )
    assert w.resolve_lblrtm_workers(None, 3)[0] == 3  # no more threads than runs
    assert w.resolve_lblrtm_workers(16, 13) == (13, "lblrtm_workers = 16")
    assert w.resolve_lblrtm_workers(2, 13) == (2, "lblrtm_workers = 2")
    for bad in (0, -1, True, 2.0, "2"):
        with pytest.raises(ValueError, match="lblrtm_workers"):
            w.resolve_lblrtm_workers(bad, 13)  # type: ignore[arg-type]


# ---- setting --------------------------------------------------------------------------
def test_setting_default_is_auto(isolated_env: dict[str, str]) -> None:
    assert Settings.load(environ=isolated_env).lblrtm_workers is None


def test_setting_priority(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text("lblrtm_workers = 2\n")
    env = {**isolated_env, "GTELLCORR_SETTINGS": str(settings_file)}
    assert Settings.load(environ=env).lblrtm_workers == 2
    env["GTELLCORR_LBLRTM_WORKERS"] = "3"
    assert Settings.load(environ=env).lblrtm_workers == 3
    assert Settings.load(environ=env, lblrtm_workers=5).lblrtm_workers == 5
    assert Settings.load(environ=env, lblrtm_workers="auto").lblrtm_workers is None
    env["GTELLCORR_LBLRTM_WORKERS"] = " Auto "
    assert Settings.load(environ=env).lblrtm_workers is None


@pytest.mark.parametrize("bad", ["0", "-1", "two", "2.5", 0, -3, 1.5, True])
def test_setting_rejects_invalid_values(isolated_env: dict[str, str], bad: object) -> None:
    with pytest.raises(ConfigError, match="lblrtm_workers"):
        Settings.load(environ=isolated_env, lblrtm_workers=bad)
    if isinstance(bad, str):
        with pytest.raises(ConfigError, match="lblrtm_workers"):
            Settings.load(environ={**isolated_env, "GTELLCORR_LBLRTM_WORKERS": bad})


def test_setting_in_file_rejects_invalid_values(
    tmp_path: Path, isolated_env: dict[str, str]
) -> None:
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text('lblrtm_workers = "many"\n')
    with pytest.raises(ConfigError, match="lblrtm_workers"):
        Settings.load(environ={**isolated_env, "GTELLCORR_SETTINGS": str(settings_file)})


# ---- runs at the same time ------------------------------------------------------------
class _FakeRuns:
    """Stand-in for ``TauCache._run_lblrtm``: returns its relative columns after a
    delay that makes the later runs finish first; records how many run at once."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.lock = threading.Lock()
        self.active = 0
        self.most_active = 0
        self.started: list[int] = []
        self.finished: list[int] = []

    def __call__(self, columns: dict[str, float], continuum_mode: str = "none") -> object:
        index = int(columns["i"])
        with self.lock:
            self.started.append(index)
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        try:
            time.sleep(0.05 if index == self.fail_at else 0.02 * (6 - index % 6))
            if index == self.fail_at:
                raise LblrtmExecutionError(f"run {index} failed")
            return (index, continuum_mode)
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(index)


def _cache_with(fake: _FakeRuns) -> TauCache:
    cache = TauCache.__new__(TauCache)
    cache._run_lblrtm = fake  # type: ignore[method-assign]
    return cache


RUNS = [({"i": float(i)}, "full" if i % 2 else "none") for i in range(12)]


@pytest.mark.parametrize("workers", [1, 3, 8, 50])
def test_run_all_keeps_the_order(workers: int) -> None:
    fake = _FakeRuns()
    results = list(_cache_with(fake)._run_all(RUNS, workers))
    assert results == [(i, "full" if i % 2 else "none") for i in range(12)]
    assert fake.most_active == min(workers, len(RUNS))


def test_run_all_failure_cancels_the_runs_not_started() -> None:
    fake = _FakeRuns(fail_at=1)
    with pytest.raises(LblrtmExecutionError, match="run 1 failed"):
        list(_cache_with(fake)._run_all(RUNS, 2))
    # the runs not started when run 1 failed were cancelled, and every started
    # run had ended (removing its work directory) before the error was raised
    assert len(fake.started) < len(RUNS)
    assert sorted(fake.finished) == sorted(fake.started)
    assert fake.active == 0


def test_run_all_stopped_by_the_caller_waits_for_the_running_runs() -> None:
    fake = _FakeRuns()
    results = _cache_with(fake)._run_all(RUNS, 3)
    assert next(results) == (0, "none")
    results.close()  # e.g. the first result was rejected
    assert len(fake.started) < len(RUNS)
    assert sorted(fake.finished) == sorted(fake.started)
    assert fake.active == 0


# ---- error advice ---------------------------------------------------------------------
def _chained(inner: BaseException) -> RuntimeError:
    try:
        try:
            raise inner
        except BaseException as exc:
            raise LblrtmExecutionError("LBLRTM exceeded the finite timeout") from exc
    except LblrtmExecutionError as exc:
        try:
            raise RuntimeError("failed to build the cache") from exc
        except RuntimeError as outer:
            return outer


def test_workers_hint() -> None:
    timeout = _chained(subprocess.TimeoutExpired("lblrtm", 600))
    full_disk = _chained(OSError(28, "No space left on device"))
    other = _chained(OSError(2, "No such file or directory"))
    assert "share the CPUs" in _workers_hint(timeout, None)
    assert "GTELLCORR_LBLRTM_WORKERS" in _workers_hint(timeout, 4)
    assert "1 GB in TMPDIR" in _workers_hint(full_disk, None)
    assert _workers_hint(other, None) == ""
    assert _workers_hint(timeout, 1) == ""  # one run at a time: nothing to reduce
