"""How many LBLRTM runs of one optical-depth cache to make at the same time.

The runs of a cache (one per fitted molecule, the fixed molecules and the
continua) are independent single-threaded programs, each in its own work
directory. ``Settings.lblrtm_workers`` sets how many run at once; ``None``
("auto") takes the CPUs this process may use, divided between the processes of
a run that build caches at the same time, and at most :data:`MAX_AUTO_WORKERS`.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MAX_AUTO_WORKERS",
    "auto_lblrtm_workers",
    "available_cpus",
    "resolve_lblrtm_workers",
]

#: Largest number of simultaneous LBLRTM runs chosen automatically (each needs
#: about 1 GB in TMPDIR while it runs).
MAX_AUTO_WORKERS = 8

#: CPU limit of the cgroup (v2) of this process, on Linux.
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def _cgroup_cpu_limit(path: Path) -> int | None:
    """Whole CPUs allowed by a cgroup v2 ``cpu.max`` file (``"<quota> <period>"``,
    or ``"max <period>"`` without limit); ``None`` without a readable limit."""
    try:
        fields = path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError):
        return None
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def available_cpus(*, cgroup_cpu_max: Path = CGROUP_CPU_MAX) -> int:
    """CPUs this process may use, at least 1.

    On Linux the CPUs of the affinity mask (which Slurm and ``taskset`` set),
    limited by the CPU quota of a cgroup v2 (containers, Slurm with cgroups);
    elsewhere ``os.cpu_count()``.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # macOS has no affinity masks
        cpus = os.cpu_count() or 1
    limit = _cgroup_cpu_limit(cgroup_cpu_max)
    if limit is not None:
        cpus = min(cpus, limit)
    return max(1, cpus)


def auto_lblrtm_workers(processes: int = 1) -> int:
    """Automatic number of simultaneous LBLRTM runs of each of *processes*
    processes that build caches at the same time: their share of
    :func:`available_cpus`, at most :data:`MAX_AUTO_WORKERS`, at least 1."""
    return max(1, min(available_cpus() // max(1, int(processes)), MAX_AUTO_WORKERS))


def resolve_lblrtm_workers(workers: int | None, runs: int) -> tuple[int, str]:
    """``(number of simultaneous runs, where it comes from)`` for a cache of
    *runs* LBLRTM runs; ``workers`` is ``Settings.lblrtm_workers`` (``None``:
    automatic)."""
    if workers is None:
        cpus = available_cpus()
        return (
            max(1, min(auto_lblrtm_workers(), runs)),
            f"lblrtm_workers = auto: {cpus} CPUs available, at most {MAX_AUTO_WORKERS}",
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"lblrtm_workers must be an integer >= 1 or None, got {workers!r}")
    return max(1, min(workers, runs)), f"lblrtm_workers = {workers}"
