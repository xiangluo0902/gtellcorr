"""Import iSpec from its installation directory.

iSpec is not an installable package: its directory is put on ``sys.path`` and the
module imported from there. Needed by the iSpec synthesis backend, the stellar-line
search (:mod:`~gtellcorr.preprocessing.stellar_lines`) and the continuum fit
(:mod:`~gtellcorr.preprocessing.continuum`); pySME templates do not need it.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from gtellcorr.errors import ConfigError

__all__ = ["import_ispec"]

log = logging.getLogger(__name__)


def import_ispec(ispec_dir: str | os.PathLike[str] | None, *, fork: bool = False) -> Any:
    """Return the ``ispec`` module from ``ispec_dir`` (``Settings.ispec_dir``).

    Sets ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` unless already set, so forked
    processes do not abort on macOS. ``fork`` also forces the multiprocessing start
    method to ``fork``, which iSpec's synthesis codes need on macOS. Raises
    :class:`~gtellcorr.errors.ConfigError` if ``ispec_dir`` is ``None``.

    iSpec's ``ispec.log`` module is replaced by an empty one: on import it sets the
    root logger to INFO, adds a second console handler (every message twice) and
    writes ``ispec.log`` in the working directory. iSpec's messages then follow the
    caller's logging configuration. Its import-time banner about missing compiled
    extensions becomes one log message.
    """
    if ispec_dir is None:
        raise ConfigError("iSpec is needed here; set ispec_dir in the gtellcorr settings")
    path = str(Path(ispec_dir).expanduser())
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    if fork:
        import multiprocessing as mp

        with contextlib.suppress(RuntimeError, ValueError):
            mp.set_start_method("fork", force=True)
    if path not in sys.path:
        sys.path.insert(0, path)
    sys.modules.setdefault("ispec.log", types.ModuleType("ispec.log"))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        import ispec
    if "Not optimized version loaded" in printed.getvalue():
        log.info("iSpec's compiled extensions are not built; it uses its slower Python versions")
    return ispec
