from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from gtellcorr.preprocessing._ispec import import_ispec

# What iSpec's ispec/log.py does on import.
_LOG_PY = """
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("ispec.log"))
"""


def test_ispec_import_leaves_logging_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = tmp_path / "ispec_dir" / "ispec"
    package.mkdir(parents=True)
    banner = 'print("*" * 69)\nprint("Not optimized version loaded!")\n'
    (package / "__init__.py").write_text(f"from . import log\n{banner * 3}MARK = 1\n")
    (package / "log.py").write_text(_LOG_PY)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("ispec", "ispec.log"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)

    with caplog.at_level("INFO", logger="gtellcorr.preprocessing._ispec"):
        ispec = import_ispec(tmp_path / "ispec_dir")

    assert ispec.MARK == 1
    assert root.level == level and root.handlers == handlers
    assert not (work / "ispec.log").exists()
    assert capsys.readouterr().out == ""
    assert caplog.text.count("compiled extensions are not built") == 1
