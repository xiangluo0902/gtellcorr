from __future__ import annotations

import logging

from gtellcorr._logging import get_logger


def test_get_logger_places_names_under_gtellcorr() -> None:
    assert get_logger("fitting").name == "gtellcorr.fitting"
    assert get_logger("gtellcorr.telluric").name == "gtellcorr.telluric"
    assert get_logger("gtellcorr").name == "gtellcorr"


def test_root_logger_has_null_handler() -> None:
    handlers = logging.getLogger("gtellcorr").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
