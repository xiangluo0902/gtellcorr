"""Logging helpers (internal).

Library code logs to the ``gtellcorr`` logger hierarchy instead of printing. The
root ``gtellcorr`` logger carries only a ``NullHandler``, so the application (a
user script, a notebook or ``gtellcorr.cli``) decides where messages go, e.g.::

    import logging
    logging.basicConfig(level=logging.INFO)

Progress of a pipeline run is reported through the ``progress`` callback of
:class:`~gtellcorr.pipeline.Pipeline`.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger"]

ROOT_LOGGER_NAME = "gtellcorr"
logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the logger for module *name* inside the ``gtellcorr`` hierarchy."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
