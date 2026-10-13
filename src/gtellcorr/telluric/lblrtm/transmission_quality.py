"""Non-destructive quality classification for direct LBLRTM transmission.

This module never changes transmission values.  It only classifies each raw
finite TAPE12 sample and constructs the mask consumed by a fitting adapter.
The strict TAPE12 reader remains a separate check that rejects bad files.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Final

import numpy as np

from .lblrtm_errors import LblrtmInputError

__all__ = [
    "TransmissionQualityFlag",
    "TransmissionQualitySummary",
    "build_transmission_fit_mask",
    "classify_transmission_values",
    "summarize_transmission_quality",
]


class TransmissionQualityFlag(IntFlag):
    """One mutually exclusive raw-transmission quality classification."""

    PHYSICAL = 0
    SATURATED_ZERO = 1
    BELOW_ZERO = 2
    ABOVE_ONE = 4
    NONFINITE = 8


_FLAG_DTYPE: Final = np.dtype(np.uint8)


@dataclass(frozen=True, slots=True)
class TransmissionQualitySummary:
    """Read-only, JSON-friendly summary of one classified spectrum."""

    total_count: int
    physical_count: int
    saturated_zero_count: int
    below_zero_count: int
    above_one_count: int
    nonfinite_count: int
    minimum: float
    maximum: float
    first_bad_index: int | None
    last_bad_index: int | None


def _as_1d_float64(values: object, *, field_name: str) -> np.ndarray:
    """*values* as a one-dimensional float64 array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(f"{field_name} must be numeric") from exc
    if array.ndim != 1:
        raise LblrtmInputError(f"{field_name} must be one-dimensional")
    return array


def classify_transmission_values(
    transmission: object,
) -> np.ndarray:
    """Classify raw transmission without changing or copying its values."""

    values = _as_1d_float64(transmission, field_name="transmission")
    flags = np.zeros(values.shape, dtype=_FLAG_DTYPE)
    nonfinite = ~np.isfinite(values)
    flags[nonfinite] = int(TransmissionQualityFlag.NONFINITE)
    finite = ~nonfinite
    flags[finite & (values == 0.0)] = int(TransmissionQualityFlag.SATURATED_ZERO)
    flags[finite & (values < 0.0)] = int(TransmissionQualityFlag.BELOW_ZERO)
    flags[finite & (values > 1.0)] = int(TransmissionQualityFlag.ABOVE_ONE)
    flags.setflags(write=False)
    return flags


def build_transmission_fit_mask(
    transmission: object,
    quality_flags: object | None = None,
) -> np.ndarray:
    """Return pixels safe for fitting, excluding zero and nonphysical values."""

    values = _as_1d_float64(transmission, field_name="transmission")
    flags = (
        classify_transmission_values(values)
        if quality_flags is None
        else np.asarray(quality_flags, dtype=_FLAG_DTYPE)
    )
    if flags.ndim != 1 or flags.shape != values.shape:
        raise LblrtmInputError("quality_flags must be one-dimensional and match transmission")
    usable = (
        np.isfinite(values)
        & (values > 0.0)
        & (values <= 1.0)
        & (flags == int(TransmissionQualityFlag.PHYSICAL))
    )
    usable.setflags(write=False)
    return usable


def summarize_transmission_quality(
    transmission: object,
    quality_flags: object | None = None,
) -> TransmissionQualitySummary:
    """Summarize classified raw values for the cache record and fitting reports."""

    values = _as_1d_float64(transmission, field_name="transmission")
    flags = (
        classify_transmission_values(values)
        if quality_flags is None
        else np.asarray(quality_flags, dtype=_FLAG_DTYPE)
    )
    if flags.ndim != 1 or flags.shape != values.shape:
        raise LblrtmInputError("quality_flags must be one-dimensional and match transmission")
    bad = flags != int(TransmissionQualityFlag.PHYSICAL)
    bad_indices = np.flatnonzero(bad)
    finite = values[np.isfinite(values)]
    minimum = float(np.min(finite)) if finite.size else float("nan")
    maximum = float(np.max(finite)) if finite.size else float("nan")
    return TransmissionQualitySummary(
        total_count=int(values.size),
        physical_count=int(np.count_nonzero(flags == 0)),
        saturated_zero_count=int(
            np.count_nonzero(flags == int(TransmissionQualityFlag.SATURATED_ZERO))
        ),
        below_zero_count=int(np.count_nonzero(flags == int(TransmissionQualityFlag.BELOW_ZERO))),
        above_one_count=int(np.count_nonzero(flags == int(TransmissionQualityFlag.ABOVE_ONE))),
        nonfinite_count=int(np.count_nonzero(flags == int(TransmissionQualityFlag.NONFINITE))),
        minimum=minimum,
        maximum=maximum,
        first_bad_index=int(bad_indices[0]) if bad_indices.size else None,
        last_bad_index=int(bad_indices[-1]) if bad_indices.size else None,
    )
