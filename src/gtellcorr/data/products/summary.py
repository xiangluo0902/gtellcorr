"""``summary.csv``: one row per (exposure, order) of a run.

Fitted orders carry the fit parameters and the reduced chi-square; skipped orders
carry only their status. ``wavesol_coeffs`` is the JSON list of wavelength-solution
correction coefficients [km/s], empty when no correction was fitted. ``converged`` (``true`` /
``false``) and ``n_eval`` are the optimiser's outcome (in a joint fit, the joint
fit's); both are empty for orders that were not fitted. ``stellar_shift`` is the
fitted template velocity beyond the placement RV [km/s], empty when not fitted.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

__all__ = ["SUMMARY_COLUMNS", "joint_row", "shift_cell", "single_row", "skipped_row"]

SUMMARY_COLUMNS = (
    "exposure", "order", "wave_lo", "wave_hi", "time_utc", "molecules", "relcol", "resolution",
    "chi2_reduced", "dof_mask_removed", "chi2_reduced_order", "dof_mask_removed_order",
    "chi2_reduced_joint", "dof_mask_removed_joint", "masked_frac", "lsf_shape", "status",
    "elapsed_s", "wavesol_coeffs", "converged", "n_eval", "stellar_shift",
)  # fmt: skip


def _masked(result: Any) -> str:
    """Fraction of excluded pixels."""
    frac = float(np.asarray(result.exclusion, bool).mean()) if result.exclusion is not None else 0.0
    return f"{frac:.3f}"


def _shape(result: Any) -> str:
    """LSF shape parameter, empty for a Gaussian LSF."""
    return "" if result.lsf_shape is None else f"{result.lsf_shape:.4f}"


def _outcome(result: Any) -> list[str]:
    """``converged`` and ``n_eval`` cells (empty when unknown)."""
    converged = getattr(result, "converged", None)
    n_eval = getattr(result, "n_eval", None)
    return [
        "" if converged is None else ("true" if converged else "false"),
        "" if n_eval is None else str(int(n_eval)),
    ]


def _stellar_shift(result: Any) -> str:
    """``stellar_shift`` cell [km/s], ``""`` when the template was not moved."""
    value = getattr(result, "stellar_shift", None)
    return "" if value is None else f"{float(value):.5f}"


def shift_cell(values: Any) -> str:
    """``wavesol_coeffs`` cell: JSON list of coefficients [km/s], ``""`` without a correction."""
    if values is None:
        return ""
    return json.dumps([round(float(v), 5) for v in values])


def single_row(
    exposure: str, result: Any, time_iso: str, elapsed: float, *, fitted: bool
) -> list[Any]:
    """Row of a single-order fit (``fitted=False``: skipped, no parameters)."""
    x = np.asarray(result.wave_fit, float)
    head = [exposure, int(result.order), f"{x.min():.1f}", f"{x.max():.1f}", time_iso,
            json.dumps(list(result.relcol.keys()))]  # fmt: skip
    if not fitted:
        return [*head, *[""] * 10, result.status, f"{elapsed:.1f}", "", "", "", ""]
    return [
        *head, json.dumps(result.relcol), f"{result.resolution:.0f}",
        result.chi2_reduced, result.dof_mask_removed, result.chi2_reduced, result.dof_mask_removed,
        "", "", _masked(result), _shape(result), result.status, f"{elapsed:.1f}",
        shift_cell(result.wave_shift), *_outcome(result), _stellar_shift(result),
    ]  # fmt: skip


def skipped_row(
    exposure: str, order: int, time_iso: str, *, error: BaseException | None
) -> list[Any]:
    """Row of an order that could not be read (status ``ERROR:<exception type>``)."""
    kind = type(error).__name__ if error is not None else "ValueError"
    return [exposure, int(order), "", "", time_iso, *[""] * 11, f"ERROR:{kind}", "", "", "", "", ""]


def joint_row(exposure: str, result: Any, time_iso: str, elapsed: float) -> list[Any]:
    """Row of an order used in a joint fit."""
    x = np.asarray(result.wave_fit, float)
    return [
        exposure, int(result.order), f"{x.min():.1f}", f"{x.max():.1f}", time_iso,
        json.dumps(list(result.relcol.keys())), json.dumps(result.relcol),
        f"{result.resolution:.0f}",
        result.chi2_reduced_order, result.dof_mask_removed_order,
        result.chi2_reduced_order, result.dof_mask_removed_order,
        result.chi2_reduced_joint, result.dof_mask_removed_joint,
        _masked(result), _shape(result), result.status, f"{elapsed:.1f}",
        shift_cell(result.wave_shift), *_outcome(result), _stellar_shift(result),
    ]  # fmt: skip
