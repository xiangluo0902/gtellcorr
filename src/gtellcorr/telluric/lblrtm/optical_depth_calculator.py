"""Calculate direct LBLRTM IEMIT=0 sight-line optical depth.

This module is the optical-depth counterpart of
``transmission_calculator``.  It deliberately reuses that module's frozen
wavelength interpretation, wide-window segment plan, fixed ownership, and
wavelength-medium conversion.  The only radiative-transfer difference is
that every LBLRTM run uses IEMIT=0 and TAPE12 is read as optical depth.

Finite TAPE12 values, including finite negative values, are returned without
clipping or repair.  NaN and infinity reject the complete calculation.
No instrumental convolution is applied.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from .lblrtm_data_models import (
    LblrtmLineOfSight,
    LblrtmOpticalDepthSpectrum,
    LblrtmRunConfiguration,
    LblrtmSpectralWindow,
)
from .lblrtm_errors import LblrtmInputError, LblrtmOutputError
from .lblrtm_resources import ResolvedLblrtmResourceFiles
from .lblrtm_runner import execute_lblrtm
from .tape5_writer import render_lblrtm_tape5
from .tape12_reader import read_lblrtm_tape12_optical_depth
from .transmission_calculator import (
    _MAXIMUM_OUTPUT_GRID_POINTS,
    _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS,
    _build_lblrtm_multisegment_plan,
    _canonicalize_molecule_relative_columns,
    _compute_lblrtm_spectrum_semantic_sha256,
    _convert_vacuum_wavelength_to_requested_medium,
    _convert_wavenumber_spectrum_to_increasing_vacuum_wavelength,
    _f10p3_ticks_to_wavenumber_cm1,
    _LblrtmPlannedSpectralSegment,
    _LblrtmSpectralSegmentResult,
    _LblrtmValidatedSpectralSegmentResult,
    _normalize_transmission_request,
    _path_free_tape12_abi_metadata,
    _resample_transmission_to_requested_grid,
    _select_owned_lblrtm_samples,
    _snapshot_lblrtm_full_profile_arrays,
    _validate_complete_planned_lblrtm_segment_result,
    _validate_lblrtm_segment_seam_structure,
    _validate_lblrtm_wavelength_covers_expanded_input,
    _validate_spectral_window_inside_resource_support,
    _validate_tape12_abi_against_resources,
    _wavenumber_cm1_to_f10p3_ticks,
)

__all__ = ["calculate_lblrtm_optical_depth"]


_OPTICAL_DEPTH_METADATA_SCHEMA_VERSION: Final[int] = 1
_OPTICAL_DEPTH_POLICY_ID: Final[str] = "binary_tape12_iemit0_raw_finite_no_clipping_v1"
_OWNERSHIP_POLICY_ID: Final[str] = "fixed_core_half_open_right_owner_v1"


def calculate_lblrtm_optical_depth(
    full_profile: Mapping[str, object],
    wavelength_range_nm: ArrayLike,
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
    line_of_sight: LblrtmLineOfSight,
    molecule_relative_columns: Mapping[str, float] | None = None,
    resources: ResolvedLblrtmResourceFiles,
    wave_margin_nm: float = 0.0,
    output_wavelength_grid_nm: ArrayLike | None = None,
    continuum_mode: str = "full",
    optical_depth_grid_spacing_cm1: float = 0.002,
    timeout_s: float | None = None,
    verbose: bool = False,
) -> LblrtmOpticalDepthSpectrum:
    """Return direct, unconvolved IEMIT=0 sight-line optical depth.

    ``optical_depth_grid_spacing_cm1`` is written as IOD=1 ``DVOUT``.  Wide
    requests use the same overlapping segment plan, each sample taken from one
    fixed segment, as the transmission calculator.  Segment
    values are never blended.
    """

    if isinstance(optical_depth_grid_spacing_cm1, bool) or not np.isscalar(
        optical_depth_grid_spacing_cm1
    ):
        raise LblrtmInputError("optical_depth_grid_spacing_cm1 must be a positive finite scalar")
    try:
        spacing = float(optical_depth_grid_spacing_cm1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(
            "optical_depth_grid_spacing_cm1 must be a positive finite scalar"
        ) from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise LblrtmInputError("optical_depth_grid_spacing_cm1 must be a positive finite scalar")

    request = _normalize_transmission_request(
        wavelength_range_nm,
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=wavelength_conversion_formula,
        wave_margin_nm=wave_margin_nm,
        output_wavelength_grid_nm=output_wavelength_grid_nm,
    )
    relative_columns = _canonicalize_molecule_relative_columns(molecule_relative_columns)
    safe_range = _validate_spectral_window_inside_resource_support(
        request.spectral_window,
        resources=resources,
    )
    profile = _snapshot_lblrtm_full_profile_arrays(full_profile)

    lower_ticks = _wavenumber_cm1_to_f10p3_ticks(request.spectral_window.minimum_wavenumber_cm1)
    upper_ticks = _wavenumber_cm1_to_f10p3_ticks(request.spectral_window.maximum_wavenumber_cm1)
    if upper_ticks - lower_ticks > _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS:
        planned = _build_lblrtm_multisegment_plan(request.spectral_window)
        planned_segments = planned.segments
        segment_plan_sha256 = planned.segment_plan_sha256
    else:
        planned_segments = (
            _LblrtmPlannedSpectralSegment(
                segment_index=0,
                rendered_window_ticks=(lower_ticks, upper_ticks),
                core_window_ticks=(lower_ticks, upper_ticks),
                core_upper_inclusive=True,
            ),
        )
        segment_plan_sha256 = "single_segment"

    owned_wavenumbers: list[np.ndarray] = []
    owned_optical_depths: list[np.ndarray] = []
    segment_metadata: list[dict[str, object]] = []
    validated_segments: list[_LblrtmValidatedSpectralSegmentResult] = []
    expected_resource_fingerprint = resources.manifest_fingerprint
    if not isinstance(expected_resource_fingerprint, str):
        raise LblrtmInputError("internal error: the LBLRTM bundle was not checked before use")

    for planned_segment in planned_segments:
        rendered_lower, rendered_upper = planned_segment.rendered_window_ticks
        spectral_window = LblrtmSpectralWindow(
            minimum_wavenumber_cm1=(_f10p3_ticks_to_wavenumber_cm1(rendered_lower)),
            maximum_wavenumber_cm1=(_f10p3_ticks_to_wavenumber_cm1(rendered_upper)),
        )
        run_configuration = LblrtmRunConfiguration(
            spectral_window=spectral_window,
            line_of_sight=line_of_sight,
            continuum_mode=continuum_mode,
            spectral_output="optical_depth",
            optical_depth_grid_spacing_cm1=spacing,
        )
        tape5_text = render_lblrtm_tape5(
            profile,
            run_configuration,
            molecule_relative_columns=relative_columns,
        )
        local_tape5_sha256 = hashlib.sha256(tape5_text.encode("ascii")).hexdigest()
        run_result = execute_lblrtm(
            tape5_text,
            resources,
            timeout_s=timeout_s,
            verbose=verbose,
        )
        if run_result.tape5_sha256 != local_tape5_sha256:
            raise LblrtmOutputError("runner TAPE5 SHA-256 differs from the rendered input")
        if run_result.resource_fingerprint != expected_resource_fingerprint:
            raise LblrtmOutputError("internal error: the LBLRTM bundle changed during the run")

        wavenumber, optical_depth = read_lblrtm_tape12_optical_depth(run_result.tape12_bytes)
        if (
            wavenumber.ndim != 1
            or optical_depth.ndim != 1
            or wavenumber.shape != optical_depth.shape
            or wavenumber.size < 2
            or not np.all(np.isfinite(wavenumber))
            or np.any(wavenumber <= 0.0)
            or np.any(np.diff(wavenumber) <= 0.0)
        ):
            raise LblrtmOutputError(
                "IEMIT=0 TAPE12 coordinates are not finite, positive, and strictly increasing"
            )
        if not np.all(np.isfinite(optical_depth)):
            raise LblrtmOutputError("IEMIT=0 TAPE12 optical depth contains NaN or infinity")
        tape12_abi = _validate_tape12_abi_against_resources(
            run_result.tape12_bytes,
            resources,
        )
        raw_segment = _LblrtmSpectralSegmentResult(
            spectral_window=spectral_window,
            wavenumber_cm1=wavenumber,
            transmission=optical_depth,
            run_result=run_result,
            tape12_abi=tape12_abi,
        )
        wavenumber, optical_depth = _validate_complete_planned_lblrtm_segment_result(
            raw_segment,
            planned_segment=planned_segment,
            expected_resource_fingerprint=expected_resource_fingerprint,
            allow_nonphysical=True,
        )
        owned_wavenumber, owned_tau = _select_owned_lblrtm_samples(
            wavenumber,
            optical_depth,
            planned_segment=planned_segment,
        )
        owned_wavenumbers.append(owned_wavenumber)
        owned_optical_depths.append(owned_tau)
        semantic_sha256 = _compute_lblrtm_spectrum_semantic_sha256(
            wavenumber,
            optical_depth,
            coordinate_kind_and_unit="vacuum_wavenumber_cm-1",
            wavelength_medium_or_empty="",
            allow_nonphysical=True,
        )
        validated_segments.append(
            _LblrtmValidatedSpectralSegmentResult(
                planned_segment=planned_segment,
                spectral_window=spectral_window,
                wavenumber_cm1=wavenumber,
                transmission=optical_depth,
                owned_wavenumber_cm1=owned_wavenumber,
                owned_transmission=owned_tau,
                tape12_spectrum_semantic_sha256=semantic_sha256,
                resource_fingerprint=run_result.resource_fingerprint,
                tape5_sha256=run_result.tape5_sha256,
                tape6_sha256=run_result.tape6_sha256,
                tape12_sha256=run_result.tape12_sha256,
                tape12_abi=tape12_abi,
            )
        )
        record: dict[str, object] = {
            "segment_index": planned_segment.segment_index,
            "rendered_wavenumber_range_cm1": (
                spectral_window.minimum_wavenumber_cm1,
                spectral_window.maximum_wavenumber_cm1,
            ),
            "returned_wavenumber_range_cm1": (
                float(wavenumber[0]),
                float(wavenumber[-1]),
            ),
            "returned_point_count": int(wavenumber.size),
            "owned_point_count": int(owned_wavenumber.size),
            "tape5_sha256": run_result.tape5_sha256,
            "tape6_sha256": run_result.tape6_sha256,
            "tape12_sha256": run_result.tape12_sha256,
        }
        if tape12_abi is not None:
            record["tape12_abi"] = _path_free_tape12_abi_metadata(tape12_abi)
        segment_metadata.append(record)

    seam_metadata = []
    for lower_segment, upper_segment in itertools.pairwise(validated_segments):
        seam = _validate_lblrtm_segment_seam_structure(
            lower_segment,
            upper_segment,
        )
        seam_metadata.append(
            {
                "lower_segment_index": seam.lower_segment_index,
                "upper_segment_index": seam.upper_segment_index,
                "source_switch_wavenumber_cm1": (seam.source_switch_wavenumber_cm1),
                "selected_seam_gap_cm1": seam.selected_seam_gap_cm1,
                "diagnostic_point_count": seam.diagnostic_point_count,
                "rms_optical_depth_difference": seam.rms,
                "p99_absolute_optical_depth_difference": seam.p99_absolute,
                "maximum_absolute_optical_depth_difference": (seam.maximum_absolute),
            }
        )

    total_point_count = sum(array.size for array in owned_wavenumbers)
    if total_point_count > _MAXIMUM_OUTPUT_GRID_POINTS:
        raise LblrtmOutputError(
            f"merged IEMIT=0 point count exceeds the project limit of {_MAXIMUM_OUTPUT_GRID_POINTS}"
        )
    try:
        merged_wavenumber = np.concatenate(owned_wavenumbers)
        merged_tau = np.concatenate(owned_optical_depths)
    except MemoryError as exc:
        raise LblrtmOutputError("cannot allocate merged IEMIT=0 optical-depth arrays") from exc
    if (
        merged_wavenumber.size < 2
        or np.any(np.diff(merged_wavenumber) <= 0.0)
        or not np.all(np.isfinite(merged_tau))
    ):
        raise LblrtmOutputError("fixed-ownership IEMIT=0 merge is invalid")

    vacuum_wavelength_nm, tau_in_wavelength_order = (
        _convert_wavenumber_spectrum_to_increasing_vacuum_wavelength(
            merged_wavenumber,
            merged_tau,
        )
    )
    converted_wavelength_nm = _convert_vacuum_wavelength_to_requested_medium(
        vacuum_wavelength_nm,
        wavelength_medium=request.wavelength_medium,
        wavelength_conversion_formula=request.wavelength_conversion_formula,
    )
    _validate_lblrtm_wavelength_covers_expanded_input(
        converted_wavelength_nm,
        expanded_input_wavelength_range_nm=(request.expanded_wavelength_range_nm),
    )
    if request.output_wavelength_grid_nm is None:
        output_wavelength_nm = converted_wavelength_nm
        output_tau = tau_in_wavelength_order
        output_grid_kind = "tape12_grid_converted"
        resampling = "none"
    else:
        output_wavelength_nm = request.output_wavelength_grid_nm
        output_tau = _resample_transmission_to_requested_grid(
            converted_wavelength_nm,
            tau_in_wavelength_order,
            output_wavelength_nm,
        )
        output_grid_kind = "caller_supplied"
        resampling = "linear_within_coverage"

    negative_mask = output_tau < 0.0
    metadata = {
        "metadata_schema_version": _OPTICAL_DEPTH_METADATA_SCHEMA_VERSION,
        "source_product": "binary_TAPE12",
        "spectral_output": "optical_depth",
        "iemit": 0,
        "iod": 1,
        "optical_depth_grid_spacing_cm1": spacing,
        "optical_depth_policy": _OPTICAL_DEPTH_POLICY_ID,
        "raw_values_modified": False,
        "unconvolved": True,
        "continuum_mode": continuum_mode,
        "requested_wavelength_range_nm": request.requested_wavelength_range_nm,
        "expanded_input_wavelength_range_nm": request.expanded_wavelength_range_nm,
        "expanded_vacuum_wavelength_range_nm": (request.expanded_vacuum_wavelength_range_nm),
        "input_wavelength_medium": request.wavelength_medium,
        "output_wavelength_medium": request.wavelength_medium,
        "wavelength_conversion_formula": request.wavelength_conversion_formula,
        "wave_margin_nm": request.wave_margin_nm,
        "output_grid_kind": output_grid_kind,
        "resampling": resampling,
        "output_point_count": int(output_tau.size),
        "output_wavelength_range_nm": (
            float(output_wavelength_nm[0]),
            float(output_wavelength_nm[-1]),
        ),
        "line_of_sight": {
            "airmass": line_of_sight.airmass,
            "observer_altitude_km": line_of_sight.observer_altitude_km,
            "path_end_altitude_km": line_of_sight.path_end_altitude_km,
            "reference_latitude_deg": line_of_sight.reference_latitude_deg,
        },
        "molecule_relative_columns": dict(relative_columns),
        "resource_fingerprint": expected_resource_fingerprint,
        "resource_safe_wavenumber_range_cm1": safe_range,
        "segment_count": len(planned_segments),
        "segment_plan_sha256": segment_plan_sha256,
        "ownership_policy": _OWNERSHIP_POLICY_ID,
        "segments": segment_metadata,
        "seams": seam_metadata,
        "negative_optical_depth_count": int(np.count_nonzero(negative_mask)),
        "minimum_optical_depth": float(np.min(output_tau)),
        "maximum_optical_depth": float(np.max(output_tau)),
    }
    try:
        return LblrtmOpticalDepthSpectrum(
            wavelength_nm=output_wavelength_nm,
            optical_depth=output_tau,
            wavelength_medium=request.wavelength_medium,
            metadata=metadata,
        )
    except LblrtmInputError as exc:
        raise LblrtmOutputError(
            "internal error: cannot assemble the IEMIT=0 optical-depth result"
        ) from exc
