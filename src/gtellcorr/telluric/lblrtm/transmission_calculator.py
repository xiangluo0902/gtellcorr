"""Calculate direct LBLRTM transmission for one explicit wavelength request.

This module runs the telluric radiative transfer.
A request turns one explicit wavelength range into a
TAPE5, delegates isolated execution to
:mod:`gtellcorr.telluric.lblrtm.lblrtm_runner`, parses
the returned TAPE12 with the independent public reader, and returns an
increasing wavelength spectrum.  Wider requests are planned and executed as
overlapping LBLRTM segments, then merged, each wavenumber taken from one fixed segment.

It does not read the gtellcorr configuration, apply an instrument line-spread
function, fit an observational continuum, or alter non-physical transmission
values.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    Context,
    Decimal,
    InvalidOperation,
    localcontext,
)
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from gtellcorr.data.observation.frames import air_to_vac, vac_to_air

from .lblrtm_data_models import (
    LblrtmLineOfSight,
    LblrtmRunConfiguration,
    LblrtmRunResult,
    LblrtmSpectralWindow,
    LblrtmTransmissionSpectrum,
)
from .lblrtm_errors import (
    LblrtmInputError,
    LblrtmOutputError,
    LblrtmResourceIntegrityError,
)
from .lblrtm_molecules import (
    LBLRTM_MOLECULES_BY_ID,
    normalize_lblrtm_molecule_name,
)
from .lblrtm_quality_models import TelluricTransmissionQualityResult
from .lblrtm_resources import ResolvedLblrtmResourceFiles
from .lblrtm_runner import execute_lblrtm
from .tape5_writer import (
    render_lblrtm_tape5,
    validate_lblrtm_layer_input,
)
from .tape12_reader import (
    LblrtmTape12Abi,
    inspect_lblrtm_tape12_abi,
    read_lblrtm_tape12_transmission,
    read_lblrtm_tape12_transmission_quality,
)
from .transmission_quality import (
    TransmissionQualityFlag,
    build_transmission_fit_mask,
    classify_transmission_values,
)

__all__ = ["calculate_lblrtm_transmission"]


_SUPPORTED_WAVELENGTH_MEDIA: Final = frozenset({"air", "vacuum"})
_SUPPORTED_WAVELENGTH_CONVERSION_FORMULAS: Final = frozenset(
    {"espresso", "edlen1966", "ciddor1996"}
)
_TAPE5_WAVENUMBER_QUANTUM_CM1: Final = Decimal("0.001")
_MAXIMUM_RENDERED_SEGMENT_WIDTH_CM1: Final = Decimal("2000.000")
_ADJACENT_RENDERED_SEGMENT_OVERLAP_CM1: Final = Decimal("10.000")
_HALF_RENDERED_SEGMENT_OVERLAP_CM1: Final = Decimal("5.000")
_MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS: Final[int] = 2_000_000
_ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS: Final[int] = 10_000
_HALF_RENDERED_SEGMENT_OVERLAP_TICKS: Final[int] = 5_000
_MAXIMUM_SEGMENT_COUNT: Final[int] = 32
_TRANSMISSION_METADATA_SCHEMA_VERSION: Final[int] = 3
_SEGMENT_PLAN_SCHEMA_VERSION: Final[int] = 1
_SEGMENT_PLAN_POLICY_ID: Final = "vacuum_wavenumber_f10p3_balanced_overlap_v1"
_SEGMENT_PLAN_HASH_DOMAIN: Final = b"gtellcorr.lblrtm.segment-plan.v1\0"
_SPECTRUM_SEMANTIC_HASH_DOMAIN: Final = b"gtellcorr.lblrtm.spectrum.semantic.v1\0"
_SPECTRUM_SEMANTIC_HASH_SCHEMA_VERSION: Final[int] = 1
_MAXIMUM_TRANSMISSION: Final[float] = 1.00001
_WAVENUMBER_TO_WAVELENGTH_NM_CM1: Final[float] = 1.0e7
_MAXIMUM_OUTPUT_GRID_POINTS: Final[int] = 5_000_000
_MAXIMUM_AIR_BOUND_EXPANSION_STEPS: Final[int] = 64
_CIDDOR_AIR_WAVELENGTH_RANGE_NM: Final = (230.0, 1_700.0)
_BINARY_SOURCE_PRODUCT: Final = "binary_TAPE12"
_OWNERSHIP_POLICY_ID: Final = "fixed_core_half_open_right_owner_v1"
_TRANSMISSION_POLICY_ID: Final = "binary_tape12_strict_raw_no_clipping_v1"
_TAPE6_COMPLETION_POLICY_ID: Final = "v12.17_footer_v1"
_TAPE12_READER_POLICY_ID: Final = "aer_lblrtm_v12.17_direct_rt_v1"
_PROFILE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "z_km",
    "P_hPa",
    "T_K",
    *tuple(molecule.profile_field for _, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())),
)
_PROFILE_MOLECULE_NAMES: Final[tuple[str, ...]] = tuple(
    molecule.canonical_name for _, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())
)
_TAPE12_ABI_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "build_contract.tape12_abi.byte_order",
    "build_contract.tape12_abi.record_marker_bytes",
    "build_contract.tape12_abi.real_bytes",
    "build_contract.tape12_abi.integer_bytes",
)


def _declared_tape12_abi(
    resources: ResolvedLblrtmResourceFiles,
) -> tuple[LblrtmTape12Abi | None, bool]:
    """Return the full TAPE12 binary layout of the bundle and how its metadata is extended."""

    evidence = getattr(resources, "validation_evidence", None)
    if not isinstance(evidence, Mapping):
        return None, False
    present_keys = tuple(key for key in _TAPE12_ABI_EVIDENCE_KEYS if key in evidence)
    if not present_keys:
        return None, False
    if len(present_keys) != len(_TAPE12_ABI_EVIDENCE_KEYS):
        missing = tuple(key for key in _TAPE12_ABI_EVIDENCE_KEYS if key not in evidence)
        raise LblrtmResourceIntegrityError(
            "internal error: the bundle check did not record the full TAPE12 binary "
            f"layout; missing: {missing}"
        )
    try:
        tape12_abi = LblrtmTape12Abi(
            byte_order=evidence[_TAPE12_ABI_EVIDENCE_KEYS[0]],
            record_marker_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[1]]),
            real_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[2]]),
            integer_bytes=int(evidence[_TAPE12_ABI_EVIDENCE_KEYS[3]]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmResourceIntegrityError(
            "internal error: the bundle check recorded an invalid TAPE12 binary layout"
        ) from exc
    return tape12_abi, evidence.get("manifest_schema_version") == "2"


def _validate_tape12_abi_against_resources(
    source: bytes,
    resources: ResolvedLblrtmResourceFiles,
) -> LblrtmTape12Abi | None:
    """Match one detected TAPE12 binary layout to the one the bundle declares."""

    declared, include_in_metadata = _declared_tape12_abi(resources)
    if declared is None:
        return None
    detected = inspect_lblrtm_tape12_abi(source)
    if detected != declared:
        raise LblrtmResourceIntegrityError(
            "the TAPE12 written by LBLRTM has a different binary layout than the bundle "
            f"declares: expected {declared}, found {detected}"
        )
    return detected if include_in_metadata else None


def _path_free_tape12_abi_metadata(
    tape12_abi: LblrtmTape12Abi,
) -> dict[str, object]:
    """Serialize one detected ABI without executable or resource paths."""

    return {
        "byte_order": tape12_abi.byte_order,
        "record_marker_bytes": tape12_abi.record_marker_bytes,
        "real_bytes": tape12_abi.real_bytes,
        "integer_bytes": tape12_abi.integer_bytes,
    }


def audit_lblrtm_tape12_transmission(
    source: bytes | bytearray | memoryview | str,
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
) -> TelluricTransmissionQualityResult:
    """Compatibility wrapper for non-destructive TAPE12 quality evaluation.

    New callers should use ``calculate_lblrtm_transmission(...,
    quality_mode="evaluate")``.  This wrapper remains for diagnostic callers
    and does not clip, replace, or interpolate across a flagged sample.
    """

    if wavelength_medium not in _SUPPORTED_WAVELENGTH_MEDIA:
        raise LblrtmInputError(
            f"wavelength_medium must be one of: {sorted(_SUPPORTED_WAVELENGTH_MEDIA)}"
        )
    if wavelength_conversion_formula not in _SUPPORTED_WAVELENGTH_CONVERSION_FORMULAS:
        raise LblrtmInputError(
            f"unsupported wavelength conversion formula: {wavelength_conversion_formula!r}"
        )
    wavenumber, transmission, flags, fit_usable = read_lblrtm_tape12_transmission_quality(source)
    vacuum_wavelength = _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / wavenumber
    order = np.argsort(vacuum_wavelength)
    vacuum_wavelength = vacuum_wavelength[order]
    transmission = transmission[order]
    flags = flags[order]
    fit_usable = fit_usable[order]
    wavelength = _convert_vacuum_wavelength_to_requested_medium(
        vacuum_wavelength,
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=wavelength_conversion_formula,
    )
    return TelluricTransmissionQualityResult(
        wavelength_nm=wavelength,
        transmission=transmission,
        wavelength_medium=wavelength_medium,
        quality_flags=flags,
        fit_usable=fit_usable,
        metadata={
            "source_product": "binary_TAPE12",
            "quality_policy_id": "transmission_mask_nonphysical_v1",
            "raw_values_modified": False,
            "wavelength_conversion_formula": wavelength_conversion_formula,
        },
    )


@dataclass(frozen=True, slots=True)
class _NormalizedTransmissionRequest:
    """Numerical form of one wavelength request (read-only)."""

    requested_wavelength_range_nm: tuple[float, float]
    expanded_wavelength_range_nm: tuple[float, float]
    expanded_vacuum_wavelength_range_nm: tuple[float, float]
    requested_wavenumber_range_cm1: tuple[float, float]
    spectral_window: LblrtmSpectralWindow
    wavelength_medium: str
    wavelength_conversion_formula: str
    wave_margin_nm: float
    output_wavelength_grid_nm: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _LblrtmPlannedSpectralSegment:
    """One TAPE5 window and the core range it alone supplies."""

    segment_index: int
    rendered_window_ticks: tuple[int, int]
    core_window_ticks: tuple[int, int]
    core_upper_inclusive: bool


@dataclass(frozen=True, slots=True)
class _LblrtmMultisegmentPlan:
    """Segment plan for one complete wavenumber window, rounded to TAPE5 precision."""

    complete_window_ticks: tuple[int, int]
    segments: tuple[_LblrtmPlannedSpectralSegment, ...]
    segment_plan_sha256: str


@dataclass(frozen=True, slots=True)
class _LblrtmSpectralSegmentResult:
    """One parsed TAPE12 segment together with the checked result of its run."""

    spectral_window: LblrtmSpectralWindow
    wavenumber_cm1: np.ndarray
    transmission: np.ndarray
    run_result: LblrtmRunResult
    quality_flags: np.ndarray | None = None
    fit_usable: np.ndarray | None = None
    tape12_abi: LblrtmTape12Abi | None = None


@dataclass(frozen=True, slots=True)
class _LblrtmMultisegmentPhysicalInputSnapshot:
    """The physical inputs and LBLRTM bundle shared by every segment (read-only)."""

    normalized_request: _NormalizedTransmissionRequest
    full_profile: Mapping[str, np.ndarray]
    line_of_sight: LblrtmLineOfSight
    molecule_relative_columns: Mapping[str, float]
    continuum_mode: str
    resources: ResolvedLblrtmResourceFiles
    resource_fingerprint: str
    resource_safe_wavenumber_range_cm1: tuple[float, float]
    timeout_s: float | None
    verbose: bool


@dataclass(frozen=True, slots=True)
class _LblrtmValidatedSpectralSegmentResult:
    """Checked TAPE12 arrays and short checksums of the run files for one segment.

    Raw TAPE6 text and TAPE12 bytes are intentionally not retained after the
    strict reader and complete-segment checks finish.  Keeping only their
    digests prevents a wide request from holding every isolated run artifact
    in memory at once.
    """

    planned_segment: _LblrtmPlannedSpectralSegment
    spectral_window: LblrtmSpectralWindow
    wavenumber_cm1: np.ndarray
    transmission: np.ndarray
    owned_wavenumber_cm1: np.ndarray
    owned_transmission: np.ndarray
    tape12_spectrum_semantic_sha256: str
    resource_fingerprint: str
    tape5_sha256: str
    tape6_sha256: str
    tape12_sha256: str
    tape12_abi: LblrtmTape12Abi | None = None


@dataclass(frozen=True, slots=True)
class _LblrtmSeamDiagnostic:
    """Structural gate and report-only comparison for two adjacent runs."""

    lower_segment_index: int
    upper_segment_index: int
    source_switch_wavenumber_cm1: float
    planned_overlap_wavenumber_range_cm1: tuple[float, float]
    left_bracketing_spacing_cm1: float
    right_bracketing_spacing_cm1: float
    selected_seam_gap_cm1: float
    diagnostic_minimum_wavenumber_cm1: float
    diagnostic_maximum_wavenumber_cm1: float
    diagnostic_point_count: int
    rms: float
    p99_absolute: float
    maximum_absolute: float


@dataclass(frozen=True, slots=True)
class _LblrtmMergedSpectrum:
    """Fixed-ownership LBLRTM result in increasing vacuum wavenumber."""

    wavenumber_cm1: np.ndarray
    transmission: np.ndarray
    semantic_sha256: str


def calculate_lblrtm_transmission(
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
    timeout_s: float | None = None,
    verbose: bool = False,
    quality_mode: str = "strict",
) -> LblrtmTransmissionSpectrum | TelluricTransmissionQualityResult:
    """Return one direct, unconvolved LBLRTM transmission spectrum.

    ``quality_mode="strict"`` (the default) preserves the scientific
    acceptance path and rejects non-physical transmission.  With
    ``quality_mode="evaluate"``, the complete raw transmission is returned
    together with quality flags and ``fit_usable``; values are never clipped
    or repaired.  The evaluation mode supplies complete samples and quality
    flags to :class:`~gtellcorr.telluric.lblrtm.tau_cache.TauCache`; those flags
    are only recorded and are not converted into a fit mask.

    ``wave_margin_nm`` is applied symmetrically in ``wavelength_medium``
    before an air request is converted to vacuum wavelength and reciprocal
    centimetres.  When ``output_wavelength_grid_nm`` is omitted, the complete
    padded TAPE12 grid is returned so a later instrument-LSF layer retains
    line-wing support.  A supplied output grid is interpolated only when every
    point lies inside both the declared padded interval and the actual parsed
    TAPE12 coverage.

    A request no wider than 2000 cm-1 after outward fixed-width quantization
    runs as a single segment.  Wider requests are executed
    one after another on overlapping windows.  Their complete binary-TAPE12
    spectra are checked before each wavenumber is taken from one fixed segment
    (no blending).
    """

    if quality_mode not in {"strict", "evaluate"}:
        raise LblrtmInputError("quality_mode must be either 'strict' or 'evaluate'")
    normalized_request = _normalize_transmission_request(
        wavelength_range_nm,
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=wavelength_conversion_formula,
        wave_margin_nm=wave_margin_nm,
        output_wavelength_grid_nm=output_wavelength_grid_nm,
    )
    effective_relative_columns = _canonicalize_molecule_relative_columns(molecule_relative_columns)
    run_configuration = LblrtmRunConfiguration(
        spectral_window=normalized_request.spectral_window,
        line_of_sight=line_of_sight,
        continuum_mode=continuum_mode,
    )
    resource_safe_wavenumber_range_cm1 = _validate_spectral_window_inside_resource_support(
        normalized_request.spectral_window,
        resources=resources,
    )
    complete_lower_ticks = _wavenumber_cm1_to_f10p3_ticks(
        normalized_request.spectral_window.minimum_wavenumber_cm1
    )
    complete_upper_ticks = _wavenumber_cm1_to_f10p3_ticks(
        normalized_request.spectral_window.maximum_wavenumber_cm1
    )
    if complete_upper_ticks - complete_lower_ticks > _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS:
        segment_plan = _build_lblrtm_multisegment_plan(normalized_request.spectral_window)
        return _calculate_lblrtm_multisegment_transmission(
            full_profile,
            normalized_request=normalized_request,
            segment_plan=segment_plan,
            line_of_sight=line_of_sight,
            molecule_relative_columns=effective_relative_columns,
            resources=resources,
            resource_safe_wavenumber_range_cm1=(resource_safe_wavenumber_range_cm1),
            continuum_mode=continuum_mode,
            timeout_s=timeout_s,
            verbose=verbose,
            quality_mode=quality_mode,
        )
    segment = _calculate_one_lblrtm_spectral_segment(
        full_profile,
        run_configuration,
        molecule_relative_columns=effective_relative_columns,
        resources=resources,
        timeout_s=timeout_s,
        verbose=verbose,
        quality_mode=quality_mode,
    )
    try:
        _validate_lblrtm_segment_matches_requested_window(
            segment,
            requested_wavenumber_range_cm1=(normalized_request.requested_wavenumber_range_cm1),
        )

        lblrtm_vacuum_wavelength_nm, lblrtm_transmission = (
            _convert_wavenumber_spectrum_to_increasing_vacuum_wavelength(
                segment.wavenumber_cm1,
                segment.transmission,
            )
        )
        lblrtm_wavelength_nm = _convert_vacuum_wavelength_to_requested_medium(
            lblrtm_vacuum_wavelength_nm,
            wavelength_medium=normalized_request.wavelength_medium,
            wavelength_conversion_formula=(normalized_request.wavelength_conversion_formula),
        )
        _validate_lblrtm_wavelength_covers_expanded_input(
            lblrtm_wavelength_nm,
            expanded_input_wavelength_range_nm=(normalized_request.expanded_wavelength_range_nm),
        )
        if quality_mode == "strict":
            _validate_transmission_spectrum(
                lblrtm_wavelength_nm,
                lblrtm_transmission,
                context="converted LBLRTM/TAPE12 spectrum",
            )

        if normalized_request.output_wavelength_grid_nm is None:
            output_wavelength_nm = lblrtm_wavelength_nm
            output_transmission = lblrtm_transmission
            output_grid_kind = "tape12_grid_converted"
            resampling = "none"
        else:
            output_wavelength_nm = normalized_request.output_wavelength_grid_nm
            output_transmission = _resample_transmission_to_requested_grid(
                lblrtm_wavelength_nm,
                lblrtm_transmission,
                output_wavelength_nm,
            )
            output_grid_kind = "caller_supplied"
            resampling = "linear_within_coverage"

        quality_flags = None
        fit_usable = None
        if quality_mode == "evaluate":
            lblrtm_quality_flags = np.asarray(segment.quality_flags, dtype=np.uint8)[::-1]
            lblrtm_fit_usable = np.asarray(segment.fit_usable, dtype=bool)[::-1]
            if normalized_request.output_wavelength_grid_nm is None:
                quality_flags = lblrtm_quality_flags
                fit_usable = lblrtm_fit_usable
            else:
                quality_flags, fit_usable = _resample_quality_to_requested_grid(
                    lblrtm_wavelength_nm,
                    lblrtm_quality_flags,
                    lblrtm_fit_usable,
                    output_wavelength_nm,
                )
        if quality_mode == "strict":
            _validate_transmission_spectrum(
                output_wavelength_nm,
                output_transmission,
                context="final LBLRTM transmission spectrum",
            )
        metadata = _build_transmission_metadata(
            normalized_request,
            segment,
            line_of_sight=line_of_sight,
            molecule_relative_columns=effective_relative_columns,
            continuum_mode=continuum_mode,
            resource_safe_wavenumber_range_cm1=(resource_safe_wavenumber_range_cm1),
            lblrtm_wavelength_nm=lblrtm_wavelength_nm,
            output_wavelength_nm=output_wavelength_nm,
            output_grid_kind=output_grid_kind,
            resampling=resampling,
        )
        try:
            if quality_mode == "evaluate":
                metadata = dict(metadata)
                metadata.update(
                    {
                        "quality_mode": "evaluate",
                        "raw_values_modified": False,
                        "fit_mask_policy_id": "transmission_mask_nonphysical_v1",
                    }
                )
                return TelluricTransmissionQualityResult(
                    wavelength_nm=output_wavelength_nm,
                    transmission=output_transmission,
                    wavelength_medium=normalized_request.wavelength_medium,
                    quality_flags=quality_flags,
                    fit_usable=fit_usable,
                    metadata=metadata,
                )
            return LblrtmTransmissionSpectrum(
                wavelength_nm=output_wavelength_nm,
                transmission=output_transmission,
                wavelength_medium=normalized_request.wavelength_medium,
                metadata=metadata,
            )
        except (LblrtmInputError, MemoryError) as exc:
            raise LblrtmOutputError(
                "internal error: cannot assemble the LBLRTM transmission result"
            ) from exc
    except LblrtmOutputError as exc:
        _add_run_provenance_note(exc, segment.run_result)
        raise
    except MemoryError as exc:
        output_error = LblrtmOutputError(
            "cannot allocate memory while processing the LBLRTM output"
        )
        _add_run_provenance_note(output_error, segment.run_result)
        raise output_error from exc


def _normalize_transmission_request(
    wavelength_range_nm: ArrayLike,
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
    wave_margin_nm: float,
    output_wavelength_grid_nm: ArrayLike | None,
) -> _NormalizedTransmissionRequest:
    """Validate, copy, and physically interpret one wavelength request."""

    if (
        not isinstance(wavelength_medium, str)
        or wavelength_medium not in _SUPPORTED_WAVELENGTH_MEDIA
    ):
        choices = ", ".join(sorted(_SUPPORTED_WAVELENGTH_MEDIA))
        raise LblrtmInputError(f"wavelength_medium must be one of: {choices}")
    if not isinstance(wavelength_conversion_formula, str):
        raise LblrtmInputError("wavelength_conversion_formula must be an explicit string")
    normalized_formula = wavelength_conversion_formula.strip().lower()
    if not normalized_formula:
        raise LblrtmInputError("wavelength_conversion_formula must not be empty")
    if normalized_formula not in _SUPPORTED_WAVELENGTH_CONVERSION_FORMULAS:
        choices = ", ".join(sorted(_SUPPORTED_WAVELENGTH_CONVERSION_FORMULAS))
        raise LblrtmInputError(f"wavelength_conversion_formula must be one of: {choices}")

    requested_array = _normalize_wavelength_vector(
        wavelength_range_nm,
        field_name="wavelength_range_nm",
        required_size=2,
    )
    requested_range = (
        float(requested_array[0]),
        float(requested_array[1]),
    )
    margin = _coerce_nonnegative_finite_float(
        wave_margin_nm,
        field_name="wave_margin_nm",
    )
    expanded_range = (
        requested_range[0] - margin,
        requested_range[1] + margin,
    )
    if expanded_range[0] <= 0.0:
        raise LblrtmInputError("wave_margin_nm makes the lower calculation wavelength non-positive")
    if not expanded_range[1] > expanded_range[0]:
        raise LblrtmInputError("expanded wavelength range must remain strictly increasing")
    if (
        wavelength_medium == "air"
        and normalized_formula == "ciddor1996"
        and (
            expanded_range[0] < _CIDDOR_AIR_WAVELENGTH_RANGE_NM[0]
            or expanded_range[1] > _CIDDOR_AIR_WAVELENGTH_RANGE_NM[1]
        )
    ):
        raise LblrtmInputError(
            "ciddor1996 air-wavelength conversion is supported only over "
            "the documented 230--1700 nm domain, including wave_margin_nm"
        )

    (
        expanded_vacuum_range,
        requested_wavenumber_range,
        spectral_window,
    ) = _wavelength_nm_range_to_wavenumber_cm1_range(
        expanded_range,
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=normalized_formula,
    )

    normalized_output_grid: np.ndarray | None = None
    if output_wavelength_grid_nm is not None:
        normalized_output_grid = _normalize_wavelength_vector(
            output_wavelength_grid_nm,
            field_name="output_wavelength_grid_nm",
            minimum_size=2,
            maximum_size=_MAXIMUM_OUTPUT_GRID_POINTS,
        )
        if (
            normalized_output_grid[0] < expanded_range[0]
            or normalized_output_grid[-1] > expanded_range[1]
        ):
            raise LblrtmInputError(
                "output_wavelength_grid_nm must lie inside the declared padded wavelength interval"
            )

    return _NormalizedTransmissionRequest(
        requested_wavelength_range_nm=requested_range,
        expanded_wavelength_range_nm=expanded_range,
        expanded_vacuum_wavelength_range_nm=expanded_vacuum_range,
        requested_wavenumber_range_cm1=requested_wavenumber_range,
        spectral_window=spectral_window,
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=normalized_formula,
        wave_margin_nm=margin,
        output_wavelength_grid_nm=normalized_output_grid,
    )


def _canonicalize_molecule_relative_columns(
    molecule_relative_columns: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return effective relative columns for all 12 profile molecules.

    The returned copy is passed to the writer as well as stored in the result
    metadata, so changes by the caller cannot make the two disagree.
    """

    if molecule_relative_columns is None:
        supplied_items: tuple[tuple[object, object], ...] = ()
    else:
        if not isinstance(molecule_relative_columns, Mapping):
            raise LblrtmInputError("molecule_relative_columns must be a mapping or None")
        try:
            supplied_items = tuple(molecule_relative_columns.items())
        except RuntimeError as exc:
            raise LblrtmInputError("molecule_relative_columns changed while being read") from exc

    supplied_scales: dict[str, float] = {}
    for supplied_name, supplied_scale in supplied_items:
        try:
            canonical_name = normalize_lblrtm_molecule_name(supplied_name)
        except (TypeError, ValueError) as exc:
            raise LblrtmInputError(
                f"invalid molecule_relative_columns key {supplied_name!r}: {exc}"
            ) from exc
        if canonical_name in supplied_scales:
            raise LblrtmInputError(
                f"duplicate molecule_relative_columns key after normalization: {canonical_name}"
            )
        scale = _coerce_nonnegative_finite_float(
            supplied_scale,
            field_name=f"relative-column scale for {canonical_name}",
        )
        supplied_scales[canonical_name] = scale

    return {
        molecule.canonical_name: supplied_scales.get(
            molecule.canonical_name,
            1.0,
        )
        for molecule_id, molecule in sorted(LBLRTM_MOLECULES_BY_ID.items())
    }


def _wavelength_nm_range_to_wavenumber_cm1_range(
    expanded_wavelength_range_nm: tuple[float, float],
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    LblrtmSpectralWindow,
]:
    """Convert one expanded wavelength range to a TAPE5 spectral window.

    Returns the expanded vacuum wavelength range, the unquantized reciprocal
    bounds, and the complete outward-quantized calculation window.
    """

    wavelength = np.asarray(expanded_wavelength_range_nm, dtype=np.float64)
    if wavelength_medium == "air":
        try:
            vacuum_wavelength = np.asarray(
                air_to_vac(
                    wavelength,
                    formula=wavelength_conversion_formula,
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            raise LblrtmInputError(
                "cannot convert the expanded air wavelength range to vacuum "
                f"with formula {wavelength_conversion_formula!r}"
            ) from exc
    else:
        vacuum_wavelength = np.array(
            wavelength,
            dtype=np.float64,
            copy=True,
        )

    if (
        vacuum_wavelength.shape != (2,)
        or not np.all(np.isfinite(vacuum_wavelength))
        or np.any(vacuum_wavelength <= 0.0)
        or not vacuum_wavelength[1] > vacuum_wavelength[0]
    ):
        raise LblrtmInputError(
            "wavelength conversion must produce two finite, positive, "
            "strictly increasing vacuum wavelengths"
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        minimum_wavenumber = _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / vacuum_wavelength[1]
        maximum_wavenumber = _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / vacuum_wavelength[0]
    requested_wavenumber_range = (
        float(minimum_wavenumber),
        float(maximum_wavenumber),
    )
    if (
        not all(math.isfinite(value) for value in requested_wavenumber_range)
        or requested_wavenumber_range[0] <= 0.0
        or requested_wavenumber_range[1] <= requested_wavenumber_range[0]
    ):
        raise LblrtmInputError(
            "expanded vacuum wavelength range does not map to a finite, "
            "positive, increasing wavenumber interval"
        )

    spectral_window = _quantize_wavenumber_range_outward_for_tape5(requested_wavenumber_range)
    spectral_window = _expand_spectral_window_for_output_medium_coverage(
        spectral_window,
        expanded_input_wavelength_range_nm=(expanded_wavelength_range_nm),
        wavelength_medium=wavelength_medium,
        wavelength_conversion_formula=wavelength_conversion_formula,
    )
    return (
        (
            float(vacuum_wavelength[0]),
            float(vacuum_wavelength[1]),
        ),
        requested_wavenumber_range,
        spectral_window,
    )


def _quantize_wavenumber_range_outward_for_tape5(
    wavenumber_range_cm1: tuple[float, float],
) -> LblrtmSpectralWindow:
    """Outward-quantize complete reciprocal bounds to TAPE5 F10.3 precision."""

    raw_lower, raw_upper = wavenumber_range_cm1
    try:
        with localcontext(Context(prec=64)):
            lower_decimal = Decimal(str(raw_lower)).quantize(
                _TAPE5_WAVENUMBER_QUANTUM_CM1,
                rounding=ROUND_FLOOR,
            )
            upper_decimal = Decimal(str(raw_upper)).quantize(
                _TAPE5_WAVENUMBER_QUANTUM_CM1,
                rounding=ROUND_CEILING,
            )
    except InvalidOperation as exc:
        raise LblrtmInputError("wavenumber bounds exceed the supported TAPE5 F10.3 range") from exc
    # Decimal(str(float)) gives a stable shortest round-trip spelling.  Recheck
    # the binary floats explicitly so an exactly-on-grid representation cannot
    # move inward by a final float conversion.
    if float(lower_decimal) > raw_lower:
        lower_decimal -= _TAPE5_WAVENUMBER_QUANTUM_CM1
    if float(upper_decimal) < raw_upper:
        upper_decimal += _TAPE5_WAVENUMBER_QUANTUM_CM1
    if lower_decimal <= 0:
        raise LblrtmInputError(
            "the requested range reaches 0 cm-1 once its TAPE5 wavenumbers are rounded outward"
        )
    if upper_decimal <= lower_decimal:
        raise LblrtmInputError(
            "the requested wavenumber range is empty at TAPE5 precision (0.001 cm-1)"
        )
    lower_float = float(lower_decimal)
    upper_float = float(upper_decimal)
    lower_text = f"{lower_float:10.3f}"
    upper_text = f"{upper_float:10.3f}"
    if (
        len(lower_text) != 10
        or len(upper_text) != 10
        or float(lower_text) != lower_float
        or float(upper_text) != upper_float
        or lower_float > raw_lower
        or upper_float < raw_upper
    ):
        raise LblrtmInputError("outward wavenumber bounds cannot be preserved by TAPE5 F10.3")
    return LblrtmSpectralWindow(
        minimum_wavenumber_cm1=lower_float,
        maximum_wavenumber_cm1=upper_float,
    )


def _wavenumber_cm1_to_f10p3_ticks(value_cm1: float) -> int:
    """Convert one positive F10.3 wavenumber boundary to exact integer ticks."""

    if isinstance(value_cm1, (bool, np.bool_)) or not isinstance(
        value_cm1,
        (int, float, np.integer, np.floating),
    ):
        raise LblrtmInputError("F10.3 wavenumber boundary must be a real numeric scalar, not bool")
    try:
        with localcontext(Context(prec=64)):
            decimal_value = Decimal(str(value_cm1))
            quotient = decimal_value / _TAPE5_WAVENUMBER_QUANTUM_CM1
            integral = quotient.to_integral_exact()
    except (InvalidOperation, ValueError) as exc:
        raise LblrtmInputError("F10.3 wavenumber boundary must be a finite decimal value") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise LblrtmInputError("F10.3 wavenumber boundary must be finite and positive")
    if quotient != integral:
        raise LblrtmInputError("wavenumber boundary must be an exact multiple of 0.001 cm-1")
    return int(integral)


def _f10p3_ticks_to_wavenumber_cm1(ticks: int) -> float:
    """Convert positive integer F10.3 ticks to a representable wavenumber."""

    if isinstance(ticks, bool) or not isinstance(ticks, int):
        raise LblrtmInputError("internal error: TAPE5 wavenumbers must be whole 0.001 cm-1 steps")
    if ticks <= 0:
        raise LblrtmInputError("internal error: TAPE5 wavenumbers must be positive")
    with localcontext(Context(prec=64)):
        decimal_value = Decimal(ticks) * _TAPE5_WAVENUMBER_QUANTUM_CM1
    value = float(decimal_value)
    rendered = f"{value:10.3f}"
    if (
        len(rendered) != 10
        or not math.isfinite(value)
        or Decimal(rendered.strip()) != decimal_value
    ):
        raise LblrtmInputError("the wavenumber does not fit the TAPE5 F10.3 field")
    return value


def _calculate_balanced_segment_width_ticks(
    complete_width_ticks: int,
) -> tuple[int, ...]:
    """Return minimum-count, balanced rendered widths for one full window."""

    if isinstance(complete_width_ticks, bool) or not isinstance(complete_width_ticks, int):
        raise LblrtmInputError(
            "internal error: the wavenumber width must be whole 0.001 cm-1 steps"
        )
    if complete_width_ticks <= 0:
        raise LblrtmInputError("internal error: the wavenumber width must be positive")
    with localcontext(Context(prec=64)):
        frozen_width_quotient = _MAXIMUM_RENDERED_SEGMENT_WIDTH_CM1 / _TAPE5_WAVENUMBER_QUANTUM_CM1
        frozen_overlap_quotient = (
            _ADJACENT_RENDERED_SEGMENT_OVERLAP_CM1 / _TAPE5_WAVENUMBER_QUANTUM_CM1
        )
        frozen_half_overlap_quotient = (
            _HALF_RENDERED_SEGMENT_OVERLAP_CM1 / _TAPE5_WAVENUMBER_QUANTUM_CM1
        )
        frozen_quotients = (
            frozen_width_quotient,
            frozen_overlap_quotient,
            frozen_half_overlap_quotient,
        )
        if any(quotient != quotient.to_integral_exact() for quotient in frozen_quotients):
            raise RuntimeError(
                "internal error: the LBLRTM segment constants must be whole 0.001 cm-1 steps"
            )
        (
            frozen_width_ticks,
            frozen_overlap_ticks,
            frozen_half_overlap_ticks,
        ) = (int(quotient) for quotient in frozen_quotients)
    if (
        frozen_width_ticks != _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS
        or frozen_overlap_ticks != _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
        or frozen_half_overlap_ticks != _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
        or _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS != 2 * _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
        or _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS <= _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
    ):
        raise RuntimeError("internal error: invalid LBLRTM segment constants")

    maximum_complete_width_ticks = (
        _MAXIMUM_SEGMENT_COUNT * _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS
        - (_MAXIMUM_SEGMENT_COUNT - 1) * _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
    )
    if complete_width_ticks > maximum_complete_width_ticks:
        raise LblrtmInputError(
            "the requested range needs more than "
            f"{_MAXIMUM_SEGMENT_COUNT} LBLRTM segments; request a narrower wavelength range"
        )

    if complete_width_ticks <= _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS:
        return (complete_width_ticks,)

    capacity_step = _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS - _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
    numerator = complete_width_ticks - _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
    segment_count = (numerator + capacity_step - 1) // capacity_step
    if segment_count > _MAXIMUM_SEGMENT_COUNT:
        raise LblrtmInputError(
            "the requested range needs too many LBLRTM segments; request a narrower "
            "wavelength range"
        )

    total_rendered_width_ticks = (
        complete_width_ticks + (segment_count - 1) * _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
    )
    base_width, remainder = divmod(
        total_rendered_width_ticks,
        segment_count,
    )
    widths = tuple(base_width + (1 if index < remainder else 0) for index in range(segment_count))
    if (
        any(width <= 0 for width in widths)
        or any(width > _MAXIMUM_RENDERED_SEGMENT_WIDTH_TICKS for width in widths)
        or sum(widths) != total_rendered_width_ticks
    ):
        raise LblrtmInputError("cannot construct valid balanced LBLRTM segment widths")
    return widths


def _compute_lblrtm_segment_plan_sha256(
    plan: _LblrtmMultisegmentPlan,
) -> str:
    """Return the SHA-256 of one segment plan."""

    if not isinstance(plan, _LblrtmMultisegmentPlan):
        raise LblrtmInputError("internal error: the segment plan has the wrong type")
    payload = {
        "schema_version": _SEGMENT_PLAN_SCHEMA_VERSION,
        "policy_id": _SEGMENT_PLAN_POLICY_ID,
        "boundary_quantum_cm1": str(_TAPE5_WAVENUMBER_QUANTUM_CM1),
        "complete_window_ticks": list(plan.complete_window_ticks),
        "segments": [
            {
                "segment_index": segment.segment_index,
                "rendered_window_ticks": list(segment.rendered_window_ticks),
                "core_window_ticks": list(segment.core_window_ticks),
                "core_upper_inclusive": segment.core_upper_inclusive,
            }
            for segment in plan.segments
        ],
    }
    try:
        payload_bytes = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        framed = _SEGMENT_PLAN_HASH_DOMAIN + struct.pack(">Q", len(payload_bytes)) + payload_bytes
    except (TypeError, ValueError, OverflowError, struct.error) as exc:
        raise LblrtmInputError(
            "internal error: cannot serialize the segment plan for its checksum"
        ) from exc
    return hashlib.sha256(framed).hexdigest()


def _validate_integer_tick_window(
    window: tuple[int, int],
    *,
    field_name: str,
) -> None:
    """Require one strictly increasing two-integer tick interval."""

    if not isinstance(window, tuple) or len(window) != 2:
        raise LblrtmInputError(f"{field_name} must be a two-item tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in window):
        raise LblrtmInputError(f"internal error: {field_name} must be whole 0.001 cm-1 steps")
    if window[0] <= 0 or window[1] <= window[0]:
        raise LblrtmInputError(f"{field_name} must be finite-positive and strictly increasing")


def _validate_lblrtm_multisegment_plan(
    plan: _LblrtmMultisegmentPlan,
) -> None:
    """Raise unless a plan follows the fixed segmentation rules exactly."""

    if not isinstance(plan, _LblrtmMultisegmentPlan):
        raise LblrtmInputError("internal error: plan must be an _LblrtmMultisegmentPlan")
    _validate_integer_tick_window(
        plan.complete_window_ticks,
        field_name="complete_window_ticks",
    )
    if not isinstance(plan.segments, tuple) or not plan.segments:
        raise LblrtmInputError("segments must be a non-empty tuple")
    if len(plan.segments) > _MAXIMUM_SEGMENT_COUNT:
        raise LblrtmInputError(
            f"internal error: more than {_MAXIMUM_SEGMENT_COUNT} LBLRTM segments"
        )

    complete_lower, complete_upper = plan.complete_window_ticks
    expected_widths = _calculate_balanced_segment_width_ticks(complete_upper - complete_lower)
    if len(plan.segments) != len(expected_widths):
        raise LblrtmInputError(
            "internal error: the number of LBLRTM segments is not the minimum needed"
        )

    previous_rendered_end: int | None = None
    previous_core_upper: int | None = None
    for index, (segment, expected_width) in enumerate(
        zip(plan.segments, expected_widths, strict=False)
    ):
        if not isinstance(segment, _LblrtmPlannedSpectralSegment):
            raise LblrtmInputError(
                "internal error: segments must contain only _LblrtmPlannedSpectralSegment"
            )
        if (
            isinstance(segment.segment_index, bool)
            or not isinstance(segment.segment_index, int)
            or segment.segment_index != index
        ):
            raise LblrtmInputError("segment_index must be consecutive from zero")
        if not isinstance(segment.core_upper_inclusive, bool):
            raise LblrtmInputError("core_upper_inclusive must be bool")
        _validate_integer_tick_window(
            segment.rendered_window_ticks,
            field_name=f"segments[{index}].rendered_window_ticks",
        )
        _validate_integer_tick_window(
            segment.core_window_ticks,
            field_name=f"segments[{index}].core_window_ticks",
        )
        for boundary_ticks in segment.rendered_window_ticks:
            _f10p3_ticks_to_wavenumber_cm1(boundary_ticks)

        rendered_lower, rendered_upper = segment.rendered_window_ticks
        core_lower, core_upper = segment.core_window_ticks
        if rendered_upper - rendered_lower != expected_width:
            raise LblrtmInputError(
                f"internal error: segment {index} does not have the planned width"
            )
        expected_rendered_lower = (
            complete_lower
            if index == 0
            else previous_rendered_end - _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
        )
        if rendered_lower != expected_rendered_lower:
            raise LblrtmInputError(
                f"internal error: segment {index} does not overlap the previous segment as planned"
            )
        if (
            index > 0
            and previous_rendered_end - rendered_lower != _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS
        ):
            raise LblrtmInputError(
                f"segment {index} does not overlap its predecessor by exactly 10.000 cm-1"
            )

        expected_core_lower = complete_lower if index == 0 else previous_core_upper
        expected_core_upper = (
            complete_upper
            if index == len(plan.segments) - 1
            else rendered_upper - _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
        )
        if (core_lower, core_upper) != (
            expected_core_lower,
            expected_core_upper,
        ):
            raise LblrtmInputError(f"segment {index} core does not follow fixed midpoint ownership")
        if not (rendered_lower <= core_lower < core_upper <= rendered_upper):
            raise LblrtmInputError(
                f"segment {index} core must be non-empty and contained in its rendered window"
            )
        expected_inclusive = index == len(plan.segments) - 1
        if segment.core_upper_inclusive is not expected_inclusive:
            raise LblrtmInputError("only the final segment core may include its upper boundary")
        if index > 0:
            source_switch = core_lower
            if (
                source_switch != previous_rendered_end - _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
                or source_switch != rendered_lower + _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
            ):
                raise LblrtmInputError(f"segment {index} source switch is not the overlap midpoint")

        previous_rendered_end = rendered_upper
        previous_core_upper = core_upper

    if plan.segments[0].rendered_window_ticks[0] != complete_lower:
        raise LblrtmInputError("first rendered segment must start at the complete lower bound")
    if plan.segments[-1].rendered_window_ticks[1] != complete_upper:
        raise LblrtmInputError("last rendered segment must end at the complete upper bound")
    if plan.segments[0].core_window_ticks[0] != complete_lower:
        raise LblrtmInputError("first core must start at the complete lower bound")
    if plan.segments[-1].core_window_ticks[1] != complete_upper:
        raise LblrtmInputError("last core must end at the complete upper bound")

    if (
        not isinstance(plan.segment_plan_sha256, str)
        or len(plan.segment_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan.segment_plan_sha256)
        or plan.segment_plan_sha256 != _compute_lblrtm_segment_plan_sha256(plan)
    ):
        raise LblrtmInputError("internal error: segment_plan_sha256 does not match the plan")


def _build_lblrtm_multisegment_plan(
    complete_window: LblrtmSpectralWindow,
) -> _LblrtmMultisegmentPlan:
    """Build and check one segment plan in whole steps of 0.001 cm-1."""

    if not isinstance(complete_window, LblrtmSpectralWindow):
        raise LblrtmInputError("complete_window must be an LblrtmSpectralWindow")
    complete_lower = _wavenumber_cm1_to_f10p3_ticks(complete_window.minimum_wavenumber_cm1)
    complete_upper = _wavenumber_cm1_to_f10p3_ticks(complete_window.maximum_wavenumber_cm1)
    complete_width = complete_upper - complete_lower
    widths = _calculate_balanced_segment_width_ticks(complete_width)

    segments: list[_LblrtmPlannedSpectralSegment] = []
    rendered_lower = complete_lower
    previous_source_switch = complete_lower
    for index, width in enumerate(widths):
        rendered_upper = rendered_lower + width
        final_segment = index == len(widths) - 1
        core_upper = (
            complete_upper
            if final_segment
            else rendered_upper - _HALF_RENDERED_SEGMENT_OVERLAP_TICKS
        )
        segments.append(
            _LblrtmPlannedSpectralSegment(
                segment_index=index,
                rendered_window_ticks=(
                    rendered_lower,
                    rendered_upper,
                ),
                core_window_ticks=(
                    previous_source_switch,
                    core_upper,
                ),
                core_upper_inclusive=final_segment,
            )
        )
        previous_source_switch = core_upper
        rendered_lower = rendered_upper - _ADJACENT_RENDERED_SEGMENT_OVERLAP_TICKS

    provisional_plan = _LblrtmMultisegmentPlan(
        complete_window_ticks=(complete_lower, complete_upper),
        segments=tuple(segments),
        segment_plan_sha256="",
    )
    plan = _LblrtmMultisegmentPlan(
        complete_window_ticks=provisional_plan.complete_window_ticks,
        segments=provisional_plan.segments,
        segment_plan_sha256=_compute_lblrtm_segment_plan_sha256(provisional_plan),
    )
    _validate_lblrtm_multisegment_plan(plan)
    return plan


def _expand_spectral_window_for_output_medium_coverage(
    spectral_window: LblrtmSpectralWindow,
    *,
    expanded_input_wavelength_range_nm: tuple[float, float],
    wavelength_medium: str,
    wavelength_conversion_formula: str,
) -> LblrtmSpectralWindow:
    """Preserve outward support after the frozen one-step air conversion.

    ``air_to_vac`` and ``vac_to_air`` intentionally use the project's
    literature-faithful one-step conventions and are therefore not exact
    numerical inverses.  For an air-labelled request, an outward-rounded
    vacuum-wavenumber window can consequently map a few representation units
    inward.  Expand only in exact TAPE5 quanta until the returned-medium
    calculation support encloses the caller's padded interval.
    """

    if wavelength_medium == "vacuum":
        return spectral_window

    lower_decimal = Decimal(str(spectral_window.minimum_wavenumber_cm1))
    upper_decimal = Decimal(str(spectral_window.maximum_wavenumber_cm1))
    decimal_context = Context(prec=64)
    for _ in range(_MAXIMUM_AIR_BOUND_EXPANSION_STEPS):
        lower_float = float(lower_decimal)
        upper_float = float(upper_decimal)
        vacuum_wavelength_bounds = np.array(
            [
                _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / upper_float,
                _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / lower_float,
            ],
            dtype=np.float64,
        )
        try:
            returned_medium_bounds = np.asarray(
                vac_to_air(
                    vacuum_wavelength_bounds,
                    formula=wavelength_conversion_formula,
                ),
                dtype=np.float64,
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
            FloatingPointError,
            MemoryError,
        ) as exc:
            raise LblrtmInputError(
                "cannot check that the air-wavelength range is covered after rounding the "
                f"TAPE5 wavenumbers (formula {wavelength_conversion_formula!r})"
            ) from exc
        if (
            returned_medium_bounds.shape != (2,)
            or not np.all(np.isfinite(returned_medium_bounds))
            or np.any(returned_medium_bounds <= 0.0)
            or not returned_medium_bounds[1] > returned_medium_bounds[0]
        ):
            raise LblrtmInputError(
                "air-wavelength conversion must preserve two finite, "
                "positive, strictly increasing TAPE5 boundary wavelengths"
            )

        coordinate_tolerance = _floating_coordinate_tolerance(
            *expanded_input_wavelength_range_nm,
            *returned_medium_bounds,
        )
        lower_is_covered = (
            float(returned_medium_bounds[0])
            <= expanded_input_wavelength_range_nm[0] + coordinate_tolerance
        )
        upper_is_covered = (
            float(returned_medium_bounds[1])
            >= expanded_input_wavelength_range_nm[1] - coordinate_tolerance
        )
        if lower_is_covered and upper_is_covered:
            return _quantize_wavenumber_range_outward_for_tape5((lower_float, upper_float))
        if not lower_is_covered:
            upper_decimal = decimal_context.add(
                upper_decimal,
                _TAPE5_WAVENUMBER_QUANTUM_CM1,
            )
        if not upper_is_covered:
            lower_decimal = decimal_context.subtract(
                lower_decimal,
                _TAPE5_WAVENUMBER_QUANTUM_CM1,
            )
            if lower_decimal <= 0:
                raise LblrtmInputError(
                    "air-wavelength coverage expansion makes the minimum "
                    "TAPE5 wavenumber non-positive"
                )
        # Re-run every F10.3 representability gate after each step.
        _quantize_wavenumber_range_outward_for_tape5((float(lower_decimal), float(upper_decimal)))

    raise LblrtmInputError(
        "cannot cover the padded air-wavelength range with TAPE5 wavenumbers rounded "
        f"outward (tried {_MAXIMUM_AIR_BOUND_EXPANSION_STEPS} steps of 0.001 cm-1)"
    )


def _validate_spectral_window_inside_resource_support(
    spectral_window: LblrtmSpectralWindow,
    *,
    resources: ResolvedLblrtmResourceFiles,
) -> tuple[float, float]:
    """Require this rendered calculation to lie inside validated TAPE3 support."""

    if not isinstance(resources, ResolvedLblrtmResourceFiles):
        raise LblrtmInputError(
            "resources must be a ResolvedLblrtmResourceFiles instance "
            "returned by validate_lblrtm_resource_bundle"
        )
    minimum_supported, maximum_supported = resources.require_validated_calculation_bounds_cm1()
    coordinate_tolerance = _floating_coordinate_tolerance(
        minimum_supported,
        maximum_supported,
        spectral_window.minimum_wavenumber_cm1,
        spectral_window.maximum_wavenumber_cm1,
    )
    if (
        spectral_window.minimum_wavenumber_cm1 < minimum_supported - coordinate_tolerance
        or spectral_window.maximum_wavenumber_cm1 > maximum_supported + coordinate_tolerance
    ):
        raise LblrtmInputError(
            "rendered LBLRTM spectral window lies outside the "
            "validator-bound TAPE3 safe calculation support "
            f"[{minimum_supported}, {maximum_supported}] cm-1"
        )
    return minimum_supported, maximum_supported


def _snapshot_lblrtm_full_profile_arrays(
    full_profile: Mapping[str, object],
) -> Mapping[str, np.ndarray]:
    """Copy the complete MODEL=0 atmosphere once into read-only float64 arrays."""

    if not isinstance(full_profile, Mapping):
        raise LblrtmInputError("full_profile must be a mapping of level arrays")
    captured_objects: dict[str, np.ndarray] = {}
    try:
        for field_name in _PROFILE_FIELD_NAMES:
            supplied = full_profile[field_name]
            captured_objects[field_name] = np.array(
                supplied,
                dtype=object,
                copy=True,
            )
    except KeyError as exc:
        raise LblrtmInputError(f"full_profile is missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError("internal error: cannot copy the atmospheric profile") from exc
    except MemoryError as exc:
        raise LblrtmInputError(
            "cannot allocate memory for a copy of the atmospheric profile"
        ) from exc

    captured_profile: Mapping[str, object] = MappingProxyType(captured_objects)
    # Validate the captured object arrays before float conversion so booleans
    # retain exactly the writer's rejection semantics instead of becoming 1.0.
    validate_lblrtm_layer_input(
        captured_profile,
        _PROFILE_MOLECULE_NAMES,
    )
    snapshots: dict[str, np.ndarray] = {}
    try:
        for field_name, object_array in captured_objects.items():
            numeric_array = np.array(
                object_array,
                dtype=np.float64,
                copy=True,
            )
            snapshots[field_name] = np.frombuffer(
                numeric_array.tobytes(order="C"),
                dtype=np.float64,
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(
            "internal error: cannot normalize the copy of the atmospheric profile"
        ) from exc
    except MemoryError as exc:
        raise LblrtmInputError(
            "cannot allocate memory for the normalized copy of the atmospheric profile"
        ) from exc

    frozen_profile: Mapping[str, np.ndarray] = MappingProxyType(snapshots)
    # Revalidate only the copied values.  Every subsequent renderer receives
    # this exact mapping, never the caller-owned mapping or arrays.
    validate_lblrtm_layer_input(
        frozen_profile,
        _PROFILE_MOLECULE_NAMES,
    )
    return frozen_profile


def _build_lblrtm_multisegment_physical_input_snapshot(
    full_profile: Mapping[str, object],
    *,
    normalized_request: _NormalizedTransmissionRequest,
    line_of_sight: LblrtmLineOfSight,
    molecule_relative_columns: Mapping[str, float],
    resources: ResolvedLblrtmResourceFiles,
    resource_safe_wavenumber_range_cm1: tuple[float, float],
    continuum_mode: str,
    timeout_s: float | None,
    verbose: bool,
) -> _LblrtmMultisegmentPhysicalInputSnapshot:
    """Freeze all physical inputs before the first segment is rendered."""

    resource_fingerprint = resources.manifest_fingerprint
    if (
        not isinstance(resource_fingerprint, str)
        or len(resource_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in resource_fingerprint)
    ):
        raise LblrtmInputError(
            "internal error: the checked LBLRTM bundle has no valid manifest checksum"
        )
    frozen_relative_columns = MappingProxyType(dict(molecule_relative_columns))
    return _LblrtmMultisegmentPhysicalInputSnapshot(
        normalized_request=normalized_request,
        full_profile=_snapshot_lblrtm_full_profile_arrays(full_profile),
        line_of_sight=line_of_sight,
        molecule_relative_columns=frozen_relative_columns,
        continuum_mode=continuum_mode,
        resources=resources,
        resource_fingerprint=resource_fingerprint,
        resource_safe_wavenumber_range_cm1=(resource_safe_wavenumber_range_cm1),
        timeout_s=timeout_s,
        verbose=verbose,
    )


def _planned_segment_to_run_configuration(
    planned_segment: _LblrtmPlannedSpectralSegment,
    *,
    physical_inputs: _LblrtmMultisegmentPhysicalInputSnapshot,
) -> LblrtmRunConfiguration:
    """Convert exact integer rendered ticks into one normal writer request."""

    rendered_lower_ticks, rendered_upper_ticks = planned_segment.rendered_window_ticks
    return LblrtmRunConfiguration(
        spectral_window=LblrtmSpectralWindow(
            minimum_wavenumber_cm1=(_f10p3_ticks_to_wavenumber_cm1(rendered_lower_ticks)),
            maximum_wavenumber_cm1=(_f10p3_ticks_to_wavenumber_cm1(rendered_upper_ticks)),
        ),
        line_of_sight=physical_inputs.line_of_sight,
        continuum_mode=physical_inputs.continuum_mode,
    )


def _freeze_lblrtm_float64_output_array(
    value: object,
    *,
    field_name: str,
) -> np.ndarray:
    """Return a read-only copy of one parsed output array."""

    try:
        array = np.array(value, dtype=np.float64, copy=True)
        return np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmOutputError(f"{field_name} cannot be represented as float64") from exc
    except MemoryError as exc:
        raise LblrtmOutputError(f"cannot allocate memory for a copy of {field_name}") from exc


def _validate_complete_planned_lblrtm_segment_result(
    segment: _LblrtmSpectralSegmentResult,
    *,
    planned_segment: _LblrtmPlannedSpectralSegment,
    expected_resource_fingerprint: str,
    allow_nonphysical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the complete parsed TAPE12 before ownership is applied."""

    wavenumber = _freeze_lblrtm_float64_output_array(
        segment.wavenumber_cm1,
        field_name="parsed TAPE12 wavenumber",
    )
    transmission = _freeze_lblrtm_float64_output_array(
        segment.transmission,
        field_name="parsed TAPE12 transmission",
    )
    if (
        wavenumber.ndim != 1
        or transmission.ndim != 1
        or wavenumber.shape != transmission.shape
        or wavenumber.size < 2
    ):
        raise LblrtmOutputError(
            "complete parsed TAPE12 arrays must be equal-length, "
            "one-dimensional, and contain at least two samples"
        )
    if (
        not np.all(np.isfinite(wavenumber))
        or np.any(wavenumber <= 0.0)
        or np.any(np.diff(wavenumber) <= 0.0)
    ):
        raise LblrtmOutputError(
            "complete parsed TAPE12 wavenumbers must be finite, positive, and strictly increasing"
        )
    if not np.all(np.isfinite(transmission)) and not allow_nonphysical:
        raise LblrtmOutputError("complete parsed TAPE12 transmission contains NaN or infinity")
    if not allow_nonphysical and (
        np.any(transmission < 0.0) or np.any(transmission > _MAXIMUM_TRANSMISSION)
    ):
        raise LblrtmOutputError(
            f"complete parsed TAPE12 transmission lies outside [0, {_MAXIMUM_TRANSMISSION}]"
        )
    if segment.run_result.resource_fingerprint != expected_resource_fingerprint:
        raise LblrtmOutputError(
            "internal error: an LBLRTM segment ran with a different bundle than the other segments"
        )

    core_lower_ticks, core_upper_ticks = planned_segment.core_window_ticks
    _validate_lblrtm_segment_matches_requested_window(
        _LblrtmSpectralSegmentResult(
            spectral_window=segment.spectral_window,
            wavenumber_cm1=wavenumber,
            transmission=transmission,
            run_result=segment.run_result,
        ),
        requested_wavenumber_range_cm1=(
            _f10p3_ticks_to_wavenumber_cm1(core_lower_ticks),
            _f10p3_ticks_to_wavenumber_cm1(core_upper_ticks),
        ),
    )
    return wavenumber, transmission


def _select_owned_lblrtm_samples(
    wavenumber_cm1: np.ndarray,
    transmission: np.ndarray,
    *,
    planned_segment: _LblrtmPlannedSpectralSegment,
) -> tuple[np.ndarray, np.ndarray]:
    """Select owned samples and retain an outer endpoint bracket if needed.

    Interior ownership remains the exact frozen half-open rule.  At the first
    and final physical request boundaries there is no neighboring segment to
    own a TAPE12 sample just outside the core.  Retaining the first TAPE12
    sample at or beyond the final upper boundary (and any first-segment sample
    below its lower boundary) preserves complete source-grid coverage without
    extrapolation, interpolation, clipping, or a tolerance change.
    """

    core_lower_ticks, core_upper_ticks = planned_segment.core_window_ticks
    core_lower = _f10p3_ticks_to_wavenumber_cm1(core_lower_ticks)
    core_upper = _f10p3_ticks_to_wavenumber_cm1(core_upper_ticks)
    if planned_segment.segment_index == 0:
        lower_mask = np.ones(wavenumber_cm1.shape, dtype=bool)
    else:
        lower_mask = wavenumber_cm1 >= core_lower
    if planned_segment.core_upper_inclusive:
        upper_indices = np.flatnonzero(wavenumber_cm1 >= core_upper)
        if upper_indices.size == 0:
            raise LblrtmOutputError("final LBLRTM segment has no TAPE12 upper endpoint bracket")
        upper_index = int(upper_indices[0])
        ownership_mask = lower_mask & (np.arange(wavenumber_cm1.size) <= upper_index)
    elif planned_segment.segment_index == 0:
        ownership_mask = lower_mask & (wavenumber_cm1 < core_upper)
    else:
        ownership_mask = lower_mask & (wavenumber_cm1 < core_upper)
    try:
        owned_wavenumber = _freeze_lblrtm_float64_output_array(
            wavenumber_cm1[ownership_mask],
            field_name="owned TAPE12 wavenumber",
        )
        owned_transmission = _freeze_lblrtm_float64_output_array(
            transmission[ownership_mask],
            field_name="owned TAPE12 transmission",
        )
    except (IndexError, ValueError) as exc:
        raise LblrtmOutputError(
            "cannot apply fixed ownership to the complete TAPE12 segment"
        ) from exc
    if owned_wavenumber.size == 0:
        raise LblrtmOutputError(
            f"the LBLRTM output of segment {planned_segment.segment_index} has no samples "
            "in the part of the range it supplies"
        )
    return owned_wavenumber, owned_transmission


def _compute_lblrtm_spectrum_semantic_sha256(
    coordinate: ArrayLike,
    transmission: ArrayLike,
    *,
    coordinate_kind_and_unit: str,
    wavelength_medium_or_empty: str,
    allow_nonphysical: bool = False,
) -> str:
    """Hash one scientific spectrum with the frozen byte framing."""

    if not isinstance(coordinate_kind_and_unit, str):
        raise LblrtmOutputError("coordinate_kind_and_unit must be a string")
    if not isinstance(wavelength_medium_or_empty, str):
        raise LblrtmOutputError("wavelength_medium_or_empty must be a string")
    try:
        coordinate_array = np.asarray(coordinate, dtype=np.float64)
        transmission_array = np.asarray(transmission, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise LblrtmOutputError(
            "internal error: the spectrum checksum needs float64 arrays"
        ) from exc
    if (
        coordinate_array.ndim != 1
        or transmission_array.ndim != 1
        or coordinate_array.shape != transmission_array.shape
        or coordinate_array.size < 2
    ):
        raise LblrtmOutputError(
            "internal error: the spectrum checksum needs two equal-length 1-D arrays "
            "with at least two samples"
        )
    if (
        not np.all(np.isfinite(coordinate_array))
        or np.any(coordinate_array <= 0.0)
        or np.any(np.diff(coordinate_array) <= 0.0)
    ):
        raise LblrtmOutputError(
            "internal error: the spectrum wavenumbers must be finite, positive and increasing"
        )
    if not allow_nonphysical and (
        not np.all(np.isfinite(transmission_array))
        or np.any(transmission_array < 0.0)
        or np.any(transmission_array > _MAXIMUM_TRANSMISSION)
    ):
        raise LblrtmOutputError(
            f"the LBLRTM transmission lies outside [0, {_MAXIMUM_TRANSMISSION}]"
        )
    try:
        coordinate_payload = np.ascontiguousarray(
            coordinate_array,
            dtype=np.dtype("<f8"),
        )
        transmission_payload = np.ascontiguousarray(
            transmission_array,
            dtype=np.dtype("<f8"),
        )
        coordinate_text = coordinate_kind_and_unit.encode("utf-8")
        medium_text = wavelength_medium_or_empty.encode("utf-8")
        digest = hashlib.sha256()
        digest.update(_SPECTRUM_SEMANTIC_HASH_DOMAIN)
        digest.update(struct.pack(">Q", _SPECTRUM_SEMANTIC_HASH_SCHEMA_VERSION))
        digest.update(struct.pack(">Q", len(coordinate_text)))
        digest.update(coordinate_text)
        digest.update(struct.pack(">Q", len(medium_text)))
        digest.update(medium_text)
        digest.update(struct.pack(">Q", 1))
        digest.update(struct.pack(">Q", int(coordinate_array.size)))
        digest.update(struct.pack(">Q", int(coordinate_payload.nbytes)))
        digest.update(memoryview(coordinate_payload).cast("B"))
        digest.update(struct.pack(">Q", int(transmission_payload.nbytes)))
        digest.update(memoryview(transmission_payload).cast("B"))
    except (
        UnicodeEncodeError,
        OverflowError,
        struct.error,
        TypeError,
        BufferError,
        MemoryError,
    ) as exc:
        raise LblrtmOutputError(
            "internal error: cannot serialize the spectrum for its checksum"
        ) from exc
    return digest.hexdigest()


def _add_lblrtm_multisegment_failure_provenance_note(
    exception: BaseException,
    *,
    segment_plan: _LblrtmMultisegmentPlan,
    planned_segment: _LblrtmPlannedSpectralSegment | None,
    resource_fingerprint: str,
    tape5_sha256: str = "unavailable",
    tape6_sha256: str = "unavailable",
    tape12_sha256: str = "unavailable",
    completed_segment_count: int,
) -> None:
    """Attach a nine-field description of the first failure (without file paths)."""

    note_prefix = "LBLRTM multisegment failure provenance:"
    existing_evidence = (
        *(str(argument) for argument in exception.args),
        *(str(note) for note in getattr(exception, "__notes__", ())),
    )
    if any(note_prefix in evidence for evidence in existing_evidence):
        return
    if planned_segment is None:
        failing_segment_index: object = "none"
        core_range: object = "unavailable"
        rendered_range: object = "unavailable"
    else:
        failing_segment_index = planned_segment.segment_index
        core_range = tuple(
            _f10p3_ticks_to_wavenumber_cm1(value) for value in planned_segment.core_window_ticks
        )
        rendered_range = tuple(
            _f10p3_ticks_to_wavenumber_cm1(value) for value in planned_segment.rendered_window_ticks
        )
    _add_exception_note(
        exception,
        f"{note_prefix} "
        f"failing_segment_index={failing_segment_index}, "
        f"segment_plan_sha256={segment_plan.segment_plan_sha256}, "
        f"core_wavenumber_range_cm1={core_range}, "
        f"rendered_wavenumber_range_cm1={rendered_range}, "
        f"resource_fingerprint={resource_fingerprint}, "
        f"tape5_sha256_if_available={tape5_sha256}, "
        f"tape6_sha256_if_available={tape6_sha256}, "
        f"tape12_sha256_if_available={tape12_sha256}, "
        f"completed_segment_count={completed_segment_count}",
    )


def _execute_and_validate_one_planned_lblrtm_segment(
    *,
    planned_segment: _LblrtmPlannedSpectralSegment,
    segment_plan: _LblrtmMultisegmentPlan,
    physical_inputs: _LblrtmMultisegmentPhysicalInputSnapshot,
    completed_segment_count: int,
    quality_mode: str = "strict",
) -> _LblrtmValidatedSpectralSegmentResult:
    """Run writer, isolated executable, and strict reader for one segment."""

    tape5_sha256 = "unavailable"
    run_result: LblrtmRunResult | None = None
    try:
        run_configuration = _planned_segment_to_run_configuration(
            planned_segment,
            physical_inputs=physical_inputs,
        )
        tape5_text = render_lblrtm_tape5(
            physical_inputs.full_profile,
            run_configuration,
            molecule_relative_columns=(physical_inputs.molecule_relative_columns),
        )
        tape5_sha256 = hashlib.sha256(tape5_text.encode("ascii")).hexdigest()
        run_result = execute_lblrtm(
            tape5_text,
            physical_inputs.resources,
            timeout_s=physical_inputs.timeout_s,
            verbose=physical_inputs.verbose,
        )
        if run_result.tape5_sha256 != tape5_sha256:
            raise LblrtmOutputError(
                "internal error: the TAPE5 given to LBLRTM differs from the one written"
            )
        if quality_mode == "evaluate":
            parsed_wavenumber, parsed_transmission, _flags, _usable = (
                read_lblrtm_tape12_transmission_quality(run_result.tape12_bytes)
            )
        else:
            parsed_wavenumber, parsed_transmission = read_lblrtm_tape12_transmission(
                run_result.tape12_bytes
            )
        tape12_abi = _validate_tape12_abi_against_resources(
            run_result.tape12_bytes,
            physical_inputs.resources,
        )
        raw_segment = _LblrtmSpectralSegmentResult(
            spectral_window=run_configuration.spectral_window,
            wavenumber_cm1=parsed_wavenumber,
            transmission=parsed_transmission,
            run_result=run_result,
            tape12_abi=tape12_abi,
        )
        wavenumber, transmission = _validate_complete_planned_lblrtm_segment_result(
            raw_segment,
            planned_segment=planned_segment,
            expected_resource_fingerprint=(physical_inputs.resource_fingerprint),
            allow_nonphysical=(quality_mode == "evaluate"),
        )
        owned_wavenumber, owned_transmission = _select_owned_lblrtm_samples(
            wavenumber,
            transmission,
            planned_segment=planned_segment,
        )
        semantic_sha256 = _compute_lblrtm_spectrum_semantic_sha256(
            wavenumber,
            transmission,
            coordinate_kind_and_unit="vacuum_wavenumber_cm-1",
            wavelength_medium_or_empty="",
            allow_nonphysical=(quality_mode == "evaluate"),
        )
        return _LblrtmValidatedSpectralSegmentResult(
            planned_segment=planned_segment,
            spectral_window=run_configuration.spectral_window,
            wavenumber_cm1=wavenumber,
            transmission=transmission,
            owned_wavenumber_cm1=owned_wavenumber,
            owned_transmission=owned_transmission,
            tape12_spectrum_semantic_sha256=semantic_sha256,
            resource_fingerprint=run_result.resource_fingerprint,
            tape5_sha256=run_result.tape5_sha256,
            tape6_sha256=run_result.tape6_sha256,
            tape12_sha256=run_result.tape12_sha256,
            tape12_abi=tape12_abi,
        )
    except MemoryError as exc:
        output_error = LblrtmOutputError(
            "cannot allocate memory while processing one LBLRTM segment"
        )
        _add_lblrtm_multisegment_failure_provenance_note(
            output_error,
            segment_plan=segment_plan,
            planned_segment=planned_segment,
            resource_fingerprint=physical_inputs.resource_fingerprint,
            tape5_sha256=tape5_sha256,
            tape6_sha256=(run_result.tape6_sha256 if run_result is not None else "unavailable"),
            tape12_sha256=(run_result.tape12_sha256 if run_result is not None else "unavailable"),
            completed_segment_count=completed_segment_count,
        )
        raise output_error from exc
    except Exception as exc:
        _add_lblrtm_multisegment_failure_provenance_note(
            exc,
            segment_plan=segment_plan,
            planned_segment=planned_segment,
            resource_fingerprint=physical_inputs.resource_fingerprint,
            tape5_sha256=(run_result.tape5_sha256 if run_result is not None else tape5_sha256),
            tape6_sha256=(run_result.tape6_sha256 if run_result is not None else "unavailable"),
            tape12_sha256=(run_result.tape12_sha256 if run_result is not None else "unavailable"),
            completed_segment_count=completed_segment_count,
        )
        raise


def _source_switch_coordinate_tolerance_cm1(
    source_switch_wavenumber_cm1: float,
) -> float:
    """Return exactly eight float64 ULPs for seam coordinate structure."""

    return 8.0 * math.ulp(max(abs(float(source_switch_wavenumber_cm1)), 1.0))


def _source_switch_bracketing_spacing_cm1(
    wavenumber_cm1: np.ndarray,
    *,
    source_switch_wavenumber_cm1: float,
    exact_switch_uses_upper_interval: bool,
) -> float:
    """Return the local source-switch cell on the segment's owned side.

    When the switch itself is a TAPE12 sample, a lower-wavenumber owner uses
    the cell immediately below it and a higher-wavenumber owner uses the cell
    immediately above it.  This makes the structural gap bound depend on the
    two sampling cells that actually border the selected seam.
    """

    switch = source_switch_wavenumber_cm1
    if float(wavenumber_cm1[0]) > switch or float(wavenumber_cm1[-1]) < switch:
        raise LblrtmOutputError("complete TAPE12 segment does not bracket the source switch")
    side = "right" if exact_switch_uses_upper_interval else "left"
    upper_index = int(np.searchsorted(wavenumber_cm1, switch, side=side))
    if upper_index <= 0 or upper_index >= wavenumber_cm1.size:
        raise LblrtmOutputError(
            "complete TAPE12 segment lacks two local samples bracketing the source switch"
        )
    spacing = float(wavenumber_cm1[upper_index] - wavenumber_cm1[upper_index - 1])
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise LblrtmOutputError("source-switch bracketing spacing must be finite and positive")
    return spacing


def _build_lblrtm_overlap_diagnostic_grid(
    lower_segment: _LblrtmValidatedSpectralSegmentResult,
    upper_segment: _LblrtmValidatedSpectralSegmentResult,
) -> np.ndarray:
    """Build the inclusive inward-rounded common 0.001 cm-1 grid."""

    planned_lower_ticks = upper_segment.planned_segment.rendered_window_ticks[0]
    planned_upper_ticks = lower_segment.planned_segment.rendered_window_ticks[1]
    actual_lower = max(
        _f10p3_ticks_to_wavenumber_cm1(planned_lower_ticks),
        float(lower_segment.wavenumber_cm1[0]),
        float(upper_segment.wavenumber_cm1[0]),
    )
    actual_upper = min(
        _f10p3_ticks_to_wavenumber_cm1(planned_upper_ticks),
        float(lower_segment.wavenumber_cm1[-1]),
        float(upper_segment.wavenumber_cm1[-1]),
    )
    try:
        with localcontext(Context(prec=64)):
            lower_tick = int(
                (Decimal(str(actual_lower)) / _TAPE5_WAVENUMBER_QUANTUM_CM1).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            upper_tick = int(
                (Decimal(str(actual_upper)) / _TAPE5_WAVENUMBER_QUANTUM_CM1).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise LblrtmOutputError(
            "internal error: cannot round the segment-overlap check range to TAPE5 precision"
        ) from exc
    point_count = upper_tick - lower_tick + 1
    if point_count < 2:
        raise LblrtmOutputError(
            "actual two-segment overlap cannot form the required two-point "
            "0.001 cm-1 diagnostic grid"
        )
    try:
        ticks = np.arange(
            lower_tick,
            upper_tick + 1,
            dtype=np.int64,
        )
        # Division by the exact integer tick scale matches
        # _f10p3_ticks_to_wavenumber_cm1().  Multiplying by binary 0.001 can
        # double-round some otherwise exact F10.3 coordinates by one ULP.
        grid = ticks.astype(np.float64) / 1_000.0
        return _freeze_lblrtm_float64_output_array(
            grid,
            field_name="overlap diagnostic wavenumber grid",
        )
    except (MemoryError, OverflowError) as exc:
        raise LblrtmOutputError("cannot allocate the overlap diagnostic grid") from exc


def _type7_linear_percentile(
    values: np.ndarray,
    probability: float,
) -> float:
    """Return the explicitly defined Hyndman-Fan Type-7 percentile."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise LblrtmOutputError("cannot calculate a percentile from an empty diagnostic")
    position = (ordered.size - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return float(ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index]))


def _validate_lblrtm_segment_seam_structure(
    lower_segment: _LblrtmValidatedSpectralSegmentResult,
    upper_segment: _LblrtmValidatedSpectralSegmentResult,
) -> _LblrtmSeamDiagnostic:
    """Gate structural continuity and record non-gating overlap residuals."""

    if (
        upper_segment.planned_segment.segment_index
        != lower_segment.planned_segment.segment_index + 1
    ):
        raise LblrtmOutputError("LBLRTM segments must be merged in consecutive increasing order")
    source_switch = _f10p3_ticks_to_wavenumber_cm1(
        upper_segment.planned_segment.core_window_ticks[0]
    )
    left_spacing = _source_switch_bracketing_spacing_cm1(
        lower_segment.wavenumber_cm1,
        source_switch_wavenumber_cm1=source_switch,
        exact_switch_uses_upper_interval=False,
    )
    right_spacing = _source_switch_bracketing_spacing_cm1(
        upper_segment.wavenumber_cm1,
        source_switch_wavenumber_cm1=source_switch,
        exact_switch_uses_upper_interval=True,
    )
    selected_left_last = float(lower_segment.owned_wavenumber_cm1[-1])
    selected_right_first = float(upper_segment.owned_wavenumber_cm1[0])
    if not selected_left_last < selected_right_first:
        raise LblrtmOutputError(
            "fixed ownership creates a duplicate, contained, or reordered LBLRTM seam"
        )
    selected_gap = selected_right_first - selected_left_last
    maximum_gap = (
        left_spacing + right_spacing + _source_switch_coordinate_tolerance_cm1(source_switch)
    )
    if selected_gap > maximum_gap:
        raise LblrtmOutputError(
            "fixed-ownership LBLRTM seam exceeds the two local bracketing spacings"
        )

    diagnostic_grid = _build_lblrtm_overlap_diagnostic_grid(
        lower_segment,
        upper_segment,
    )
    if (
        diagnostic_grid[0] < lower_segment.wavenumber_cm1[0]
        or diagnostic_grid[-1] > lower_segment.wavenumber_cm1[-1]
        or diagnostic_grid[0] < upper_segment.wavenumber_cm1[0]
        or diagnostic_grid[-1] > upper_segment.wavenumber_cm1[-1]
    ):
        raise LblrtmOutputError("overlap diagnostic interpolation would require extrapolation")
    try:
        left_transmission = np.interp(
            diagnostic_grid,
            lower_segment.wavenumber_cm1,
            lower_segment.transmission,
            left=np.nan,
            right=np.nan,
        )
        right_transmission = np.interp(
            diagnostic_grid,
            upper_segment.wavenumber_cm1,
            upper_segment.transmission,
            left=np.nan,
            right=np.nan,
        )
        absolute_difference = np.abs(left_transmission - right_transmission)
    except MemoryError as exc:
        raise LblrtmOutputError("cannot allocate report-only overlap diagnostics") from exc
    if not np.all(np.isfinite(absolute_difference)):
        raise LblrtmOutputError("overlap diagnostic interpolation produced NaN or infinity")
    rms = float(math.sqrt(float(np.mean(absolute_difference * absolute_difference))))
    return _LblrtmSeamDiagnostic(
        lower_segment_index=lower_segment.planned_segment.segment_index,
        upper_segment_index=upper_segment.planned_segment.segment_index,
        source_switch_wavenumber_cm1=source_switch,
        planned_overlap_wavenumber_range_cm1=(
            _f10p3_ticks_to_wavenumber_cm1(upper_segment.planned_segment.rendered_window_ticks[0]),
            _f10p3_ticks_to_wavenumber_cm1(lower_segment.planned_segment.rendered_window_ticks[1]),
        ),
        left_bracketing_spacing_cm1=left_spacing,
        right_bracketing_spacing_cm1=right_spacing,
        selected_seam_gap_cm1=selected_gap,
        diagnostic_minimum_wavenumber_cm1=float(diagnostic_grid[0]),
        diagnostic_maximum_wavenumber_cm1=float(diagnostic_grid[-1]),
        diagnostic_point_count=int(diagnostic_grid.size),
        rms=rms,
        p99_absolute=_type7_linear_percentile(
            absolute_difference,
            0.99,
        ),
        maximum_absolute=float(np.max(absolute_difference)),
    )


def _merge_owned_lblrtm_segments(
    segments: tuple[_LblrtmValidatedSpectralSegmentResult, ...],
) -> _LblrtmMergedSpectrum:
    """Concatenate unchanged owned samples without sorting or repair."""

    total_point_count = sum(int(segment.owned_wavenumber_cm1.size) for segment in segments)
    if total_point_count > _MAXIMUM_OUTPUT_GRID_POINTS:
        raise LblrtmOutputError(
            "merged LBLRTM point count exceeds the "
            f"{_MAXIMUM_OUTPUT_GRID_POINTS}-point project limit"
        )
    if total_point_count < 2:
        raise LblrtmOutputError("merged LBLRTM spectrum must contain at least two points")
    try:
        wavenumber = _freeze_lblrtm_float64_output_array(
            np.concatenate(tuple(segment.owned_wavenumber_cm1 for segment in segments)),
            field_name="merged LBLRTM wavenumber",
        )
        transmission = _freeze_lblrtm_float64_output_array(
            np.concatenate(tuple(segment.owned_transmission for segment in segments)),
            field_name="merged LBLRTM transmission",
        )
    except MemoryError as exc:
        raise LblrtmOutputError("cannot allocate the merged LBLRTM spectrum") from exc
    if np.any(np.diff(wavenumber) <= 0.0):
        raise LblrtmOutputError(
            "fixed-ownership merged LBLRTM wavenumbers are not strictly increasing"
        )
    semantic_sha256 = _compute_lblrtm_spectrum_semantic_sha256(
        wavenumber,
        transmission,
        coordinate_kind_and_unit="vacuum_wavenumber_cm-1",
        wavelength_medium_or_empty="",
        allow_nonphysical=any(
            np.any(segment.owned_transmission < 0.0)
            or np.any(segment.owned_transmission > _MAXIMUM_TRANSMISSION)
            or not np.all(np.isfinite(segment.owned_transmission))
            for segment in segments
        ),
    )
    return _LblrtmMergedSpectrum(
        wavenumber_cm1=wavenumber,
        transmission=transmission,
        semantic_sha256=semantic_sha256,
    )


def _convert_requested_output_grid_to_vacuum_wavenumber(
    output_wavelength_grid_nm: np.ndarray,
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
) -> np.ndarray:
    """Convert the increasing caller wavelength grid into vacuum wavenumber."""

    try:
        if wavelength_medium == "air":
            vacuum_wavelength = np.asarray(
                air_to_vac(
                    output_wavelength_grid_nm,
                    formula=wavelength_conversion_formula,
                ),
                dtype=np.float64,
            )
        else:
            vacuum_wavelength = np.asarray(
                output_wavelength_grid_nm,
                dtype=np.float64,
            )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            wavenumber = _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / vacuum_wavelength
        frozen = _freeze_lblrtm_float64_output_array(
            wavenumber,
            field_name="caller-grid vacuum wavenumber",
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
        MemoryError,
    ) as exc:
        raise LblrtmOutputError(
            "cannot convert the caller output grid to vacuum wavenumber"
        ) from exc
    if (
        frozen.shape != output_wavelength_grid_nm.shape
        or not np.all(np.isfinite(frozen))
        or np.any(frozen <= 0.0)
        or np.any(np.diff(frozen) >= 0.0)
    ):
        raise LblrtmOutputError(
            "caller output wavelengths must map to finite, positive, "
            "strictly decreasing vacuum wavenumbers"
        )
    return frozen


def _resample_multisegment_transmission_to_requested_grid(
    *,
    request: _NormalizedTransmissionRequest,
    segment_plan: _LblrtmMultisegmentPlan,
    segments: tuple[_LblrtmValidatedSpectralSegmentResult, ...],
) -> np.ndarray:
    """Interpolate every caller point only within its fixed owning run."""

    output_wavelength = request.output_wavelength_grid_nm
    if output_wavelength is None:  # pragma: no cover - the caller passes it
        raise AssertionError("caller output grid is unavailable")
    output_wavenumber = _convert_requested_output_grid_to_vacuum_wavenumber(
        output_wavelength,
        wavelength_medium=request.wavelength_medium,
        wavelength_conversion_formula=(request.wavelength_conversion_formula),
    )
    switches = np.array(
        [
            _f10p3_ticks_to_wavenumber_cm1(planned.core_window_ticks[1])
            for planned in segment_plan.segments[:-1]
        ],
        dtype=np.float64,
    )
    owner_indices = np.searchsorted(
        switches,
        output_wavenumber,
        side="right",
    )
    output_transmission = np.empty(
        output_wavenumber.shape,
        dtype=np.float64,
    )
    for owner_index, segment in enumerate(segments):
        try:
            owner_mask = owner_indices == owner_index
            if not np.any(owner_mask):
                continue
            owned_coordinates = output_wavenumber[owner_mask]
            core_lower = _f10p3_ticks_to_wavenumber_cm1(
                segment.planned_segment.core_window_ticks[0]
            )
            core_upper = _f10p3_ticks_to_wavenumber_cm1(
                segment.planned_segment.core_window_ticks[1]
            )
            if segment.planned_segment.core_upper_inclusive:
                inside_core = (owned_coordinates >= core_lower) & (owned_coordinates <= core_upper)
            else:
                inside_core = (owned_coordinates >= core_lower) & (owned_coordinates < core_upper)
            if not np.all(inside_core):
                raise LblrtmOutputError(
                    "caller output point does not have exactly one fixed segment owner"
                )
            if np.any(owned_coordinates < segment.wavenumber_cm1[0]) or np.any(
                owned_coordinates > segment.wavenumber_cm1[-1]
            ):
                raise LblrtmOutputError(
                    "caller output point lies outside its owner segment's "
                    "complete TAPE12 coverage; extrapolation is forbidden"
                )
            interpolated = np.interp(
                owned_coordinates,
                segment.wavenumber_cm1,
                segment.transmission,
                left=np.nan,
                right=np.nan,
            )
            if not np.all(np.isfinite(interpolated)):
                raise LblrtmOutputError(
                    "owner-only caller-grid interpolation produced NaN or infinity"
                )
            output_transmission[owner_mask] = interpolated
        except MemoryError as exc:
            output_error = LblrtmOutputError("cannot allocate owner-only caller-grid interpolation")
            _add_lblrtm_multisegment_failure_provenance_note(
                output_error,
                segment_plan=segment_plan,
                planned_segment=segment.planned_segment,
                resource_fingerprint=segment.resource_fingerprint,
                tape5_sha256=segment.tape5_sha256,
                tape6_sha256=segment.tape6_sha256,
                tape12_sha256=segment.tape12_sha256,
                completed_segment_count=len(segments),
            )
            raise output_error from exc
        except Exception as exc:
            _add_lblrtm_multisegment_failure_provenance_note(
                exc,
                segment_plan=segment_plan,
                planned_segment=segment.planned_segment,
                resource_fingerprint=segment.resource_fingerprint,
                tape5_sha256=segment.tape5_sha256,
                tape6_sha256=segment.tape6_sha256,
                tape12_sha256=segment.tape12_sha256,
                completed_segment_count=len(segments),
            )
            raise
    if np.any(owner_indices < 0) or np.any(owner_indices >= len(segments)):
        raise LblrtmOutputError("caller output point does not have exactly one fixed segment owner")
    if not np.all(np.isfinite(output_transmission)):
        raise LblrtmOutputError("owner-only caller-grid interpolation produced NaN or infinity")
    return _freeze_lblrtm_float64_output_array(
        output_transmission,
        field_name="caller-grid multisegment transmission",
    )


def _build_lblrtm_multisegment_metadata(
    request: _NormalizedTransmissionRequest,
    *,
    segment_plan: _LblrtmMultisegmentPlan,
    segments: tuple[_LblrtmValidatedSpectralSegmentResult, ...],
    seam_diagnostics: tuple[_LblrtmSeamDiagnostic, ...],
    physical_inputs: _LblrtmMultisegmentPhysicalInputSnapshot,
    merged_lblrtm_spectrum: _LblrtmMergedSpectrum,
    lblrtm_wavelength_nm: np.ndarray,
    output_wavelength_nm: np.ndarray,
    output_transmission: np.ndarray,
    output_grid_kind: str,
    resampling: str,
    allow_nonphysical: bool = False,
) -> dict[str, object]:
    """Build the record of a multi-segment run (format 3, without file paths)."""

    output_semantic_sha256 = _compute_lblrtm_spectrum_semantic_sha256(
        output_wavelength_nm,
        output_transmission,
        coordinate_kind_and_unit="wavelength_nm",
        wavelength_medium_or_empty=request.wavelength_medium,
        allow_nonphysical=allow_nonphysical,
    )
    segment_abis = tuple(segment.tape12_abi for segment in segments)
    tape12_abi: LblrtmTape12Abi | None = None
    if any(item is not None for item in segment_abis):
        if any(item is None for item in segment_abis):
            raise LblrtmOutputError("an LBLRTM segment did not report its TAPE12 binary layout")
        first_abi = segment_abis[0]
        if first_abi is None:  # pragma: no cover - guarded above
            raise AssertionError("internal error: empty set of TAPE12 binary layouts")
        if any(item != first_abi for item in segment_abis[1:]):
            raise LblrtmOutputError(
                "the LBLRTM segments wrote TAPE12 with different binary layouts"
            )
        tape12_abi = first_abi
    metadata: dict[str, object] = {
        "metadata_schema_version": _TRANSMISSION_METADATA_SCHEMA_VERSION,
        "backend": "gtellcorr_lblrtm_direct",
        "binary_source_product": _BINARY_SOURCE_PRODUCT,
        "resource_fingerprint": physical_inputs.resource_fingerprint,
        "resource_safe_wavenumber_range_cm1": (physical_inputs.resource_safe_wavenumber_range_cm1),
        "resource_line_margin_policy": "aer_lblrtm_v12.17_25_cm-1_v1",
        "input_wavelength_medium": request.wavelength_medium,
        "output_wavelength_medium": request.wavelength_medium,
        "wavelength_conversion_formula": (request.wavelength_conversion_formula),
        "requested_wavelength_range_nm": (request.requested_wavelength_range_nm),
        "wave_margin_nm": request.wave_margin_nm,
        "expanded_input_wavelength_range_nm": (request.expanded_wavelength_range_nm),
        "expanded_vacuum_wavelength_range_nm": (request.expanded_vacuum_wavelength_range_nm),
        "continuum_mode": physical_inputs.continuum_mode,
        "line_of_sight": {
            "airmass": physical_inputs.line_of_sight.airmass,
            "observer_altitude_km": (physical_inputs.line_of_sight.observer_altitude_km),
            "path_end_altitude_km": (physical_inputs.line_of_sight.path_end_altitude_km),
            "reference_latitude_deg": (physical_inputs.line_of_sight.reference_latitude_deg),
        },
        "molecule_relative_columns": dict(physical_inputs.molecule_relative_columns),
        "segmentation_policy": _SEGMENT_PLAN_POLICY_ID,
        "maximum_rendered_segment_width_cm1": float(_MAXIMUM_RENDERED_SEGMENT_WIDTH_CM1),
        "adjacent_rendered_overlap_cm1": float(_ADJACENT_RENDERED_SEGMENT_OVERLAP_CM1),
        "half_overlap_cm1": float(_HALF_RENDERED_SEGMENT_OVERLAP_CM1),
        "ownership_policy": _OWNERSHIP_POLICY_ID,
        "segment_plan_sha256": segment_plan.segment_plan_sha256,
        "output_grid_kind": output_grid_kind,
        "resampling": resampling,
        "unconvolved": True,
        "transmission_policy": _TRANSMISSION_POLICY_ID,
        "lblrtm_wavelength_range_nm": (
            float(lblrtm_wavelength_nm[0]),
            float(lblrtm_wavelength_nm[-1]),
        ),
        "output_wavelength_range_nm": (
            float(output_wavelength_nm[0]),
            float(output_wavelength_nm[-1]),
        ),
        "output_point_count": int(output_wavelength_nm.size),
        "segment_count": len(segments),
        "segments": tuple(
            {
                "segment_index": segment.planned_segment.segment_index,
                "core_wavenumber_range_cm1": tuple(
                    _f10p3_ticks_to_wavenumber_cm1(value)
                    for value in segment.planned_segment.core_window_ticks
                ),
                "core_upper_inclusive": (segment.planned_segment.core_upper_inclusive),
                "rendered_wavenumber_range_cm1": (
                    segment.spectral_window.minimum_wavenumber_cm1,
                    segment.spectral_window.maximum_wavenumber_cm1,
                ),
                "returned_wavenumber_range_cm1": (
                    float(segment.wavenumber_cm1[0]),
                    float(segment.wavenumber_cm1[-1]),
                ),
                "tape12_point_count": int(segment.wavenumber_cm1.size),
                "owned_point_count": int(segment.owned_wavenumber_cm1.size),
                "tape5_sha256": segment.tape5_sha256,
                "tape12_spectrum_semantic_sha256": (segment.tape12_spectrum_semantic_sha256),
                "resource_fingerprint": segment.resource_fingerprint,
                "tape6_completion_policy": _TAPE6_COMPLETION_POLICY_ID,
                "tape12_reader_policy": _TAPE12_READER_POLICY_ID,
            }
            for segment in segments
        ),
        "seam_diagnostics": tuple(
            {
                "lower_segment_index": diagnostic.lower_segment_index,
                "upper_segment_index": diagnostic.upper_segment_index,
                "source_switch_wavenumber_cm1": (diagnostic.source_switch_wavenumber_cm1),
                "planned_overlap_wavenumber_range_cm1": (
                    diagnostic.planned_overlap_wavenumber_range_cm1
                ),
                "left_bracketing_spacing_cm1": (diagnostic.left_bracketing_spacing_cm1),
                "right_bracketing_spacing_cm1": (diagnostic.right_bracketing_spacing_cm1),
                "selected_seam_gap_cm1": (diagnostic.selected_seam_gap_cm1),
                "structural_validation": "passed",
                "diagnostic_grid": {
                    "minimum_wavenumber_cm1": (diagnostic.diagnostic_minimum_wavenumber_cm1),
                    "maximum_wavenumber_cm1": (diagnostic.diagnostic_maximum_wavenumber_cm1),
                    "point_count": diagnostic.diagnostic_point_count,
                },
                "report_only_transmission_metrics": {
                    "rms": diagnostic.rms,
                    "p99_absolute": diagnostic.p99_absolute,
                    "maximum_absolute": diagnostic.maximum_absolute,
                },
            }
            for diagnostic in seam_diagnostics
        ),
        "merged_lblrtm_spectrum_semantic_sha256": (merged_lblrtm_spectrum.semantic_sha256),
        "output_spectrum_semantic_sha256": output_semantic_sha256,
    }
    if tape12_abi is not None:
        metadata["tape12_abi"] = _path_free_tape12_abi_metadata(tape12_abi)
    return metadata


def _calculate_lblrtm_multisegment_transmission(
    full_profile: Mapping[str, object],
    *,
    normalized_request: _NormalizedTransmissionRequest,
    segment_plan: _LblrtmMultisegmentPlan,
    line_of_sight: LblrtmLineOfSight,
    molecule_relative_columns: Mapping[str, float],
    resources: ResolvedLblrtmResourceFiles,
    resource_safe_wavenumber_range_cm1: tuple[float, float],
    continuum_mode: str,
    timeout_s: float | None,
    verbose: bool,
    quality_mode: str = "strict",
) -> LblrtmTransmissionSpectrum | TelluricTransmissionQualityResult:
    """Execute, structurally validate, and fixed-owner merge one wide request."""

    _validate_lblrtm_multisegment_plan(segment_plan)
    provisional_fingerprint = resources.manifest_fingerprint
    provenance_fingerprint = (
        provisional_fingerprint if isinstance(provisional_fingerprint, str) else "unavailable"
    )
    try:
        physical_inputs = _build_lblrtm_multisegment_physical_input_snapshot(
            full_profile,
            normalized_request=normalized_request,
            line_of_sight=line_of_sight,
            molecule_relative_columns=molecule_relative_columns,
            resources=resources,
            resource_safe_wavenumber_range_cm1=(resource_safe_wavenumber_range_cm1),
            continuum_mode=continuum_mode,
            timeout_s=timeout_s,
            verbose=verbose,
        )
    except Exception as exc:
        _add_lblrtm_multisegment_failure_provenance_note(
            exc,
            segment_plan=segment_plan,
            planned_segment=None,
            resource_fingerprint=provenance_fingerprint,
            completed_segment_count=0,
        )
        raise

    completed_segments: list[_LblrtmValidatedSpectralSegmentResult] = []
    seam_diagnostics: list[_LblrtmSeamDiagnostic] = []
    cumulative_owned_point_count = 0
    for planned_segment in segment_plan.segments:
        segment = _execute_and_validate_one_planned_lblrtm_segment(
            planned_segment=planned_segment,
            segment_plan=segment_plan,
            physical_inputs=physical_inputs,
            completed_segment_count=len(completed_segments),
            quality_mode=quality_mode,
        )
        cumulative_owned_point_count += int(segment.owned_wavenumber_cm1.size)
        if cumulative_owned_point_count > _MAXIMUM_OUTPUT_GRID_POINTS:
            point_limit_error = LblrtmOutputError(
                "merged LBLRTM point count exceeds the "
                f"{_MAXIMUM_OUTPUT_GRID_POINTS}-point project limit"
            )
            _add_lblrtm_multisegment_failure_provenance_note(
                point_limit_error,
                segment_plan=segment_plan,
                planned_segment=planned_segment,
                resource_fingerprint=physical_inputs.resource_fingerprint,
                tape5_sha256=segment.tape5_sha256,
                tape6_sha256=segment.tape6_sha256,
                tape12_sha256=segment.tape12_sha256,
                completed_segment_count=len(completed_segments),
            )
            raise point_limit_error
        if completed_segments:
            try:
                seam_diagnostic = _validate_lblrtm_segment_seam_structure(
                    completed_segments[-1],
                    segment,
                )
            except MemoryError as exc:
                output_error = LblrtmOutputError(
                    "cannot allocate memory while validating an LBLRTM seam"
                )
                _add_lblrtm_multisegment_failure_provenance_note(
                    output_error,
                    segment_plan=segment_plan,
                    planned_segment=planned_segment,
                    resource_fingerprint=(physical_inputs.resource_fingerprint),
                    tape5_sha256=segment.tape5_sha256,
                    tape6_sha256=segment.tape6_sha256,
                    tape12_sha256=segment.tape12_sha256,
                    completed_segment_count=len(completed_segments),
                )
                raise output_error from exc
            except Exception as exc:
                _add_lblrtm_multisegment_failure_provenance_note(
                    exc,
                    segment_plan=segment_plan,
                    planned_segment=planned_segment,
                    resource_fingerprint=(physical_inputs.resource_fingerprint),
                    tape5_sha256=segment.tape5_sha256,
                    tape6_sha256=segment.tape6_sha256,
                    tape12_sha256=segment.tape12_sha256,
                    completed_segment_count=len(completed_segments),
                )
                raise
            seam_diagnostics.append(seam_diagnostic)
        completed_segments.append(segment)

    frozen_segments = tuple(completed_segments)
    try:
        merged_lblrtm_spectrum = _merge_owned_lblrtm_segments(frozen_segments)
        (
            lblrtm_vacuum_wavelength_nm,
            lblrtm_transmission,
        ) = _convert_wavenumber_spectrum_to_increasing_vacuum_wavelength(
            merged_lblrtm_spectrum.wavenumber_cm1,
            merged_lblrtm_spectrum.transmission,
        )
        lblrtm_wavelength_nm = _convert_vacuum_wavelength_to_requested_medium(
            lblrtm_vacuum_wavelength_nm,
            wavelength_medium=normalized_request.wavelength_medium,
            wavelength_conversion_formula=(normalized_request.wavelength_conversion_formula),
        )
        _validate_lblrtm_wavelength_covers_expanded_input(
            lblrtm_wavelength_nm,
            expanded_input_wavelength_range_nm=(normalized_request.expanded_wavelength_range_nm),
        )
        _validate_transmission_spectrum(
            lblrtm_wavelength_nm,
            lblrtm_transmission,
            context="merged LBLRTM spectrum",
        ) if quality_mode == "strict" else None

        if normalized_request.output_wavelength_grid_nm is None:
            output_wavelength_nm = lblrtm_wavelength_nm
            output_transmission = lblrtm_transmission
            output_grid_kind = "merged_tape12_grid_converted"
            resampling = "none"
        else:
            output_wavelength_nm = normalized_request.output_wavelength_grid_nm
            output_transmission = _resample_multisegment_transmission_to_requested_grid(
                request=normalized_request,
                segment_plan=segment_plan,
                segments=frozen_segments,
            )
            output_grid_kind = "caller_supplied"
            resampling = "linear_within_owner_segment_coverage_v1"

        quality_flags = None
        fit_usable = None
        if quality_mode == "evaluate":
            merged_wavelength_flags = classify_transmission_values(lblrtm_transmission)
            merged_fit_usable = build_transmission_fit_mask(
                lblrtm_transmission,
                merged_wavelength_flags,
            )
            if normalized_request.output_wavelength_grid_nm is None:
                quality_flags = merged_wavelength_flags
                fit_usable = merged_fit_usable
            else:
                quality_flags, fit_usable = _resample_quality_to_requested_grid(
                    lblrtm_wavelength_nm,
                    merged_wavelength_flags,
                    merged_fit_usable,
                    output_wavelength_nm,
                )
        else:
            _validate_transmission_spectrum(
                output_wavelength_nm,
                output_transmission,
                context="final multisegment LBLRTM transmission spectrum",
            )
        metadata = _build_lblrtm_multisegment_metadata(
            normalized_request,
            segment_plan=segment_plan,
            segments=frozen_segments,
            seam_diagnostics=tuple(seam_diagnostics),
            physical_inputs=physical_inputs,
            merged_lblrtm_spectrum=merged_lblrtm_spectrum,
            lblrtm_wavelength_nm=lblrtm_wavelength_nm,
            output_wavelength_nm=output_wavelength_nm,
            output_transmission=output_transmission,
            output_grid_kind=output_grid_kind,
            resampling=resampling,
            allow_nonphysical=(quality_mode == "evaluate"),
        )
        try:
            if quality_mode == "evaluate":
                metadata = dict(metadata)
                metadata.update(
                    {
                        "quality_mode": "evaluate",
                        "raw_values_modified": False,
                        "fit_mask_policy_id": "transmission_mask_nonphysical_v1",
                    }
                )
                return TelluricTransmissionQualityResult(
                    wavelength_nm=output_wavelength_nm,
                    transmission=output_transmission,
                    wavelength_medium=normalized_request.wavelength_medium,
                    quality_flags=quality_flags,
                    fit_usable=fit_usable,
                    metadata=metadata,
                )
            return LblrtmTransmissionSpectrum(
                wavelength_nm=output_wavelength_nm,
                transmission=output_transmission,
                wavelength_medium=normalized_request.wavelength_medium,
                metadata=metadata,
            )
        except (LblrtmInputError, MemoryError) as exc:
            raise LblrtmOutputError(
                "internal error: cannot assemble the multi-segment LBLRTM transmission result"
            ) from exc
    except MemoryError as exc:
        output_error = LblrtmOutputError(
            "cannot allocate memory while processing the merged LBLRTM output"
        )
        _add_lblrtm_multisegment_failure_provenance_note(
            output_error,
            segment_plan=segment_plan,
            planned_segment=None,
            resource_fingerprint=physical_inputs.resource_fingerprint,
            completed_segment_count=len(frozen_segments),
        )
        raise output_error from exc
    except Exception as exc:
        _add_lblrtm_multisegment_failure_provenance_note(
            exc,
            segment_plan=segment_plan,
            planned_segment=None,
            resource_fingerprint=physical_inputs.resource_fingerprint,
            completed_segment_count=len(frozen_segments),
        )
        raise


def _calculate_one_lblrtm_spectral_segment(
    full_profile: Mapping[str, object],
    run_configuration: LblrtmRunConfiguration,
    *,
    molecule_relative_columns: Mapping[str, float],
    resources: ResolvedLblrtmResourceFiles,
    timeout_s: float | None,
    verbose: bool,
    quality_mode: str = "strict",
) -> _LblrtmSpectralSegmentResult:
    """Render, execute, and parse exactly one direct-RT spectral segment."""

    tape5_text = render_lblrtm_tape5(
        full_profile,
        run_configuration,
        molecule_relative_columns=molecule_relative_columns,
    )
    run_result = execute_lblrtm(
        tape5_text,
        resources,
        timeout_s=timeout_s,
        verbose=verbose,
    )
    try:
        if quality_mode == "evaluate":
            wavenumber_cm1, transmission, quality_flags, fit_usable = (
                read_lblrtm_tape12_transmission_quality(run_result.tape12_bytes)
            )
        else:
            wavenumber_cm1, transmission = read_lblrtm_tape12_transmission(run_result.tape12_bytes)
            quality_flags = None
            fit_usable = None
        tape12_abi = _validate_tape12_abi_against_resources(
            run_result.tape12_bytes,
            resources,
        )
    except (LblrtmOutputError, LblrtmResourceIntegrityError) as exc:
        _add_run_provenance_note(exc, run_result)
        raise
    return _LblrtmSpectralSegmentResult(
        spectral_window=run_configuration.spectral_window,
        wavenumber_cm1=wavenumber_cm1,
        transmission=transmission,
        run_result=run_result,
        quality_flags=quality_flags,
        fit_usable=fit_usable,
        tape12_abi=tape12_abi,
    )


def _validate_lblrtm_segment_matches_requested_window(
    segment: _LblrtmSpectralSegmentResult,
    *,
    requested_wavenumber_range_cm1: tuple[float, float],
) -> None:
    """Require actual TAPE12 samples to cover the raw physical request."""

    wavenumber = np.asarray(segment.wavenumber_cm1, dtype=np.float64)
    if wavenumber.ndim != 1 or wavenumber.size < 2:
        raise LblrtmOutputError("parsed TAPE12 segment must contain at least two wavenumbers")
    if not np.all(np.isfinite(wavenumber)):
        raise LblrtmOutputError("parsed TAPE12 segment contains non-finite wavenumbers")
    differences = np.diff(wavenumber)
    if np.any(differences <= 0.0):
        raise LblrtmOutputError("parsed TAPE12 segment wavenumbers must be strictly increasing")

    requested_minimum, requested_maximum = requested_wavenumber_range_cm1
    rendered_minimum = segment.spectral_window.minimum_wavenumber_cm1
    rendered_maximum = segment.spectral_window.maximum_wavenumber_cm1
    numerical_tolerance = _floating_coordinate_tolerance(
        requested_minimum,
        requested_maximum,
        rendered_minimum,
        rendered_maximum,
        float(wavenumber[0]),
        float(wavenumber[-1]),
    )
    if abs(float(wavenumber[0]) - rendered_minimum) > numerical_tolerance:
        raise LblrtmOutputError("TAPE12 first wavenumber does not match the rendered TAPE5 V1")
    final_spacing = float(differences[-1])
    final_grid_tolerance = 0.25 * final_spacing + numerical_tolerance
    if float(wavenumber[-1]) < rendered_maximum - final_grid_tolerance:
        raise LblrtmOutputError("TAPE12 final wavenumber falls short of the rendered TAPE5 V2")
    if float(wavenumber[-1]) > rendered_maximum + final_spacing + final_grid_tolerance:
        raise LblrtmOutputError(
            "TAPE12 final wavenumber extends more than one grid cell beyond the rendered TAPE5 V2"
        )
    if float(wavenumber[0]) > requested_minimum + numerical_tolerance:
        raise LblrtmOutputError("TAPE12 samples do not cover the requested lower wavenumber")
    if float(wavenumber[-1]) < requested_maximum - numerical_tolerance:
        raise LblrtmOutputError("TAPE12 samples do not cover the requested upper wavenumber")


def _convert_wavenumber_spectrum_to_increasing_vacuum_wavelength(
    wavenumber_cm1: np.ndarray,
    transmission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert increasing wavenumber to increasing vacuum nm as paired data."""

    try:
        wavenumber = np.asarray(wavenumber_cm1, dtype=np.float64)
        transmittance = np.asarray(transmission, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise LblrtmOutputError("cannot normalize the parsed LBLRTM spectrum") from exc
    if wavenumber.ndim != 1 or transmittance.ndim != 1 or wavenumber.shape != transmittance.shape:
        raise LblrtmOutputError(
            "LBLRTM wavenumber and transmission must be equal-length one-dimensional arrays"
        )
    if (
        wavenumber.size < 2
        or not np.all(np.isfinite(wavenumber))
        or np.any(wavenumber <= 0.0)
        or np.any(np.diff(wavenumber) <= 0.0)
    ):
        raise LblrtmOutputError(
            "LBLRTM wavenumber must be finite, positive, and strictly increasing"
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        decreasing_vacuum_wavelength = _WAVENUMBER_TO_WAVELENGTH_NM_CM1 / wavenumber
    vacuum_wavelength = np.array(
        decreasing_vacuum_wavelength[::-1],
        dtype=np.float64,
        copy=True,
    )
    increasing_wavelength_transmission = np.array(
        transmittance[::-1],
        dtype=np.float64,
        copy=True,
    )
    return vacuum_wavelength, increasing_wavelength_transmission


def _convert_vacuum_wavelength_to_requested_medium(
    vacuum_wavelength_nm: np.ndarray,
    *,
    wavelength_medium: str,
    wavelength_conversion_formula: str,
) -> np.ndarray:
    """Return vacuum wavelengths or explicitly label them in standard air."""

    try:
        if wavelength_medium == "vacuum":
            return np.array(
                vacuum_wavelength_nm,
                dtype=np.float64,
                copy=True,
            )
        return np.array(
            vac_to_air(
                vacuum_wavelength_nm,
                formula=wavelength_conversion_formula,
            ),
            dtype=np.float64,
            copy=True,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
        MemoryError,
    ) as exc:
        raise LblrtmOutputError(
            "cannot convert or copy the parsed LBLRTM vacuum wavelength grid "
            f"to {wavelength_medium} with formula "
            f"{wavelength_conversion_formula!r}"
        ) from exc


def _validate_lblrtm_wavelength_covers_expanded_input(
    lblrtm_wavelength_nm: np.ndarray,
    *,
    expanded_input_wavelength_range_nm: tuple[float, float],
) -> None:
    """Require final-medium LBLRTM samples to retain all padded support."""

    wavelength = np.asarray(lblrtm_wavelength_nm, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise LblrtmOutputError("converted LBLRTM wavelength grid must contain at least two points")
    if not np.all(np.isfinite(wavelength)):
        raise LblrtmOutputError("converted LBLRTM wavelength grid contains non-finite values")
    coordinate_tolerance = _floating_coordinate_tolerance(
        float(wavelength[0]),
        float(wavelength[-1]),
        *expanded_input_wavelength_range_nm,
    )
    if (
        float(wavelength[0]) > expanded_input_wavelength_range_nm[0] + coordinate_tolerance
        or float(wavelength[-1]) < expanded_input_wavelength_range_nm[1] - coordinate_tolerance
    ):
        raise LblrtmOutputError(
            "converted LBLRTM/TAPE12 grid does not cover the complete padded "
            "input wavelength interval"
        )


def _resample_transmission_to_requested_grid(
    source_wavelength_nm: np.ndarray,
    source_transmission: np.ndarray,
    output_wavelength_grid_nm: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate only inside actual parsed wavelength coverage."""

    if (
        output_wavelength_grid_nm[0] < source_wavelength_nm[0]
        or output_wavelength_grid_nm[-1] > source_wavelength_nm[-1]
    ):
        raise LblrtmOutputError(
            "actual TAPE12 wavelength coverage does not contain the complete "
            "requested output grid; extrapolation is forbidden"
        )
    try:
        interpolated = np.interp(
            output_wavelength_grid_nm,
            source_wavelength_nm,
            source_transmission,
            left=np.nan,
            right=np.nan,
        )
        return np.array(interpolated, dtype=np.float64, copy=True)
    except MemoryError as exc:
        raise LblrtmOutputError(
            "cannot allocate the requested interpolated transmission grid"
        ) from exc


def _resample_quality_to_requested_grid(
    source_wavelength_nm: np.ndarray,
    source_quality_flags: np.ndarray | None,
    source_fit_usable: np.ndarray | None,
    output_wavelength_grid_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map quality conservatively while transmission values are interpolated.

    The numerical transmission interpolation is retained for the caller's
    requested grid, but a point is fit-usable only when its exact TAPE12 sample
    is usable or both TAPE12 bracket samples are usable.  Quality is never
    inferred from an interpolated value alone.
    """

    if source_quality_flags is None or source_fit_usable is None:
        raise LblrtmOutputError("evaluate mode requires LBLRTM/TAPE12 quality arrays")
    source_wave = np.asarray(source_wavelength_nm, dtype=np.float64)
    flags = np.asarray(source_quality_flags, dtype=np.uint8)
    usable = np.asarray(source_fit_usable, dtype=bool)
    target = np.asarray(output_wavelength_grid_nm, dtype=np.float64)
    if (
        source_wave.ndim != 1
        or flags.shape != source_wave.shape
        or usable.shape != source_wave.shape
        or target.ndim != 1
    ):
        raise LblrtmOutputError("quality arrays must match the LBLRTM wavelength grid")
    right = np.searchsorted(source_wave, target, side="left")
    target_flags = np.full(target.size, int(TransmissionQualityFlag.NONFINITE), dtype=np.uint8)
    target_usable = np.zeros(target.size, dtype=bool)
    in_bounds = right < source_wave.size
    tolerance = 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(target))
    exact = in_bounds.copy()
    if np.any(exact):
        exact[exact] = np.isclose(
            source_wave[right[exact]],
            target[exact],
            rtol=0.0,
            atol=tolerance[exact],
        )
        exact_indices = np.flatnonzero(exact)
        target_flags[exact_indices] = flags[right[exact_indices]]
        target_usable[exact_indices] = usable[right[exact_indices]]
    bracketed = (~exact) & (right > 0) & (right < source_wave.size)
    if np.any(bracketed):
        indices = np.flatnonzero(bracketed)
        left = right[indices] - 1
        right_indices = right[indices]
        target_flags[indices] = np.where(
            usable[left] & usable[right_indices],
            int(TransmissionQualityFlag.PHYSICAL),
            int(TransmissionQualityFlag.NONFINITE),
        ).astype(np.uint8)
        target_usable[indices] = usable[left] & usable[right_indices]
    target_flags.setflags(write=False)
    target_usable.setflags(write=False)
    return target_flags, target_usable


def _validate_transmission_spectrum(
    wavelength_nm: np.ndarray,
    transmission: np.ndarray,
    *,
    context: str,
) -> None:
    """Validate transformed output without sorting, clipping, or repairing it."""

    try:
        wavelength = np.asarray(wavelength_nm, dtype=np.float64)
        transmittance = np.asarray(transmission, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise LblrtmOutputError(f"{context} cannot be represented as float64 arrays") from exc
    if wavelength.ndim != 1 or transmittance.ndim != 1:
        raise LblrtmOutputError(f"{context} arrays must be one-dimensional")
    if wavelength.shape != transmittance.shape:
        raise LblrtmOutputError(f"{context} wavelength and transmission shapes differ")
    if wavelength.size < 2:
        raise LblrtmOutputError(f"{context} must contain at least two samples")
    if (
        not np.all(np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise LblrtmOutputError(
            f"{context} wavelength must be finite, positive, and strictly increasing"
        )
    if not np.all(np.isfinite(transmittance)):
        raise LblrtmOutputError(f"{context} transmission contains NaN or infinity")
    if np.any(transmittance < 0.0) or np.any(transmittance > _MAXIMUM_TRANSMISSION):
        raise LblrtmOutputError(f"{context} transmission lies outside [0, {_MAXIMUM_TRANSMISSION}]")


def _build_transmission_metadata(
    request: _NormalizedTransmissionRequest,
    segment: _LblrtmSpectralSegmentResult,
    *,
    line_of_sight: LblrtmLineOfSight,
    molecule_relative_columns: Mapping[str, float],
    continuum_mode: str,
    resource_safe_wavenumber_range_cm1: tuple[float, float],
    lblrtm_wavelength_nm: np.ndarray,
    output_wavelength_nm: np.ndarray,
    output_grid_kind: str,
    resampling: str,
) -> dict[str, object]:
    """Build the record of the calculation inputs (the same on every computer)."""

    run_result = segment.run_result
    metadata: dict[str, object] = {
        "metadata_schema_version": _TRANSMISSION_METADATA_SCHEMA_VERSION,
        "backend": "gtellcorr_lblrtm_direct",
        "resource_fingerprint": run_result.resource_fingerprint,
        "resource_safe_wavenumber_range_cm1": (resource_safe_wavenumber_range_cm1),
        "resource_line_margin_policy": "aer_lblrtm_v12.17_25_cm-1_v1",
        "input_wavelength_medium": request.wavelength_medium,
        "output_wavelength_medium": request.wavelength_medium,
        "wavelength_conversion_formula": (request.wavelength_conversion_formula),
        "requested_wavelength_range_nm": (request.requested_wavelength_range_nm),
        "wave_margin_nm": request.wave_margin_nm,
        "expanded_input_wavelength_range_nm": (request.expanded_wavelength_range_nm),
        "expanded_vacuum_wavelength_range_nm": (request.expanded_vacuum_wavelength_range_nm),
        "continuum_mode": continuum_mode,
        "line_of_sight": {
            "airmass": line_of_sight.airmass,
            "observer_altitude_km": line_of_sight.observer_altitude_km,
            "path_end_altitude_km": line_of_sight.path_end_altitude_km,
            "reference_latitude_deg": line_of_sight.reference_latitude_deg,
        },
        "molecule_relative_columns": dict(molecule_relative_columns),
        "unconvolved": True,
        "transmission_policy": "strict_raw_no_clipping_v1",
        "output_grid_kind": output_grid_kind,
        "resampling": resampling,
        "lblrtm_wavelength_range_nm": (
            float(lblrtm_wavelength_nm[0]),
            float(lblrtm_wavelength_nm[-1]),
        ),
        "output_wavelength_range_nm": (
            float(output_wavelength_nm[0]),
            float(output_wavelength_nm[-1]),
        ),
        "output_point_count": int(output_wavelength_nm.size),
        "segment_count": 1,
        "segments": (
            {
                "raw_padded_wavenumber_range_cm1": (request.requested_wavenumber_range_cm1),
                "rendered_wavenumber_range_cm1": (
                    segment.spectral_window.minimum_wavenumber_cm1,
                    segment.spectral_window.maximum_wavenumber_cm1,
                ),
                "returned_wavenumber_range_cm1": (
                    float(segment.wavenumber_cm1[0]),
                    float(segment.wavenumber_cm1[-1]),
                ),
                "tape12_point_count": int(segment.wavenumber_cm1.size),
                "tape5_sha256": run_result.tape5_sha256,
                "tape6_completion_policy": "v12.17_footer_v1",
                "tape12_reader_policy": "aer_lblrtm_v12.17_direct_rt_v1",
            },
        ),
    }
    if segment.tape12_abi is not None:
        metadata["tape12_abi"] = _path_free_tape12_abi_metadata(segment.tape12_abi)
    return metadata


def _normalize_wavelength_vector(
    value: ArrayLike,
    *,
    field_name: str,
    required_size: int | None = None,
    minimum_size: int | None = None,
    maximum_size: int | None = None,
) -> np.ndarray:
    """Return one read-only, finite, positive, increasing float64 vector."""

    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise LblrtmInputError(f"{field_name} must be a numerical one-dimensional collection")
    for unit_attribute in ("unit", "units"):
        try:
            getattr(value, unit_attribute)
        except AttributeError:
            continue
        except Exception as exc:
            raise LblrtmInputError(f"cannot inspect {field_name} for physical units") from exc
        raise LblrtmInputError(
            f"{field_name} must contain explicit unitless values in nm; "
            f"unit-bearing inputs are not accepted"
        )
    try:
        size_hint = len(value)  # type: ignore[arg-type]
    except (TypeError, OverflowError):
        size_hint = None
    if size_hint is not None:
        if required_size is not None and size_hint != required_size:
            raise LblrtmInputError(f"{field_name} must contain exactly {required_size} values")
        if minimum_size is not None and size_hint < minimum_size:
            raise LblrtmInputError(f"{field_name} must contain at least {minimum_size} values")
        if maximum_size is not None and size_hint > maximum_size:
            raise LblrtmInputError(f"{field_name} exceeds the {maximum_size}-point project limit")
    if np.ma.isMaskedArray(value):
        try:
            mask = np.ma.getmask(value)
            has_masked_values = mask is not np.ma.nomask and bool(np.any(mask))
        except (TypeError, ValueError, MemoryError) as exc:
            raise LblrtmInputError(f"cannot inspect the mask on {field_name}") from exc
        if has_masked_values:
            raise LblrtmInputError(f"{field_name} must not contain masked or missing wavelengths")
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        source_array = value
        if source_array.dtype.kind not in {"i", "u", "f"}:
            raise LblrtmInputError(
                f"{field_name} must contain real numbers, not booleans or strings"
            )
    else:
        try:
            source_array = np.asarray(value, dtype=object)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise LblrtmInputError(
                f"{field_name} must be a numerical one-dimensional collection"
            ) from exc
        if source_array.ndim != 1:
            raise LblrtmInputError(f"{field_name} must be one-dimensional")
        if any(
            isinstance(
                item,
                (
                    bool,
                    np.bool_,
                    str,
                    bytes,
                    complex,
                    np.complexfloating,
                    np.datetime64,
                    np.timedelta64,
                ),
            )
            for item in source_array
        ):
            raise LblrtmInputError(
                f"{field_name} must contain real numbers, not booleans or strings"
            )
    if source_array.ndim != 1:
        raise LblrtmInputError(f"{field_name} must be one-dimensional")
    if required_size is not None and source_array.size != required_size:
        raise LblrtmInputError(f"{field_name} must contain exactly {required_size} values")
    if minimum_size is not None and source_array.size < minimum_size:
        raise LblrtmInputError(f"{field_name} must contain at least {minimum_size} values")
    if maximum_size is not None and source_array.size > maximum_size:
        raise LblrtmInputError(f"{field_name} exceeds the {maximum_size}-point project limit")
    try:
        numeric_array = np.array(
            source_array,
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise LblrtmInputError(f"{field_name} must contain only finite real numbers") from exc
    if not np.all(np.isfinite(numeric_array)):
        raise LblrtmInputError(f"{field_name} must contain only finite values")
    if np.any(numeric_array <= 0.0):
        raise LblrtmInputError(f"{field_name} must contain only positive wavelengths")
    if numeric_array.size > 1 and np.any(np.diff(numeric_array) <= 0.0):
        raise LblrtmInputError(f"{field_name} must be strictly increasing")
    try:
        immutable_bytes = numeric_array.tobytes(order="C")
        return np.frombuffer(
            immutable_bytes,
            dtype=np.float64,
        )
    except MemoryError as exc:
        raise LblrtmInputError(f"cannot allocate memory for a copy of {field_name}") from exc


def _floating_coordinate_tolerance(*values: float) -> float:
    """Return only a float64-representation-scale coordinate tolerance."""

    return 32.0 * np.finfo(np.float64).eps * max(1.0, *(abs(float(value)) for value in values))


def _add_exception_note(exception: BaseException, note: str) -> None:
    """Attach a secondary error on Python 3.10 and Python 3.11+."""

    add_note = getattr(exception, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    if exception.args and isinstance(exception.args[0], str):
        exception.args = (
            f"{exception.args[0]}\n{note}",
            *exception.args[1:],
        )
    else:
        exception.args = (*exception.args, note)


def _add_run_provenance_note(
    exception: BaseException,
    run_result: LblrtmRunResult,
) -> None:
    """Attach a description (without file paths) of each output rejected after the run."""

    _add_exception_note(
        exception,
        "LBLRTM provenance: "
        f"resource_fingerprint={run_result.resource_fingerprint}, "
        f"tape5_sha256={run_result.tape5_sha256}, "
        f"tape6_sha256={run_result.tape6_sha256}, "
        f"tape12_sha256={run_result.tape12_sha256}",
    )


def _coerce_nonnegative_finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    """Return one finite float greater than or equal to zero."""

    if isinstance(value, (bool, np.bool_)):
        raise LblrtmInputError(f"{field_name} must be a real number, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(f"{field_name} must be convertible to a finite real number") from exc
    if not math.isfinite(normalized):
        raise LblrtmInputError(f"{field_name} must be finite")
    if normalized < 0.0:
        raise LblrtmInputError(f"{field_name} must be >= 0")
    return 0.0 if normalized == 0.0 else normalized
