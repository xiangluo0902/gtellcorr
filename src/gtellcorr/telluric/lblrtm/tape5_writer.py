"""Write LBLRTM v12.17 spectral TAPE5 input (the same input always gives the same file).

The records written here follow the fixed-width ``READ``/``FORMAT`` statements
in the official AER LBLRTM v12.17 source:

* ``src/lblrtm.f90`` records 1.1--1.4;
* ``src/lblatm.f90`` records 3.1--3.6; and
* ``docs/html/lblrtm_instructions.html`` for the corresponding field
  definitions.

This module does not read the gtellcorr configuration.  It accepts an
already-complete user atmosphere and
renders ``MODEL=0``, ``IATM=1``, ``NMOL=12`` transmission or optical-depth
input.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import numpy as np

from .lblrtm_data_models import (
    LblrtmLineOfSight,
    LblrtmRunConfiguration,
    LblrtmSpectralWindow,
)
from .lblrtm_errors import LblrtmInputError
from .lblrtm_molecules import (
    LBLRTM_MOLECULES_BY_ID,
    LBLRTM_MOLECULES_BY_NAME,
    normalize_lblrtm_molecule_name,
    order_lblrtm_molecule_names,
)

__all__ = [
    "render_lblrtm_tape5",
    "validate_lblrtm_layer_input",
    "write_lblrtm_tape5",
]


_MAX_PROFILE_LEVELS: Final[int] = 7_000
_MAX_EXPLICIT_BOUNDARIES: Final[int] = 600
_MAX_SPECTRAL_WINDOW_WIDTH_CM1: Final[float] = 2_000.0
_PROFILE_TITLE: Final[str] = "GTELLCORR MODEL0 PROFILE"
_RUN_TITLE: Final[str] = "$ GTELLCORR LBLRTM DIRECT TRANSMISSION"
_EARTH_RADIUS_KM: Final[float] = 6_371.230
_PROFILE_MOLECULE_NAMES: Final[tuple[str, ...]] = tuple(
    LBLRTM_MOLECULES_BY_ID[molecule_id].canonical_name for molecule_id in range(1, 13)
)


def validate_lblrtm_layer_input(
    full_profile: Mapping[str, object],
    required_molecules: Iterable[str],
) -> None:
    """Validate one complete set of LBLRTM user-atmosphere level arrays.

    Parameters
    ----------
    full_profile
        Mapping containing ``z_km``, ``P_hPa``, ``T_K``, and one ``*_ppmv``
        one-dimensional array for every requested molecule.
    required_molecules
        Molecule names or registered aliases.  Validation is independent of
        caller order and uses the official LBLRTM molecule-ID order.

    Notes
    -----
    This function validates only.  The renderer repeats the same private
    checks on its own read-only copies, so it never formats arrays the caller
    could still change.
    """

    if not isinstance(full_profile, Mapping):
        raise LblrtmInputError("full_profile must be a mapping of level arrays")
    try:
        canonical_molecules = order_lblrtm_molecule_names(required_molecules)
    except (TypeError, ValueError) as exc:
        raise LblrtmInputError(f"required_molecules is invalid: {exc}") from exc

    altitude_km = _validate_vertical_grid(full_profile)
    _validate_pressure_temperature_layers(
        full_profile,
        expected_level_count=altitude_km.size,
    )
    _validate_molecule_layer_arrays(
        full_profile,
        canonical_molecules,
        expected_level_count=altitude_km.size,
    )


def render_lblrtm_tape5(
    full_profile: Mapping[str, object],
    run_configuration: LblrtmRunConfiguration,
    *,
    molecule_relative_columns: Mapping[str, float] | None = None,
) -> str:
    """Return one LBLRTM v12.17 TAPE5 (ASCII, LF line ends) as text.

    The supplied profile is copied before formatting.  Missing molecule
    scales default to one, and scaling is applied only to copied molecule
    arrays.  No caller-owned mapping or array is modified.
    """

    if not isinstance(full_profile, Mapping):
        raise LblrtmInputError("full_profile must be a mapping of level arrays")
    if not isinstance(run_configuration, LblrtmRunConfiguration):
        raise LblrtmInputError("run_configuration must be an LblrtmRunConfiguration")
    spectral_window = run_configuration.spectral_window
    if not isinstance(spectral_window, LblrtmSpectralWindow):
        raise LblrtmInputError("run_configuration.spectral_window must be an LblrtmSpectralWindow")
    if spectral_window.width_cm1 > _MAX_SPECTRAL_WINDOW_WIDTH_CM1:
        raise LblrtmInputError(
            "one LBLRTM request must span no more than "
            f"{_MAX_SPECTRAL_WINDOW_WIDTH_CM1:.0f} cm-1; got "
            f"{spectral_window.width_cm1!r} cm-1"
        )

    altitude_km = _validate_vertical_grid(full_profile)
    pressure_hpa, temperature_k = _validate_pressure_temperature_layers(
        full_profile,
        expected_level_count=altitude_km.size,
    )
    molecule_arrays = _validate_molecule_layer_arrays(
        full_profile,
        _PROFILE_MOLECULE_NAMES,
        expected_level_count=altitude_km.size,
    )
    scaled_molecule_arrays = _apply_molecule_relative_columns(
        molecule_arrays,
        molecule_relative_columns,
    )

    geometry = _calculate_lblrtm_slant_path_geometry(
        altitude_km,
        run_configuration.line_of_sight,
    )
    control_records = _format_lblrtm_control_records(run_configuration)
    spectral_range_record = _format_lblrtm_spectral_range_record(
        spectral_window,
        optical_depth_grid_spacing_cm1=(run_configuration.optical_depth_grid_spacing_cm1),
    )
    geometry_records = _format_lblrtm_geometry_records(
        altitude_km,
        spectral_window,
        geometry,
        reference_latitude_deg=(run_configuration.line_of_sight.reference_latitude_deg),
    )
    atmosphere_records = _format_lblrtm_atmospheric_layer_records(
        altitude_km,
        pressure_hpa,
        temperature_k,
        scaled_molecule_arrays,
    )

    records = [
        control_records[0],
        control_records[1],
        spectral_range_record,
    ]
    if control_records[2] is not None:
        records.append(control_records[2])
    records.extend(geometry_records)
    records.extend(atmosphere_records)
    records.append(
        _format_fixed_width_tape5_record(
            "%",
            width=80,
            record_name="termination record",
        )
    )
    rendered = "\n".join(records) + "\n"
    try:
        rendered.encode("ascii")
    except UnicodeEncodeError as exc:
        raise LblrtmInputError("rendered TAPE5 must contain ASCII only") from exc
    return rendered


def write_lblrtm_tape5(
    path: str | os.PathLike[str],
    full_profile: Mapping[str, object],
    run_configuration: LblrtmRunConfiguration,
    *,
    molecule_relative_columns: Mapping[str, float] | None = None,
) -> Path:
    """Write one TAPE5 file (ASCII, LF line ends)."""

    if isinstance(path, (bytes, bytearray, memoryview)):
        raise LblrtmInputError("path must be str or os.PathLike[str], not bytes")
    try:
        output_path = Path(path)
    except (TypeError, ValueError, OSError) as exc:
        raise LblrtmInputError("path must be convertible to a filesystem path") from exc

    rendered = render_lblrtm_tape5(
        full_profile,
        run_configuration,
        molecule_relative_columns=molecule_relative_columns,
    )
    payload = rendered.encode("ascii")
    temporary_path: Path | None = None
    try:
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "wb") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError(
                        f"short write while writing LBLRTM TAPE5: {written} of {len(payload)} bytes"
                    )
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(temporary_fd)
            raise
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as exc:
        raise LblrtmInputError(f"cannot write LBLRTM TAPE5 to {output_path!s}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return output_path


def _validate_vertical_grid(
    full_profile: Mapping[str, object],
) -> np.ndarray:
    """Return a checked, read-only copy of the altitudes in kilometres."""

    try:
        supplied = full_profile["z_km"]
    except KeyError as exc:
        raise LblrtmInputError("full_profile is missing required field 'z_km'") from exc
    try:
        object_array = np.asarray(supplied, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(
            "full_profile['z_km'] must be a one-dimensional numeric array"
        ) from exc
    if object_array.ndim != 1:
        raise LblrtmInputError("full_profile['z_km'] must be one-dimensional")
    if any(isinstance(value, (bool, np.bool_)) for value in object_array):
        raise LblrtmInputError("full_profile['z_km'] must not contain booleans")
    try:
        altitude_km = np.array(object_array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError("full_profile['z_km'] must contain only real numbers") from exc
    if not 2 <= altitude_km.size <= _MAX_PROFILE_LEVELS:
        raise LblrtmInputError(
            f"full_profile must contain between 2 and {_MAX_PROFILE_LEVELS} altitude levels"
        )
    if not np.all(np.isfinite(altitude_km)):
        raise LblrtmInputError("full_profile['z_km'] must contain finite values")
    if not np.all(np.diff(altitude_km) > 0.0):
        raise LblrtmInputError("full_profile['z_km'] must be strictly increasing")
    altitude_km[altitude_km == 0.0] = 0.0
    return np.frombuffer(altitude_km.tobytes(order="C"), dtype=np.float64)


def _validate_pressure_temperature_layers(
    full_profile: Mapping[str, object],
    *,
    expected_level_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return read-only copies of pressure and temperature."""

    validated_arrays: list[np.ndarray] = []
    for field_name in ("P_hPa", "T_K"):
        try:
            supplied = full_profile[field_name]
        except KeyError as exc:
            raise LblrtmInputError(
                f"full_profile is missing required field {field_name!r}"
            ) from exc
        try:
            object_array = np.asarray(supplied, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] must be a one-dimensional numeric array"
            ) from exc
        if object_array.ndim != 1:
            raise LblrtmInputError(f"full_profile[{field_name!r}] must be one-dimensional")
        if any(isinstance(value, (bool, np.bool_)) for value in object_array):
            raise LblrtmInputError(f"full_profile[{field_name!r}] must not contain booleans")
        try:
            numeric_array = np.array(
                object_array,
                dtype=np.float64,
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] must contain only real numbers"
            ) from exc
        if numeric_array.size != expected_level_count:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] has {numeric_array.size} "
                f"levels; expected {expected_level_count}"
            )
        if not np.all(np.isfinite(numeric_array)):
            raise LblrtmInputError(f"full_profile[{field_name!r}] must contain finite values")
        validated_arrays.append(np.frombuffer(numeric_array.tobytes(order="C"), dtype=np.float64))

    pressure_hpa, temperature_k = validated_arrays
    if not np.all(pressure_hpa > 0.0):
        raise LblrtmInputError("full_profile['P_hPa'] must be strictly positive")
    if not np.all(np.diff(pressure_hpa) < 0.0):
        raise LblrtmInputError("full_profile['P_hPa'] must strictly decrease with altitude")
    if not np.all(temperature_k > 0.0):
        raise LblrtmInputError("full_profile['T_K'] must be strictly positive")
    return pressure_hpa, temperature_k


def _validate_molecule_layer_arrays(
    full_profile: Mapping[str, object],
    required_molecules: Iterable[str],
    *,
    expected_level_count: int,
) -> dict[str, np.ndarray]:
    """Return copies of the molecule arrays in LBLRTM molecule-ID order."""

    try:
        canonical_molecules = order_lblrtm_molecule_names(required_molecules)
    except (TypeError, ValueError) as exc:
        raise LblrtmInputError(f"required_molecules is invalid: {exc}") from exc

    validated_arrays: dict[str, np.ndarray] = {}
    for molecule_name in canonical_molecules:
        definition = LBLRTM_MOLECULES_BY_NAME[molecule_name]
        field_name = definition.profile_field
        try:
            supplied = full_profile[field_name]
        except KeyError as exc:
            raise LblrtmInputError(
                f"full_profile is missing required molecule field {field_name!r}"
            ) from exc
        try:
            object_array = np.asarray(supplied, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] must be a one-dimensional numeric array"
            ) from exc
        if object_array.ndim != 1:
            raise LblrtmInputError(f"full_profile[{field_name!r}] must be one-dimensional")
        if any(isinstance(value, (bool, np.bool_)) for value in object_array):
            raise LblrtmInputError(f"full_profile[{field_name!r}] must not contain booleans")
        try:
            numeric_array = np.array(
                object_array,
                dtype=np.float64,
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] must contain only real numbers"
            ) from exc
        if numeric_array.size != expected_level_count:
            raise LblrtmInputError(
                f"full_profile[{field_name!r}] has {numeric_array.size} "
                f"levels; expected {expected_level_count}"
            )
        if not np.all(np.isfinite(numeric_array)):
            raise LblrtmInputError(f"full_profile[{field_name!r}] must contain finite values")
        if np.any(numeric_array < 0.0):
            raise LblrtmInputError(f"full_profile[{field_name!r}] must not contain negative VMR")
        validated_arrays[molecule_name] = np.frombuffer(
            numeric_array.tobytes(order="C"),
            dtype=np.float64,
        )
    return validated_arrays


def _apply_molecule_relative_columns(
    molecule_arrays: Mapping[str, np.ndarray],
    molecule_relative_columns: Mapping[str, float] | None,
) -> dict[str, np.ndarray]:
    """Return copies of the molecule arrays after uniform scaling."""

    if molecule_relative_columns is None:
        supplied_items: tuple[tuple[object, object], ...] = ()
    else:
        if not isinstance(molecule_relative_columns, Mapping):
            raise LblrtmInputError("molecule_relative_columns must be a mapping or None")
        try:
            supplied_items = tuple(molecule_relative_columns.items())
        except RuntimeError as exc:
            raise LblrtmInputError("molecule_relative_columns changed while being read") from exc

    scales: dict[str, float] = {}
    for supplied_name, supplied_scale in supplied_items:
        try:
            canonical_name = normalize_lblrtm_molecule_name(supplied_name)
        except (TypeError, ValueError) as exc:
            raise LblrtmInputError(
                f"invalid molecule_relative_columns key {supplied_name!r}: {exc}"
            ) from exc
        if canonical_name not in molecule_arrays:
            raise LblrtmInputError(
                "molecule_relative_columns contains a molecule not present "
                f"in the rendered profile: {canonical_name}"
            )
        if canonical_name in scales:
            raise LblrtmInputError(
                f"duplicate molecule_relative_columns key after normalization: {canonical_name}"
            )
        if isinstance(supplied_scale, (bool, np.bool_)):
            raise LblrtmInputError(f"relative-column scale for {canonical_name} must not be bool")
        try:
            scale = float(supplied_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LblrtmInputError(
                f"relative-column scale for {canonical_name} must be a real number"
            ) from exc
        if not math.isfinite(scale):
            raise LblrtmInputError(f"relative-column scale for {canonical_name} must be finite")
        if scale < 0.0:
            raise LblrtmInputError(f"relative-column scale for {canonical_name} must be >= 0")
        scales[canonical_name] = 0.0 if scale == 0.0 else scale

    scaled_arrays: dict[str, np.ndarray] = {}
    for molecule_name, molecule_array in molecule_arrays.items():
        scale = scales.get(molecule_name, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.array(molecule_array, dtype=np.float64, copy=True) * scale
        if not np.all(np.isfinite(scaled)):
            raise LblrtmInputError(
                f"relative-column scaling produced non-finite {molecule_name} VMR values"
            )
        if np.any(scaled < 0.0):
            raise LblrtmInputError(
                f"relative-column scaling produced negative {molecule_name} VMR values"
            )
        if scale > 0.0 and np.any((molecule_array > 0.0) & (scaled == 0.0)):
            raise LblrtmInputError(
                f"relative-column scaling underflowed positive {molecule_name} VMR values to zero"
            )
        scaled[scaled == 0.0] = 0.0
        scaled_arrays[molecule_name] = np.frombuffer(
            scaled.tobytes(order="C"),
            dtype=np.float64,
        )
    return scaled_arrays


def _airmass_to_zenith_angle_deg(airmass: float) -> float:
    """Convert plane-parallel secant airmass to the angle at observer H1."""

    if isinstance(airmass, (bool, np.bool_)):
        raise LblrtmInputError("airmass must be a real number, not bool")
    try:
        normalized_airmass = float(airmass)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError("airmass must be a real number") from exc
    if not math.isfinite(normalized_airmass) or normalized_airmass < 1.0:
        raise LblrtmInputError("airmass must be finite and >= 1")
    if normalized_airmass == 1.0:
        return 0.0
    angle_deg = math.degrees(math.acos(1.0 / normalized_airmass))
    if not math.isfinite(angle_deg) or not 0.0 <= angle_deg < 90.0:
        raise LblrtmInputError("airmass maps to an invalid LBLRTM upward-looking zenith angle")
    return angle_deg


def _calculate_lblrtm_slant_path_geometry(
    altitude_km: np.ndarray,
    line_of_sight: LblrtmLineOfSight,
) -> tuple[float, float, float, float, float, int, float]:
    """Resolve the ITYPE=2 H1-to-H2 geometry."""

    if not isinstance(line_of_sight, LblrtmLineOfSight):
        raise LblrtmInputError("run_configuration.line_of_sight must be an LblrtmLineOfSight")
    observer_altitude_km = (
        float(altitude_km[0])
        if line_of_sight.observer_altitude_km is None
        else float(line_of_sight.observer_altitude_km)
    )
    path_end_altitude_km = (
        float(altitude_km[-1])
        if line_of_sight.path_end_altitude_km is None
        else float(line_of_sight.path_end_altitude_km)
    )
    if observer_altitude_km == 0.0:
        observer_altitude_km = 0.0
    if path_end_altitude_km == 0.0:
        path_end_altitude_km = 0.0
    if not float(altitude_km[0]) <= observer_altitude_km <= float(altitude_km[-1]):
        raise LblrtmInputError("observer_altitude_km must lie within the full-profile grid")
    if not float(altitude_km[0]) <= path_end_altitude_km <= float(altitude_km[-1]):
        raise LblrtmInputError("path_end_altitude_km must lie within the full-profile grid")
    if path_end_altitude_km <= observer_altitude_km:
        raise LblrtmInputError("path_end_altitude_km must exceed observer_altitude_km")
    angle_deg = _airmass_to_zenith_angle_deg(line_of_sight.airmass)
    return (
        observer_altitude_km,
        path_end_altitude_km,
        angle_deg,
        0.0,
        0.0,
        0,
        0.0,
    )


def _format_lblrtm_control_records(
    run_configuration: LblrtmRunConfiguration,
) -> tuple[str, str, str | None]:
    """Format records 1.1/1.2 and the mode-dependent record 1.4."""

    continuum_flag = 1 if run_configuration.continuum_mode == "full" else 0
    iemit = 1 if run_configuration.spectral_output == "transmission" else 0
    iod = 1 if run_configuration.optical_depth_grid_spacing_cm1 is not None else 0
    record_1_1 = _format_fixed_width_tape5_record(
        _RUN_TITLE,
        width=80,
        record_name="record 1.1",
    )

    five_column_integers = tuple(
        f"{value:5d}"
        for value in (
            1,  # IHIRAC
            1,  # ILBLF4
            0,  # IAERSL
            iemit,  # IEMIT
            0,  # ISCAN
            0,  # IFILTR
            0,  # IPLOT
            0,  # ITEST
            1,  # IATM
            0,  # ILAS
            iod,  # IOD
            0,  # IXSECT
            0,  # ISOTPL
            0,  # IBRD
        )
    )
    continuum_fragment = f"{continuum_flag:1d}"
    irad_fragment = f"{0:1d}"
    mpts_fragment = f"{0:4d}"
    npts_fragment = f"{0:4d}"
    if (
        any(len(fragment) != 5 for fragment in five_column_integers)
        or len(continuum_fragment) != 1
        or len(irad_fragment) != 1
        or len(mpts_fragment) != 4
        or len(npts_fragment) != 4
    ):
        raise LblrtmInputError("a numeric field overflows its fixed width in record 1.2")
    (
        ihirac_fragment,
        ilblf4_fragment,
        iaersl_fragment,
        iemit_fragment,
        iscan_fragment,
        ifiltr_fragment,
        iplot_fragment,
        itest_fragment,
        iatm_fragment,
        ilas_fragment,
        iod_fragment,
        ixsect_fragment,
        isotpl_fragment,
        ibrd_fragment,
    ) = five_column_integers
    record_1_2_text = (
        ihirac_fragment
        + ilblf4_fragment
        + " 0"  # IFRGNX
        + "  "
        + continuum_fragment
        + iaersl_fragment
        + iemit_fragment
        + iscan_fragment
        + ifiltr_fragment
        + iplot_fragment
        + itest_fragment
        + iatm_fragment
        + "    0"  # 3X, CMRG=' 0'
        + ilas_fragment
        + iod_fragment
        + ixsect_fragment
        + irad_fragment
        + mpts_fragment
        + " "
        + npts_fragment
        + isotpl_fragment
        + ibrd_fragment
    )
    record_1_2 = _format_fixed_width_tape5_record(
        record_1_2_text,
        width=90,
        record_name="record 1.2",
    )

    record_1_4: str | None = None
    if iemit > 0:
        record_1_4_fragments = tuple(f"{0.0:10.3f}" for _ in range(7))
        if any(len(fragment) != 10 for fragment in record_1_4_fragments):
            raise LblrtmInputError("a numeric field overflows its fixed width in record 1.4")
        record_1_4_text = "".join(record_1_4_fragments) + "    s"
        record_1_4 = _format_fixed_width_tape5_record(
            record_1_4_text,
            width=75,
            record_name="record 1.4",
        )
    return record_1_1, record_1_2, record_1_4


def _format_lblrtm_spectral_range_record(
    spectral_window: LblrtmSpectralWindow,
    *,
    optical_depth_grid_spacing_cm1: float | None = None,
) -> str:
    """Format record 1.3 with explicit stable direct-RT defaults."""

    minimum = spectral_window.minimum_wavenumber_cm1
    maximum = spectral_window.maximum_wavenumber_cm1
    formatted_minimum = f"{minimum:10.3f}"
    formatted_maximum = f"{maximum:10.3f}"
    real_fragments = (
        formatted_minimum,
        formatted_maximum,
        f"{4.0:10.3f}",  # SAMPLE
        f"{0.0:10.3f}",  # DVSET
        f"{0.05:10.3f}",  # ALFAL0 for ILBLF4 line coupling
        f"{36.0:10.3f}",  # AVMASS
    )
    exponential_fragments = (
        f"{2.0e-4:10.3E}",  # DPTMIN
        f"{1.0e-3:10.3E}",  # DPTFAC
    )
    ilnflg_fragment = f"{0:1d}"
    dvout = 0.0 if optical_depth_grid_spacing_cm1 is None else float(optical_depth_grid_spacing_cm1)
    dvout_fragment = f"{dvout:10.3f}"
    nmol_scal_fragment = f"{0:2d}"
    n_xs_scal_fragment = f"{0:2d}"
    if (
        any(len(fragment) != 10 for fragment in real_fragments)
        or any(
            len(fragment) != 10
            or fragment[0] != " "
            or fragment[6] != "E"
            or fragment[7] not in "+-"
            or not fragment[8:10].isdigit()
            for fragment in exponential_fragments
        )
        or len(ilnflg_fragment) != 1
        or len(dvout_fragment) != 10
        or len(nmol_scal_fragment) != 2
        or len(n_xs_scal_fragment) != 2
    ):
        raise LblrtmInputError("a numeric field overflows its fixed width in record 1.3")
    if optical_depth_grid_spacing_cm1 is not None:
        try:
            rendered_dvout = float(dvout_fragment)
        except ValueError as exc:  # pragma: no cover - internal formatting
            raise LblrtmInputError("optical-depth output spacing cannot be rendered") from exc
        if rendered_dvout <= 0.0:
            raise LblrtmInputError(
                "record 1.3 F10.3 precision collapses the positive optical-depth output spacing"
            )
    try:
        parsed_minimum = float(formatted_minimum)
        parsed_maximum = float(formatted_maximum)
    except ValueError as exc:
        raise LblrtmInputError(
            "spectral-window bounds cannot be represented in record 1.3"
        ) from exc
    if parsed_minimum <= 0.0:
        raise LblrtmInputError(
            "record 1.3 precision rounds the positive minimum wavenumber to a non-positive value"
        )
    if not parsed_maximum > parsed_minimum:
        raise LblrtmInputError("record 1.3 precision collapses the requested spectral window")
    if parsed_maximum - parsed_minimum > _MAX_SPECTRAL_WINDOW_WIDTH_CM1:
        raise LblrtmInputError("formatted record 1.3 exceeds the 2000 cm-1 segment limit")

    record = (
        "".join((*real_fragments, *exponential_fragments))
        + " " * 4
        + ilnflg_fragment
        + " " * 5
        + dvout_fragment
        + " " * 3
        + nmol_scal_fragment
        + " " * 3
        + n_xs_scal_fragment
    )
    return _format_fixed_width_tape5_record(
        record,
        width=110,
        record_name="record 1.3",
    )


def _format_lblrtm_geometry_records(
    altitude_km: np.ndarray,
    spectral_window: LblrtmSpectralWindow,
    geometry: tuple[float, float, float, float, float, int, float],
    *,
    reference_latitude_deg: float,
) -> tuple[str, ...]:
    """Format records 3.1--3.4 using explicit profile boundaries."""

    level_count = int(altitude_km.size)
    h1, h2, angle, path_range, beta, path_length_flag, observer_height = geometry

    canonical_profile_fragments = tuple(f"{float(altitude):10.3f}" for altitude in altitude_km)
    if any(len(fragment) != 10 for fragment in canonical_profile_fragments):
        raise LblrtmInputError("a profile altitude overflows its F10.3 fixed width")
    try:
        canonical_profile_altitudes = np.array(
            [float(fragment) for fragment in canonical_profile_fragments],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise LblrtmInputError("profile altitudes cannot be written as F10.3 values") from exc
    canonical_profile_fragments = tuple(
        f"{0.0:10.3f}" if altitude == 0.0 else fragment
        for altitude, fragment in zip(
            canonical_profile_altitudes,
            canonical_profile_fragments,
            strict=False,
        )
    )
    canonical_profile_altitudes[canonical_profile_altitudes == 0.0] = 0.0
    if not np.all(np.isfinite(canonical_profile_altitudes)) or not np.all(
        np.diff(canonical_profile_altitudes) > 0.0
    ):
        raise LblrtmInputError(
            "at F10.3 precision the profile altitudes are no longer strictly increasing"
        )

    canonical_endpoint_fragments = (
        f"{h1:10.3f}",
        f"{h2:10.3f}",
    )
    if any(len(fragment) != 10 for fragment in canonical_endpoint_fragments):
        raise LblrtmInputError("a path endpoint overflows its F10.3 fixed width")
    try:
        canonical_h1, canonical_h2 = (float(fragment) for fragment in canonical_endpoint_fragments)
    except ValueError as exc:
        raise LblrtmInputError("path endpoints cannot be written as F10.3 values") from exc
    if canonical_h1 == 0.0:
        canonical_h1 = 0.0
    if canonical_h2 == 0.0:
        canonical_h2 = 0.0
    canonical_endpoint_fragments = (
        f"{canonical_h1:10.3f}",
        f"{canonical_h2:10.3f}",
    )
    if not (
        canonical_profile_altitudes[0] <= canonical_h1 <= canonical_profile_altitudes[-1]
        and canonical_profile_altitudes[0] <= canonical_h2 <= canonical_profile_altitudes[-1]
    ):
        raise LblrtmInputError("the path endpoints (F10.3) must lie within the profile altitudes")
    if canonical_h2 <= canonical_h1:
        raise LblrtmInputError("at F10.3 precision the H1-to-H2 path has zero length")

    formatted_geometry = tuple(
        f"{value:10.4f}"
        for value in (
            canonical_h1,
            canonical_h2,
            angle,
            path_range,
            beta,
        )
    )
    if any(len(fragment) != 10 for fragment in formatted_geometry):
        raise LblrtmInputError("a numeric field overflows its fixed width in record 3.2")
    try:
        displayed_h1, displayed_h2, parsed_angle, _, _ = (
            float(fragment) for fragment in formatted_geometry
        )
    except ValueError as exc:
        raise LblrtmInputError("path geometry cannot be represented in record 3.2") from exc
    if displayed_h1 != canonical_h1 or displayed_h2 != canonical_h2 or displayed_h2 <= displayed_h1:
        raise LblrtmInputError("record 3.2 does not reproduce the F10.3 H1-to-H2 path")
    if not 0.0 <= parsed_angle < 90.0:
        raise LblrtmInputError("record 3.2 precision produces an invalid zenith angle")

    interior_indices = np.flatnonzero(
        (canonical_profile_altitudes > canonical_h1) & (canonical_profile_altitudes < canonical_h2)
    )
    formatted_boundaries = (
        canonical_endpoint_fragments[0],
        *(canonical_profile_fragments[int(index)] for index in interior_indices),
        canonical_endpoint_fragments[1],
    )
    parsed_boundaries = np.array(
        (
            canonical_h1,
            *(float(canonical_profile_altitudes[int(index)]) for index in interior_indices),
            canonical_h2,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(parsed_boundaries)) or not np.all(np.diff(parsed_boundaries) > 0.0):
        raise LblrtmInputError(
            "record 3.3B precision does not preserve strictly increasing path boundaries"
        )
    boundary_count = len(formatted_boundaries)
    if boundary_count > _MAX_EXPLICIT_BOUNDARIES:
        raise LblrtmInputError(
            "explicit Record 3.3B path requires "
            f"{boundary_count} boundaries; LBLRTM v12.17 permits at most "
            f"{_MAX_EXPLICIT_BOUNDARIES}, and the writer will not thin them"
        )

    midpoint_wavenumber = (
        spectral_window.minimum_wavenumber_cm1 + spectral_window.maximum_wavenumber_cm1
    ) / 2.0
    record_3_1_integer_fragments = tuple(
        f"{value:5d}"
        for value in (
            0,  # MODEL
            2,  # ITYPE
            boundary_count,  # IBMAX
            0,  # ZERO
            1,  # NOPRNT
            12,  # NMOL
            0,  # IPUNCH
        )
    )
    ifxtyp_fragment = f"{0:2d}"
    munits_fragment = f"{0:2d}"
    record_3_1_real_fragments = (
        f"{_EARTH_RADIUS_KM:10.3f}",
        canonical_profile_fragments[-1],  # HSPACE
        f"{midpoint_wavenumber:10.3f}",  # VBAR
        f"{0.0:10.3f}",  # retired CO2MX/dummy
        f"{reference_latitude_deg:10.3f}",
    )
    if (
        any(len(fragment) != 5 for fragment in record_3_1_integer_fragments)
        or len(ifxtyp_fragment) != 2
        or len(munits_fragment) != 2
        or any(len(fragment) != 10 for fragment in record_3_1_real_fragments)
    ):
        raise LblrtmInputError("a numeric field overflows its fixed width in record 3.1")
    record_3_1_text = (
        "".join(record_3_1_integer_fragments)
        + ifxtyp_fragment
        + " "
        + munits_fragment
        + "".join(record_3_1_real_fragments)
    )
    record_3_1 = _format_fixed_width_tape5_record(
        record_3_1_text,
        width=90,
        record_name="record 3.1",
    )

    path_length_flag_fragment = f"{path_length_flag:5d}"
    observer_height_fragment = f"{observer_height:10.4f}"
    if len(path_length_flag_fragment) != 5 or len(observer_height_fragment) != 10:
        raise LblrtmInputError("a numeric field overflows its fixed width in record 3.2")
    record_3_2_text = (
        "".join(formatted_geometry) + path_length_flag_fragment + " " * 5 + observer_height_fragment
    )
    record_3_2 = _format_fixed_width_tape5_record(
        record_3_2_text,
        width=70,
        record_name="record 3.2",
    )

    boundary_records = tuple(
        _format_fixed_width_tape5_record(
            "".join(formatted_boundaries[start : start + 8]),
            width=80,
            record_name=f"record 3.3B[{start // 8 + 1}]",
        )
        for start in range(0, boundary_count, 8)
    )

    level_count_fragment = f"{level_count:5d}"
    if len(level_count_fragment) != 5:
        raise LblrtmInputError("the profile level count overflows its fixed width in record 3.4")
    record_3_4 = _format_fixed_width_tape5_record(
        level_count_fragment + _PROFILE_TITLE,
        width=29,
        record_name="record 3.4",
    )
    return record_3_1, record_3_2, *boundary_records, record_3_4


def _format_lblrtm_atmospheric_layer_records(
    altitude_km: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    molecule_arrays: Mapping[str, np.ndarray],
) -> tuple[str, ...]:
    """Format repeated records 3.5 and 3.6 for every profile level."""

    records: list[str] = []
    unit_characters = "A" * 12 + " " * 35
    altitude_fragments = tuple(f"{float(altitude):10.3f}" for altitude in altitude_km)
    pressure_fragments = tuple(f"{float(pressure):10.3E}" for pressure in pressure_hpa)
    temperature_fragments = tuple(f"{float(temperature):10.3f}" for temperature in temperature_k)
    if (
        any(len(fragment) != 10 for fragment in altitude_fragments)
        or any(
            len(fragment) != 10
            or fragment[0] != " "
            or fragment[6] != "E"
            or fragment[7] not in "+-"
            or not fragment[8:10].isdigit()
            for fragment in pressure_fragments
        )
        or any(len(fragment) != 10 for fragment in temperature_fragments)
    ):
        raise LblrtmInputError("a numeric field overflows its fixed width in record 3.5")
    try:
        parsed_altitudes = np.array(
            [float(fragment) for fragment in altitude_fragments],
            dtype=np.float64,
        )
        parsed_pressures = np.array(
            [float(fragment) for fragment in pressure_fragments],
            dtype=np.float64,
        )
        parsed_temperatures = np.array(
            [float(fragment) for fragment in temperature_fragments],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise LblrtmInputError("profile values cannot be represented in record 3.5") from exc
    altitude_fragments = tuple(
        f"{0.0:10.3f}" if altitude == 0.0 else fragment
        for altitude, fragment in zip(parsed_altitudes, altitude_fragments, strict=False)
    )
    parsed_altitudes[parsed_altitudes == 0.0] = 0.0
    if not np.all(np.isfinite(parsed_altitudes)) or not np.all(np.diff(parsed_altitudes) > 0.0):
        raise LblrtmInputError(
            "record 3.5 precision does not preserve strictly increasing profile altitudes"
        )
    if (
        not np.all(np.isfinite(parsed_pressures))
        or not np.all(parsed_pressures > 0.0)
        or not np.all(np.diff(parsed_pressures) < 0.0)
    ):
        raise LblrtmInputError(
            "record 3.5 precision does not preserve strictly decreasing positive profile pressures"
        )
    if not np.all(np.isfinite(parsed_temperatures)) or not np.all(parsed_temperatures > 0.0):
        raise LblrtmInputError("record 3.5 cannot represent positive finite profile temperatures")

    for level_index in range(altitude_km.size):
        record_3_5_text = (
            altitude_fragments[level_index]
            + pressure_fragments[level_index]
            + temperature_fragments[level_index]
            + " " * 5
            + "AA"
            + " "
            + "L"
            + " "
            + unit_characters
        )
        records.append(
            _format_fixed_width_tape5_record(
                record_3_5_text,
                width=87,
                record_name=f"record 3.5[{level_index + 1}]",
            )
        )

        level_values = tuple(
            float(molecule_arrays[molecule_name][level_index])
            for molecule_name in _PROFILE_MOLECULE_NAMES
        )
        value_fragments = tuple(f"{value:15.8E}" for value in level_values)
        if any(
            len(fragment) != 15
            or fragment[0] != " "
            or fragment[11] != "E"
            or fragment[12] not in "+-"
            or not fragment[13:15].isdigit()
            for fragment in value_fragments
        ):
            raise LblrtmInputError(
                "a molecule VMR field overflows its fixed width in "
                f"record 3.6 at profile level {level_index}"
            )
        try:
            parsed_values = np.array(
                [float(fragment) for fragment in value_fragments],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise LblrtmInputError(
                f"profile level {level_index} molecule VMR cannot be represented in record 3.6"
            ) from exc
        if not np.all(np.isfinite(parsed_values)) or np.any(parsed_values < 0.0):
            raise LblrtmInputError(f"record 3.6 cannot represent profile level {level_index}")
        supplied_values = np.array(level_values, dtype=np.float64)
        if np.any((supplied_values > 0.0) & (parsed_values == 0.0)):
            raise LblrtmInputError(
                "record 3.6 precision underflows a positive molecule VMR "
                f"to zero at profile level {level_index}"
            )
        for record_index, start in enumerate(range(0, 12, 8), start=1):
            records.append(
                _format_fixed_width_tape5_record(
                    "".join(value_fragments[start : start + 8]),
                    width=120,
                    record_name=(f"record 3.6[{level_index + 1}.{record_index}]"),
                )
            )
    return tuple(records)


def _format_fixed_width_tape5_record(
    record: str,
    *,
    width: int,
    record_name: str,
) -> str:
    """Validate one ASCII record and right-pad it without truncation."""

    if not isinstance(record, str):
        raise LblrtmInputError(f"{record_name} must be rendered as text")
    if "\n" in record or "\r" in record:
        raise LblrtmInputError(f"{record_name} must not contain line breaks")
    try:
        record.encode("ascii")
    except UnicodeEncodeError as exc:
        raise LblrtmInputError(f"{record_name} must contain ASCII only") from exc
    if len(record) > width:
        raise LblrtmInputError(
            f"{record_name} requires {len(record)} columns but its fixed "
            f"width is {width}; refusing to truncate"
        )
    return record.ljust(width)
