"""Read LBLRTM v12.17 transmission or optical depth from TAPE12.

The format read by this module is taken from the AER LBLRTM
v12.17 source:

* ``README.md`` describes a file-header record followed by repeating
  panel-header, radiance, and transmittance records.
* ``src/xmerge.f90`` routine ``EMOUT`` writes those three panel records.
* ``src/lblrtm.f90`` distinguishes the ``IEMIT=1`` direct-RT layout from
  the ``IEMIT=0`` optical-depth layout and defines the six-``-99`` endfile
  record.

The strict public readers return the raw increasing wavenumber grid and either
the IEMIT=1 transmittance or the IEMIT=0 sight-line optical depth. A separate
diagnostic-only transmission reader can preserve finite values outside the
physical transmission interval and report their locations; neither diagnostic
output nor its API is a scientific acceptance path. Wavelength conversion and
construction of higher-level spectrum data models belong to calculators.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

import numpy as np

from .lblrtm_errors import LblrtmInputError, LblrtmOutputError
from .transmission_quality import (
    build_transmission_fit_mask,
    classify_transmission_values,
)

__all__ = [
    "LblrtmTape12Abi",
    "LblrtmTape12DiagnosticResult",
    "inspect_lblrtm_tape12_abi",
    "read_lblrtm_tape12_optical_depth",
    "read_lblrtm_tape12_transmission",
    "read_lblrtm_tape12_transmission_diagnostic",
    "read_lblrtm_tape12_transmission_quality",
]


_FILE_HEADER_MAGIC: Final[bytes] = b" LBLRTM "
_FILE_HEADER_MAGIC_OFFSET: Final[int] = 72
_MAX_PANEL_POINTS: Final[int] = 2_410
# Project safety budgets below are not AER format maxima.
_MAX_PANEL_COUNT: Final[int] = 100_000
_MAX_TOTAL_POINTS: Final[int] = 5_000_000
_MAX_TAPE12_BYTES: Final[int] = 512 * 1024 * 1024
_MAX_RECORD_BYTES: Final[int] = 1024 * 1024
_TRANSMISSION_UPPER_BOUND: Final[float] = 1.00001


@dataclass(frozen=True, slots=True)
class LblrtmTape12Abi:
    """Detected path-free ABI of one LBLRTM TAPE12 binary."""

    byte_order: str
    record_marker_bytes: int
    real_bytes: int
    integer_bytes: int

    def __post_init__(self) -> None:
        if self.byte_order not in {"little", "big"}:
            raise ValueError("byte_order must be either 'little' or 'big'")
        if self.record_marker_bytes not in {4, 8}:
            raise ValueError("record_marker_bytes must be either 4 or 8")
        if self.real_bytes not in {4, 8}:
            raise ValueError("real_bytes must be either 4 or 8")
        if self.integer_bytes not in {4, 8}:
            raise ValueError("integer_bytes must be either 4 or 8")


@dataclass(frozen=True, slots=True)
class _Tape12BinaryLayout:
    """One supported sequential-unformatted ABI candidate."""

    byte_order: str
    record_marker_bytes: int
    real_bytes: int

    @property
    def integer_bytes(self) -> int:
        """Return the integer width of a supported v12.17 build layout."""

        return self.real_bytes

    @property
    def struct_prefix(self) -> str:
        """``struct`` byte-order prefix."""
        return "<" if self.byte_order == "little" else ">"

    @property
    def record_marker_format(self) -> str:
        """``struct`` code of a Fortran record marker."""
        return "i" if self.record_marker_bytes == 4 else "q"

    @property
    def integer_format(self) -> str:
        """``struct`` code of a Fortran integer."""
        return "i" if self.integer_bytes == 4 else "q"

    @property
    def file_header_bytes(self) -> int:
        """Size of the file-header record [bytes]."""
        return 1_056 if self.real_bytes == 4 else 1_416

    @property
    def iemit_offset(self) -> int:
        """Offset of IEMIT in the file header [bytes]."""
        return 912 if self.real_bytes == 4 else 1_208

    @property
    def file_header_spacing_offset(self) -> int:
        """Offset of the wavenumber spacing in the file header [bytes]."""
        return 868 if self.real_bytes == 4 else 1_136

    @property
    def file_header_first_wavenumber_offset(self) -> int:
        """Offset of the first wavenumber in the file header [bytes]."""
        return 872 if self.real_bytes == 4 else 1_144

    @property
    def file_header_last_wavenumber_offset(self) -> int:
        """Offset of the last wavenumber in the file header [bytes]."""
        return 880 if self.real_bytes == 4 else 1_152

    @property
    def panel_header_format(self) -> str:
        """``struct`` format of a panel header (first, last, spacing, point count)."""
        return "ddfi" if self.real_bytes == 4 else "dddq"

    @property
    def panel_header_bytes(self) -> int:
        """Size of a panel header [bytes]."""
        return 24 if self.real_bytes == 4 else 32

    @property
    def value_dtype(self) -> np.dtype:
        """NumPy dtype of the spectral values."""
        code = "f4" if self.real_bytes == 4 else "f8"
        return np.dtype(self.struct_prefix + code)

    @property
    def real_epsilon(self) -> float:
        """Machine epsilon of the real type."""
        dtype = np.float32 if self.real_bytes == 4 else np.float64
        return float(np.finfo(dtype).eps)

    def grid_tolerance(
        self,
        *,
        first: float,
        last: float,
        spacing: float,
        point_count: int,
    ) -> float:
        """Return a precision-aware absolute tolerance below half a grid cell."""

        span = abs(spacing) * max(point_count - 1, 1)
        endpoint_roundoff = np.finfo(np.float64).eps * max(1.0, abs(first), abs(last))
        spacing_roundoff = self.real_epsilon * max(1.0, span)
        raw_tolerance = 8.0 * (endpoint_roundoff + spacing_roundoff)
        return min(raw_tolerance, 0.25 * abs(spacing))


@dataclass(frozen=True, slots=True)
class _Tape12FileHeader:
    """Subset of the v12.17 file header needed by the reader."""

    iemit: int
    spacing_cm1: float
    first_wavenumber_cm1: float
    last_requested_wavenumber_cm1: float


@dataclass(frozen=True, slots=True)
class _Tape12PanelHeader:
    """Decoded panel-grid declaration."""

    first_wavenumber_cm1: float
    last_wavenumber_cm1: float
    spacing_cm1: float
    point_count: int


@dataclass(frozen=True, slots=True)
class _Tape12Panel:
    """One validated panel before overlap removal."""

    header: _Tape12PanelHeader
    wavenumber_cm1: np.ndarray
    radiance: np.ndarray | None
    transmission: np.ndarray | None
    optical_depth: np.ndarray | None


@dataclass(frozen=True, slots=True)
class LblrtmTape12DiagnosticResult:
    """Raw finite TAPE12 transmission plus non-physical-value diagnostics.

    This result is diagnostic-only.  It must not be passed to the scientific
    calculators or the fit.  The strict public reader remains the only
    acceptance path for production spectra.
    """

    wavenumber_cm1: np.ndarray
    transmission: np.ndarray
    panel_count: int
    nonphysical_count: int
    nonphysical_panel_numbers: tuple[int, ...]
    minimum_transmission: float
    maximum_transmission: float
    first_nonphysical_wavenumber_cm1: float | None
    last_nonphysical_wavenumber_cm1: float | None

    def __post_init__(self) -> None:
        if self.wavenumber_cm1.ndim != 1 or self.transmission.ndim != 1:
            raise LblrtmOutputError("diagnostic TAPE12 arrays must be one-dimensional")
        if self.wavenumber_cm1.shape != self.transmission.shape:
            raise LblrtmOutputError("diagnostic TAPE12 arrays have different lengths")
        if self.panel_count < 1 or self.nonphysical_count < 0:
            raise LblrtmOutputError("diagnostic TAPE12 counts are invalid")
        if not self.nonphysical_count:
            if self.first_nonphysical_wavenumber_cm1 is not None:
                raise LblrtmOutputError("empty diagnostic nonphysical set has a first coordinate")
            if self.last_nonphysical_wavenumber_cm1 is not None:
                raise LblrtmOutputError("empty diagnostic nonphysical set has a last coordinate")
        self.wavenumber_cm1.setflags(write=False)
        self.transmission.setflags(write=False)


def _public_tape12_abi(layout: _Tape12BinaryLayout) -> LblrtmTape12Abi:
    """Return a read-only public record of one detected internal layout."""

    return LblrtmTape12Abi(
        byte_order=layout.byte_order,
        record_marker_bytes=layout.record_marker_bytes,
        real_bytes=layout.real_bytes,
        integer_bytes=layout.integer_bytes,
    )


def inspect_lblrtm_tape12_abi(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> LblrtmTape12Abi:
    """Return the detected binary ABI without applying transmission bounds.

    This inspection validates enough of the sequential-unformatted structure
    to identify a unique supported v12.17 layout.  It deliberately does not
    classify or accept the physical spectral values; those checks remain in
    the mode-specific spectrum readers.
    """

    payload = _read_tape12_source_bytes(source)
    return _public_tape12_abi(_detect_fortran_binary_layout(payload))


def read_lblrtm_tape12_optical_depth(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return increasing wavenumber and raw sight-line optical depth.

    The input must use the LBLRTM v12.17 IEMIT=0 TAPE12 layout: one panel
    header followed by one optical-depth record per panel. Values are returned
    unchanged apart from conversion to read-only host-endian float64 arrays.
    NaN and infinity are rejected; finite negative optical depths are retained
    for explicit numerical diagnosis rather than silently clipped.
    """

    return _read_tape12_optical_depth_impl(source)


def read_lblrtm_tape12_transmission(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return increasing wavenumber and transmission arrays from TAPE12.

    Parameters
    ----------
    source
        Either the complete TAPE12 contents as a bytes-like object or a path.
        Strings and path-like objects are always interpreted as paths.

    Returns
    -------
    tuple of numpy.ndarray
        ``(wavenumber_cm1, transmission)`` as one-dimensional, read-only,
        host-endian ``float64`` arrays.

    Raises
    ------
    LblrtmInputError
        If ``source`` is not one of the documented source types.
    LblrtmOutputError
        If the output is unavailable, unsupported, malformed, or internally
        inconsistent.
    """

    return _read_tape12_transmission_impl(source, permit_nonphysical=False)[0]


def read_lblrtm_tape12_transmission_diagnostic(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> LblrtmTape12DiagnosticResult:
    """Read a complete finite TAPE12 spectrum for diagnosis only.

    Unlike :func:`read_lblrtm_tape12_transmission`, this path preserves finite
    transmission values outside ``[0, 1.00001]`` and reports their locations.
    It still rejects malformed records, inconsistent grids, NaN/infinite
    values, and unsafe sizes.  The result is not a scientific acceptance
    result and is deliberately not re-exported by the package.
    """

    spectrum, panel_numbers = _read_tape12_transmission_impl(
        source,
        permit_nonphysical=True,
    )
    wavenumber_cm1, transmission = spectrum
    bad = (transmission < 0.0) | (transmission > _TRANSMISSION_UPPER_BOUND)
    bad_wavenumbers = wavenumber_cm1[bad]
    return LblrtmTape12DiagnosticResult(
        wavenumber_cm1=wavenumber_cm1,
        transmission=transmission,
        panel_count=panel_numbers[0],
        nonphysical_count=int(np.count_nonzero(bad)),
        nonphysical_panel_numbers=panel_numbers[1],
        minimum_transmission=float(np.min(transmission)),
        maximum_transmission=float(np.max(transmission)),
        first_nonphysical_wavenumber_cm1=(
            float(bad_wavenumbers[0]) if bad_wavenumbers.size else None
        ),
        last_nonphysical_wavenumber_cm1=(
            float(bad_wavenumbers[-1]) if bad_wavenumbers.size else None
        ),
    )


def read_lblrtm_tape12_transmission_quality(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw TAPE12 arrays plus non-destructive quality flags and mask.

    This path prepares fitting input; it never repairs values.  Transmission
    values are byte-decoded and returned unchanged; callers must use the
    returned ``fit_usable`` mask to exclude zero and nonphysical samples.
    """

    diagnostic = read_lblrtm_tape12_transmission_diagnostic(source)
    flags = classify_transmission_values(diagnostic.transmission)
    fit_usable = build_transmission_fit_mask(
        diagnostic.transmission,
        flags,
    )
    return (
        diagnostic.wavenumber_cm1,
        diagnostic.transmission,
        flags,
        fit_usable,
    )


def _read_tape12_optical_depth_impl(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one strict IEMIT=0 optical-depth TAPE12."""

    payload = _read_tape12_source_bytes(source)
    layout = _detect_fortran_binary_layout(payload)
    stream = io.BytesIO(payload)

    file_header_record = _read_fortran_unformatted_record(stream, layout, record_name="file header")
    if file_header_record is None:
        raise LblrtmOutputError("TAPE12 is missing its file-header record")
    file_header = _parse_tape12_file_header(
        file_header_record,
        layout,
        expected_iemit=0,
    )

    panels: list[_Tape12Panel] = []
    total_declared_points = 0
    panel_number = 1

    while True:
        panel_record = _read_fortran_unformatted_record(
            stream, layout, record_name=f"panel {panel_number} header"
        )
        if panel_record is None:
            break
        if _is_lblrtm_endfile_record(panel_record, layout):
            if not panels:
                raise LblrtmOutputError("TAPE12 endfile record appears before any spectral panel")
            trailing_record = _read_fortran_unformatted_record(
                stream, layout, record_name="record after LBLRTM endfile"
            )
            if trailing_record is not None:
                raise LblrtmOutputError("TAPE12 contains a record after the LBLRTM endfile record")
            break
        if len(panels) >= _MAX_PANEL_COUNT:
            raise LblrtmOutputError(f"TAPE12 exceeds the panel-count limit of {_MAX_PANEL_COUNT}")

        panel_header = _parse_tape12_panel_header(panel_record, layout)
        total_declared_points += panel_header.point_count
        if total_declared_points > _MAX_TOTAL_POINTS:
            raise LblrtmOutputError(
                f"TAPE12 exceeds the total declared-point limit of {_MAX_TOTAL_POINTS}"
            )

        optical_depth_record = _read_fortran_unformatted_record(
            stream,
            layout,
            record_name=f"panel {panel_number} optical depth",
        )
        if optical_depth_record is None:
            raise LblrtmOutputError(f"TAPE12 ends before panel {panel_number} optical depth")
        if _is_lblrtm_endfile_record(optical_depth_record, layout):
            raise LblrtmOutputError(
                f"TAPE12 endfile record replaces panel {panel_number} optical depth"
            )
        optical_depth = _parse_tape12_panel_values(
            optical_depth_record,
            layout,
            expected_count=panel_header.point_count,
            value_name=f"panel {panel_number} optical depth",
        )

        try:
            index = np.arange(panel_header.point_count, dtype=np.float64)
            wavenumber_cm1 = panel_header.first_wavenumber_cm1 + index * panel_header.spacing_cm1
            panels.append(
                _Tape12Panel(
                    header=panel_header,
                    wavenumber_cm1=wavenumber_cm1,
                    radiance=None,
                    transmission=None,
                    optical_depth=optical_depth,
                )
            )
        except (MemoryError, OverflowError) as exc:
            raise LblrtmOutputError(f"cannot allocate panel {panel_number} TAPE12 arrays") from exc
        panel_number += 1

    if not panels:
        raise LblrtmOutputError("TAPE12 contains no IEMIT=0 optical-depth spectral panels")

    _validate_tape12_panel_coverage(file_header, panels, layout)
    stream.close()
    del stream
    del payload
    wavenumber_cm1, optical_depth = _concatenate_tape12_panels(
        panels,
        layout,
        value_field="optical_depth",
        value_label="optical depth",
    )
    return _validate_tape12_optical_depth_spectrum(
        wavenumber_cm1,
        optical_depth,
    )


def _read_tape12_transmission_impl(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
    *,
    permit_nonphysical: bool,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[int, tuple[int, ...]]]:
    """Decode one TAPE12, optionally deferring transmission-bound checks."""

    payload = _read_tape12_source_bytes(source)
    layout = _detect_fortran_binary_layout(payload)
    stream = io.BytesIO(payload)

    file_header_record = _read_fortran_unformatted_record(stream, layout, record_name="file header")
    if file_header_record is None:
        raise LblrtmOutputError("TAPE12 is missing its file-header record")
    file_header = _parse_tape12_file_header(file_header_record, layout)

    panels: list[_Tape12Panel] = []
    nonphysical_panel_numbers: list[int] = []
    total_declared_points = 0
    panel_number = 1

    while True:
        panel_record = _read_fortran_unformatted_record(
            stream, layout, record_name=f"panel {panel_number} header"
        )
        if panel_record is None:
            break

        if _is_lblrtm_endfile_record(panel_record, layout):
            if not panels:
                raise LblrtmOutputError("TAPE12 endfile record appears before any spectral panel")
            trailing_record = _read_fortran_unformatted_record(
                stream, layout, record_name="record after LBLRTM endfile"
            )
            if trailing_record is not None:
                raise LblrtmOutputError("TAPE12 contains a record after the LBLRTM endfile record")
            break

        if len(panels) >= _MAX_PANEL_COUNT:
            raise LblrtmOutputError(f"TAPE12 exceeds the panel-count limit of {_MAX_PANEL_COUNT}")

        panel_header = _parse_tape12_panel_header(panel_record, layout)
        total_declared_points += panel_header.point_count
        if total_declared_points > _MAX_TOTAL_POINTS:
            raise LblrtmOutputError(
                f"TAPE12 exceeds the total declared-point limit of {_MAX_TOTAL_POINTS}"
            )

        radiance_record = _read_fortran_unformatted_record(
            stream, layout, record_name=f"panel {panel_number} radiance"
        )
        if radiance_record is None:
            raise LblrtmOutputError(f"TAPE12 ends before panel {panel_number} radiance")
        if _is_lblrtm_endfile_record(radiance_record, layout):
            raise LblrtmOutputError(f"TAPE12 endfile record replaces panel {panel_number} radiance")
        radiance = _parse_tape12_panel_values(
            radiance_record,
            layout,
            expected_count=panel_header.point_count,
            value_name=f"panel {panel_number} radiance",
        )

        transmission_record = _read_fortran_unformatted_record(
            stream, layout, record_name=f"panel {panel_number} transmittance"
        )
        if transmission_record is None:
            raise LblrtmOutputError(
                f"TAPE12 ends before panel {panel_number} transmittance; "
                "an IEMIT=0 optical-depth record is not direct-RT TAPE12"
            )
        if _is_lblrtm_endfile_record(transmission_record, layout):
            raise LblrtmOutputError(
                f"TAPE12 endfile record replaces panel {panel_number} transmittance"
            )
        transmission = _parse_tape12_panel_values(
            transmission_record,
            layout,
            expected_count=panel_header.point_count,
            value_name=f"panel {panel_number} transmittance",
        )
        if np.any(transmission < 0.0) or np.any(transmission > _TRANSMISSION_UPPER_BOUND):
            nonphysical_panel_numbers.append(panel_number)
        if (
            not permit_nonphysical
            and nonphysical_panel_numbers
            and (nonphysical_panel_numbers[-1] == panel_number)
        ):
            raise LblrtmOutputError(
                f"panel {panel_number} transmittance lies outside [0, {_TRANSMISSION_UPPER_BOUND}]"
            )

        try:
            index = np.arange(panel_header.point_count, dtype=np.float64)
            wavenumber_cm1 = panel_header.first_wavenumber_cm1 + index * panel_header.spacing_cm1
            panels.append(
                _Tape12Panel(
                    header=panel_header,
                    wavenumber_cm1=wavenumber_cm1,
                    radiance=radiance,
                    transmission=transmission,
                    optical_depth=None,
                )
            )
        except (MemoryError, OverflowError) as exc:
            raise LblrtmOutputError(f"cannot allocate panel {panel_number} TAPE12 arrays") from exc
        panel_number += 1

    if not panels:
        raise LblrtmOutputError("TAPE12 contains no direct-RT spectral panels")

    _validate_tape12_panel_coverage(file_header, panels, layout)

    stream.close()
    del stream
    del payload
    wavenumber_cm1, transmission = _concatenate_tape12_panels(
        panels,
        layout,
        value_field="transmission",
        value_label="transmittance",
    )
    return (
        _validate_tape12_spectrum(
            wavenumber_cm1,
            transmission,
            enforce_transmission_bounds=not permit_nonphysical,
        ),
        (len(panels), tuple(nonphysical_panel_numbers)),
    )


def _validate_tape12_panel_coverage(
    file_header: _Tape12FileHeader,
    panels: list[_Tape12Panel],
    layout: _Tape12BinaryLayout,
) -> None:
    """Require panel spacing and endpoints to satisfy the file header."""

    first_panel = panels[0]
    last_panel = panels[-1]
    header_start_tolerance = (
        8.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(file_header.first_wavenumber_cm1),
            abs(first_panel.header.first_wavenumber_cm1),
        )
    )
    if (
        abs(first_panel.header.first_wavenumber_cm1 - file_header.first_wavenumber_cm1)
        > header_start_tolerance
    ):
        raise LblrtmOutputError("TAPE12 first panel does not start at the file-header V1")
    spacing_scale = max(
        abs(file_header.spacing_cm1),
        abs(first_panel.header.spacing_cm1),
        np.finfo(np.float32 if layout.real_bytes == 4 else np.float64).tiny,
    )
    spacing_tolerance = 8.0 * layout.real_epsilon * spacing_scale
    if abs(first_panel.header.spacing_cm1 - file_header.spacing_cm1) > spacing_tolerance:
        raise LblrtmOutputError("TAPE12 first-panel DVP does not match the file-header DV")
    final_wavenumber = float(last_panel.wavenumber_cm1[-1])
    final_grid_tolerance = layout.grid_tolerance(
        first=last_panel.header.first_wavenumber_cm1,
        last=file_header.last_requested_wavenumber_cm1,
        spacing=file_header.spacing_cm1,
        point_count=last_panel.header.point_count,
    )
    if (
        final_wavenumber < file_header.last_requested_wavenumber_cm1 - final_grid_tolerance
        or final_wavenumber
        > file_header.last_requested_wavenumber_cm1 + file_header.spacing_cm1 + final_grid_tolerance
    ):
        raise LblrtmOutputError(
            "TAPE12 final panel does not cover the file-header V2 within one output grid cell"
        )


def _read_tape12_source_bytes(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> bytes:
    """Read a supported source into read-only bytes (size-limited)."""

    if isinstance(source, bytes):
        payload = source
    elif isinstance(source, bytearray):
        if len(source) > _MAX_TAPE12_BYTES:
            raise LblrtmOutputError(f"TAPE12 exceeds the {_MAX_TAPE12_BYTES}-byte input limit")
        try:
            payload = bytes(source)
        except MemoryError as exc:
            raise LblrtmOutputError("cannot allocate memory for a copy of TAPE12") from exc
    elif isinstance(source, memoryview):
        try:
            source_bytes = source.nbytes
        except ValueError as exc:
            raise LblrtmInputError("TAPE12 memoryview has already been released") from exc
        if source_bytes > _MAX_TAPE12_BYTES:
            raise LblrtmOutputError(f"TAPE12 exceeds the {_MAX_TAPE12_BYTES}-byte input limit")
        try:
            payload = source.tobytes()
        except ValueError as exc:
            raise LblrtmInputError("TAPE12 memoryview cannot be converted to bytes") from exc
        except MemoryError as exc:
            raise LblrtmOutputError("cannot allocate memory for a copy of TAPE12") from exc
    elif isinstance(source, (str, os.PathLike)):
        try:
            path = Path(source)
        except (TypeError, ValueError) as exc:
            raise LblrtmInputError("TAPE12 path is invalid") from exc

        file_descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            file_descriptor = os.open(path, flags)
            file_status = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise LblrtmOutputError(f"TAPE12 path {path!s} is not a regular file")
            if file_status.st_size > _MAX_TAPE12_BYTES:
                raise LblrtmOutputError(
                    f"TAPE12 path {path!s} exceeds the {_MAX_TAPE12_BYTES}-byte input limit"
                )
            with os.fdopen(file_descriptor, "rb") as handle:
                file_descriptor = None
                payload = handle.read(_MAX_TAPE12_BYTES + 1)
        except LblrtmOutputError:
            raise
        except ValueError as exc:
            raise LblrtmInputError(f"TAPE12 path {path!s} is invalid") from exc
        except OSError as exc:
            raise LblrtmOutputError(f"cannot read TAPE12 path {path!s}") from exc
        except MemoryError as exc:
            raise LblrtmOutputError(f"cannot allocate memory to read TAPE12 at {path!s}") from exc
        finally:
            if file_descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(file_descriptor)
    else:
        raise LblrtmInputError(
            "TAPE12 source must be bytes, bytearray, memoryview, str, or os.PathLike"
        )

    if not payload:
        raise LblrtmOutputError("TAPE12 is empty")
    if len(payload) > _MAX_TAPE12_BYTES:
        raise LblrtmOutputError(f"TAPE12 exceeds the {_MAX_TAPE12_BYTES}-byte input limit")
    return payload


def _detect_fortran_binary_layout(payload: bytes) -> _Tape12BinaryLayout:
    """Select the unique endian/marker/precision layout supported by v12.17."""

    candidates: list[_Tape12BinaryLayout] = []
    failure_details: list[str] = []

    for byte_order in ("little", "big"):
        for record_marker_bytes in (4, 8):
            for real_bytes in (4, 8):
                layout = _Tape12BinaryLayout(
                    byte_order=byte_order,
                    record_marker_bytes=record_marker_bytes,
                    real_bytes=real_bytes,
                )
                label = f"{byte_order}/marker{record_marker_bytes}/real{real_bytes}"
                try:
                    stream = io.BytesIO(payload)
                    header_record = _read_fortran_unformatted_record(
                        stream, layout, record_name="candidate file header"
                    )
                    if header_record is None:
                        raise LblrtmOutputError("file header is absent")
                    if len(header_record) != layout.file_header_bytes:
                        raise LblrtmOutputError(f"file header has {len(header_record)} bytes")
                    magic_end = _FILE_HEADER_MAGIC_OFFSET + len(_FILE_HEADER_MAGIC)
                    if header_record[_FILE_HEADER_MAGIC_OFFSET:magic_end] != _FILE_HEADER_MAGIC:
                        raise LblrtmOutputError("file header magic is not ' LBLRTM '")

                    iemit = struct.unpack_from(
                        layout.struct_prefix + layout.integer_format,
                        header_record,
                        layout.iemit_offset,
                    )[0]
                    if iemit in (0, 1):
                        first_panel_record = _read_fortran_unformatted_record(
                            stream,
                            layout,
                            record_name="candidate first panel header",
                        )
                        if first_panel_record is not None and not (
                            _is_lblrtm_endfile_record(first_panel_record, layout)
                        ):
                            panel_header = _parse_tape12_panel_header(first_panel_record, layout)
                            if iemit == 1:
                                radiance_record = _read_fortran_unformatted_record(
                                    stream,
                                    layout,
                                    record_name=("candidate first-panel radiance"),
                                )
                                transmission_record = _read_fortran_unformatted_record(
                                    stream,
                                    layout,
                                    record_name=("candidate first-panel transmittance"),
                                )
                                if radiance_record is None or transmission_record is None:
                                    raise LblrtmOutputError("first panel is incomplete")
                                _parse_tape12_panel_values(
                                    radiance_record,
                                    layout,
                                    expected_count=panel_header.point_count,
                                    value_name=("candidate first-panel radiance"),
                                )
                                _parse_tape12_panel_values(
                                    transmission_record,
                                    layout,
                                    expected_count=panel_header.point_count,
                                    value_name=("candidate first-panel transmittance"),
                                )
                            else:
                                optical_depth_record = _read_fortran_unformatted_record(
                                    stream,
                                    layout,
                                    record_name=("candidate first-panel optical depth"),
                                )
                                if optical_depth_record is None:
                                    raise LblrtmOutputError("first panel is incomplete")
                                _parse_tape12_panel_values(
                                    optical_depth_record,
                                    layout,
                                    expected_count=panel_header.point_count,
                                    value_name=("candidate first-panel optical depth"),
                                )
                    else:
                        raise LblrtmOutputError(f"unsupported TAPE12 layout IEMIT={iemit}")
                    candidates.append(layout)
                except (LblrtmOutputError, struct.error, ValueError) as exc:
                    failure_details.append(f"{label}: {exc}")

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        details = "; ".join(failure_details[:4])
        suffix = f" ({details})" if details else ""
        raise LblrtmOutputError("cannot identify a supported TAPE12 Fortran binary layout" + suffix)

    labels = ", ".join(
        f"{item.byte_order}/marker{item.record_marker_bytes}/real{item.real_bytes}"
        for item in candidates
    )
    raise LblrtmOutputError(f"TAPE12 Fortran binary layout is ambiguous: {labels}")


def _read_fortran_unformatted_record(
    stream: BinaryIO,
    layout: _Tape12BinaryLayout,
    *,
    record_name: str = "record",
) -> bytes | None:
    """Read one complete sequential-unformatted record or clean boundary EOF."""

    record_offset = stream.tell()
    marker_bytes = layout.record_marker_bytes
    opening_marker = stream.read(marker_bytes)
    if opening_marker == b"":
        return None
    if len(opening_marker) != marker_bytes:
        raise LblrtmOutputError(
            f"{record_name} has a partial opening marker at byte {record_offset}"
        )

    try:
        payload_bytes = struct.unpack(
            layout.struct_prefix + layout.record_marker_format,
            opening_marker,
        )[0]
    except struct.error as exc:
        raise LblrtmOutputError(
            f"{record_name} opening marker cannot be decoded at byte {record_offset}"
        ) from exc

    if payload_bytes <= 0:
        raise LblrtmOutputError(
            f"{record_name} declares non-positive length {payload_bytes} "
            f"at byte {record_offset}; continuation subrecords are unsupported"
        )
    if payload_bytes > _MAX_RECORD_BYTES:
        raise LblrtmOutputError(
            f"{record_name} declares {payload_bytes} bytes at byte "
            f"{record_offset}, exceeding the {_MAX_RECORD_BYTES}-byte "
            "record limit"
        )

    payload_offset = stream.tell()
    stream.seek(0, io.SEEK_END)
    remaining_bytes = stream.tell() - payload_offset
    stream.seek(payload_offset)
    required_bytes = payload_bytes + marker_bytes
    if remaining_bytes < required_bytes:
        raise LblrtmOutputError(
            f"{record_name} is truncated at byte {record_offset}: declares "
            f"{payload_bytes} data bytes but only {remaining_bytes} bytes "
            "remain for the data and the closing record marker"
        )

    record = stream.read(payload_bytes)
    closing_marker = stream.read(marker_bytes)
    try:
        closing_bytes = struct.unpack(
            layout.struct_prefix + layout.record_marker_format,
            closing_marker,
        )[0]
    except struct.error as exc:
        raise LblrtmOutputError(
            f"{record_name} has an invalid closing marker at byte "
            f"{stream.tell() - len(closing_marker)}"
        ) from exc
    if closing_bytes != payload_bytes:
        raise LblrtmOutputError(
            f"{record_name} marker mismatch at byte {record_offset}: "
            f"opening={payload_bytes}, closing={closing_bytes}"
        )
    return record


def _parse_tape12_file_header(
    record: bytes,
    layout: _Tape12BinaryLayout,
    *,
    expected_iemit: int = 1,
) -> _Tape12FileHeader:
    """Validate the v12.17 file-header ABI, grid, and spectral mode."""

    if expected_iemit not in (0, 1):
        raise LblrtmOutputError(f"unsupported expected IEMIT={expected_iemit}")

    if len(record) != layout.file_header_bytes:
        raise LblrtmOutputError(
            f"the TAPE12 file header has {len(record)} bytes; expected {layout.file_header_bytes}"
        )
    magic_end = _FILE_HEADER_MAGIC_OFFSET + len(_FILE_HEADER_MAGIC)
    if record[_FILE_HEADER_MAGIC_OFFSET:magic_end] != _FILE_HEADER_MAGIC:
        raise LblrtmOutputError("TAPE12 file header does not contain the LBLRTM v12.17 magic")
    try:
        spacing_format = "f" if layout.real_bytes == 4 else "d"
        spacing = float(
            struct.unpack_from(
                layout.struct_prefix + spacing_format,
                record,
                layout.file_header_spacing_offset,
            )[0]
        )
        first_wavenumber = float(
            struct.unpack_from(
                layout.struct_prefix + "d",
                record,
                layout.file_header_first_wavenumber_offset,
            )[0]
        )
        last_wavenumber = float(
            struct.unpack_from(
                layout.struct_prefix + "d",
                record,
                layout.file_header_last_wavenumber_offset,
            )[0]
        )
        iemit = int(
            struct.unpack_from(
                layout.struct_prefix + layout.integer_format,
                record,
                layout.iemit_offset,
            )[0]
        )
    except struct.error as exc:
        raise LblrtmOutputError(
            "TAPE12 file header does not contain complete grid/IEMIT fields"
        ) from exc
    if not np.all(np.isfinite([spacing, first_wavenumber, last_wavenumber])):
        raise LblrtmOutputError("TAPE12 file header contains a non-finite spectral grid value")
    if spacing <= 0.0:
        raise LblrtmOutputError("TAPE12 file-header DV must be strictly positive")
    if last_wavenumber <= first_wavenumber:
        raise LblrtmOutputError("TAPE12 file-header V2 must be strictly greater than V1")
    if iemit != expected_iemit:
        expected_layout = (
            "optical-depth" if expected_iemit == 0 else "direct-RT radiance/transmittance"
        )
        raise LblrtmOutputError(
            f"TAPE12 IEMIT={iemit} is not the requested "
            f"{expected_layout} layout (IEMIT={expected_iemit})"
        )
    return _Tape12FileHeader(
        iemit=iemit,
        spacing_cm1=spacing,
        first_wavenumber_cm1=first_wavenumber,
        last_requested_wavenumber_cm1=last_wavenumber,
    )


def _parse_tape12_panel_header(record: bytes, layout: _Tape12BinaryLayout) -> _Tape12PanelHeader:
    """Decode and validate one mixed-type LBLRTM panel header."""

    if len(record) != layout.panel_header_bytes:
        raise LblrtmOutputError(
            f"a TAPE12 panel header has {len(record)} bytes; expected {layout.panel_header_bytes}"
        )
    try:
        first, last, spacing, point_count = struct.unpack(
            layout.struct_prefix + layout.panel_header_format, record
        )
    except struct.error as exc:
        raise LblrtmOutputError("TAPE12 panel header cannot be decoded") from exc

    first = float(first)
    last = float(last)
    spacing = float(spacing)
    point_count = int(point_count)
    if not np.all(np.isfinite([first, last, spacing])):
        raise LblrtmOutputError("TAPE12 panel header contains a non-finite grid value")
    if first <= 0.0 or last <= 0.0:
        raise LblrtmOutputError("TAPE12 panel wavenumbers must be strictly positive")
    if spacing <= 0.0:
        raise LblrtmOutputError("TAPE12 panel spacing must be strictly positive")
    if not 1 <= point_count <= _MAX_PANEL_POINTS:
        raise LblrtmOutputError(
            f"TAPE12 panel NLIM={point_count} is outside [1, {_MAX_PANEL_POINTS}]"
        )

    expected_last = first + (point_count - 1) * spacing
    tolerance = layout.grid_tolerance(
        first=first,
        last=last,
        spacing=spacing,
        point_count=point_count,
    )
    if abs(last - expected_last) > tolerance:
        raise LblrtmOutputError(
            "TAPE12 panel V2P is inconsistent with "
            "V1P + (NLIM - 1) * DVP: "
            f"V1P={first}, V2P={last}, DVP={spacing}, NLIM={point_count}"
        )

    return _Tape12PanelHeader(
        first_wavenumber_cm1=first,
        last_wavenumber_cm1=last,
        spacing_cm1=spacing,
        point_count=point_count,
    )


def _parse_tape12_panel_values(
    record: bytes,
    layout: _Tape12BinaryLayout,
    *,
    expected_count: int,
    value_name: str,
) -> np.ndarray:
    """Decode one radiance or transmittance record as host-endian float64."""

    expected_bytes = expected_count * layout.real_bytes
    if len(record) != expected_bytes:
        raise LblrtmOutputError(
            f"{value_name} has {len(record)} bytes; expected "
            f"{expected_bytes} for NLIM={expected_count}"
        )
    try:
        values = np.frombuffer(record, dtype=layout.value_dtype, count=expected_count).astype(
            np.float64, copy=True
        )
    except (TypeError, ValueError, MemoryError, OverflowError) as exc:
        raise LblrtmOutputError(f"{value_name} cannot be decoded") from exc
    if values.shape != (expected_count,):
        raise LblrtmOutputError(f"{value_name} decoded to an unexpected shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise LblrtmOutputError(f"{value_name} contains NaN or infinity")
    return values


def _is_lblrtm_endfile_record(record: bytes, layout: _Tape12BinaryLayout) -> bool:
    """Return whether a record is the exact six-integer ``-99`` sentinel."""

    try:
        sentinel = struct.pack(
            layout.struct_prefix + "6" + layout.integer_format,
            -99,
            -99,
            -99,
            -99,
            -99,
            -99,
        )
    except struct.error as exc:  # pragma: no cover - layout is internal
        raise LblrtmOutputError("cannot construct the LBLRTM endfile sentinel") from exc
    return record == sentinel


def _concatenate_tape12_panels(
    panels: list[_Tape12Panel],
    layout: _Tape12BinaryLayout,
    *,
    value_field: str = "transmission",
    value_label: str = "transmittance",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ordered panels, removing only identical grid-aligned overlap."""

    if not panels:
        raise LblrtmOutputError("cannot concatenate an empty TAPE12 panel list")

    first_values = getattr(panels[0], value_field, None)
    if first_values is None:
        raise LblrtmOutputError(f"first TAPE12 panel does not contain {value_label}")
    wavenumber_chunks = [panels[0].wavenumber_cm1]
    value_chunks = [first_values]
    reference_spacing = panels[0].header.spacing_cm1

    for panel_number, (previous, current) in enumerate(itertools.pairwise(panels), start=2):
        current_spacing = current.header.spacing_cm1
        spacing_scale = max(
            abs(reference_spacing),
            abs(current_spacing),
            np.finfo(np.float32 if layout.real_bytes == 4 else np.float64).tiny,
        )
        spacing_tolerance = 8.0 * layout.real_epsilon * spacing_scale
        if abs(current_spacing - reference_spacing) > spacing_tolerance:
            raise LblrtmOutputError(
                f"TAPE12 panel {panel_number} changes spacing from "
                f"{reference_spacing} to {current_spacing}"
            )

        previous_first = previous.header.first_wavenumber_cm1
        previous_last = previous.wavenumber_cm1[-1]
        current_first = current.header.first_wavenumber_cm1
        coordinate_tolerance = layout.grid_tolerance(
            first=previous_first,
            last=current.header.last_wavenumber_cm1,
            spacing=reference_spacing,
            point_count=(previous.header.point_count + current.header.point_count),
        )

        next_expected = previous_last + reference_spacing
        if current_first > previous_last + coordinate_tolerance:
            if abs(current_first - next_expected) > coordinate_tolerance:
                raise LblrtmOutputError(
                    f"TAPE12 has a gap before panel {panel_number}: "
                    f"expected {next_expected}, found {current_first}"
                )
            overlap_count = 0
        else:
            offset_float = (current_first - previous_first) / reference_spacing
            if (
                not np.isfinite(offset_float)
                or offset_float < 0.0
                or offset_float > previous.header.point_count
            ):
                raise LblrtmOutputError(f"TAPE12 panel {panel_number} is reversed or off-grid")
            offset = round(offset_float)
            aligned_first = previous_first + offset * reference_spacing
            if offset <= 0 or abs(current_first - aligned_first) > coordinate_tolerance:
                raise LblrtmOutputError(f"TAPE12 panel {panel_number} is reversed or off-grid")
            overlap_count = previous.header.point_count - offset
            if overlap_count <= 0:
                if abs(current_first - next_expected) > coordinate_tolerance:
                    raise LblrtmOutputError(f"TAPE12 panel {panel_number} is off-grid")
                overlap_count = 0
            elif overlap_count >= current.header.point_count:
                raise LblrtmOutputError(
                    f"TAPE12 panel {panel_number} is fully duplicated or "
                    "contained in the preceding panel"
                )
            else:
                previous_slice = slice(previous.header.point_count - overlap_count, None)
                current_slice = slice(0, overlap_count)
                if not np.allclose(
                    previous.wavenumber_cm1[previous_slice],
                    current.wavenumber_cm1[current_slice],
                    rtol=0.0,
                    atol=coordinate_tolerance,
                ):
                    raise LblrtmOutputError(
                        f"TAPE12 panel {panel_number} overlap is not grid-aligned"
                    )
                if (
                    previous.radiance is not None
                    and current.radiance is not None
                    and not np.array_equal(
                        previous.radiance[previous_slice],
                        current.radiance[current_slice],
                    )
                ):
                    raise LblrtmOutputError(
                        f"TAPE12 panel {panel_number} overlap has conflicting radiance values"
                    )
                previous_values = getattr(previous, value_field, None)
                current_values = getattr(current, value_field, None)
                if previous_values is None or current_values is None:
                    raise LblrtmOutputError(
                        f"TAPE12 panel {panel_number} does not contain {value_label}"
                    )
                if not np.array_equal(
                    previous_values[previous_slice],
                    current_values[current_slice],
                ):
                    raise LblrtmOutputError(
                        f"TAPE12 panel {panel_number} overlap has conflicting {value_label} values"
                    )

        wavenumber_chunks.append(current.wavenumber_cm1[overlap_count:])
        current_values = getattr(current, value_field, None)
        if current_values is None:
            raise LblrtmOutputError(f"TAPE12 panel {panel_number} does not contain {value_label}")
        value_chunks.append(current_values[overlap_count:])

    try:
        wavenumber_cm1 = np.concatenate(wavenumber_chunks)
        spectral_values = np.concatenate(value_chunks)
    except (ValueError, MemoryError) as exc:
        raise LblrtmOutputError("cannot concatenate TAPE12 spectral panels") from exc
    return wavenumber_cm1, spectral_values


def _validate_tape12_optical_depth_spectrum(
    wavenumber_cm1: np.ndarray,
    optical_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and freeze an IEMIT=0 optical-depth spectrum."""

    try:
        wavenumber = np.array(wavenumber_cm1, dtype=np.float64, copy=True, order="C")
        tau = np.array(optical_depth, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, MemoryError, OverflowError) as exc:
        raise LblrtmOutputError(
            "cannot normalize the decoded TAPE12 optical-depth spectrum"
        ) from exc

    if wavenumber.ndim != 1 or tau.ndim != 1:
        raise LblrtmOutputError("TAPE12 optical-depth arrays must be one-dimensional")
    if wavenumber.shape != tau.shape:
        raise LblrtmOutputError("TAPE12 wavenumber and optical-depth arrays have different lengths")
    if wavenumber.size < 2:
        raise LblrtmOutputError("TAPE12 optical-depth spectrum must contain at least two points")
    if not np.all(np.isfinite(wavenumber)):
        raise LblrtmOutputError("TAPE12 wavenumber grid contains NaN or infinity")
    if np.any(wavenumber <= 0.0) or np.any(np.diff(wavenumber) <= 0.0):
        raise LblrtmOutputError("TAPE12 wavenumber grid must be strictly positive and increasing")
    if not np.all(np.isfinite(tau)):
        raise LblrtmOutputError("TAPE12 optical depth contains NaN or infinity")

    wavenumber.setflags(write=False)
    tau.setflags(write=False)
    return wavenumber, tau


def _validate_tape12_spectrum(
    wavenumber_cm1: np.ndarray,
    transmission: np.ndarray,
    *,
    enforce_transmission_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the arrays the reader returns and make them read-only."""

    try:
        wavenumber = np.array(wavenumber_cm1, dtype=np.float64, copy=True, order="C")
        transmittance = np.array(transmission, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, MemoryError, OverflowError) as exc:
        raise LblrtmOutputError("cannot normalize the decoded TAPE12 spectrum") from exc

    if wavenumber.ndim != 1 or transmittance.ndim != 1:
        raise LblrtmOutputError("TAPE12 spectrum arrays must be one-dimensional")
    if wavenumber.shape != transmittance.shape:
        raise LblrtmOutputError("TAPE12 wavenumber and transmittance arrays have different lengths")
    if wavenumber.size < 2:
        raise LblrtmOutputError("TAPE12 direct-RT spectrum must contain at least two points")
    if not np.all(np.isfinite(wavenumber)):
        raise LblrtmOutputError("TAPE12 wavenumber grid contains NaN or infinity")
    if np.any(wavenumber <= 0.0) or np.any(np.diff(wavenumber) <= 0.0):
        raise LblrtmOutputError("TAPE12 wavenumber grid must be strictly positive and increasing")
    if not np.all(np.isfinite(transmittance)):
        raise LblrtmOutputError("TAPE12 transmittance contains NaN or infinity")
    if enforce_transmission_bounds and (
        np.any(transmittance < 0.0) or np.any(transmittance > _TRANSMISSION_UPPER_BOUND)
    ):
        raise LblrtmOutputError(
            f"TAPE12 transmittance lies outside [0, {_TRANSMISSION_UPPER_BOUND}]"
        )

    wavenumber.setflags(write=False)
    transmittance.setflags(write=False)
    return wavenumber, transmittance
