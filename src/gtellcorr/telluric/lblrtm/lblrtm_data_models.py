"""Read-only value objects shared by the LBLRTM backend.

This module contains only the value objects of the direct radiative-transfer
path: molecules, spectral window, line of sight, run configuration and the
run, transmission and optical-depth results.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    _NumpyArray: TypeAlias = np.ndarray
else:
    # Keep the execution-only runner import free of NumPy while leaving every
    # public annotation resolvable by typing.get_type_hints at runtime.
    _NumpyArray: TypeAlias = Any

from .lblrtm_errors import LblrtmInputError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WAVELENGTH_MEDIA = frozenset({"air", "vacuum"})
_CONTINUUM_MODES = frozenset({"full", "none"})
_SPECTRAL_OUTPUTS = frozenset({"optical_depth", "transmission"})
_MAXIMUM_TRANSMISSION = 1.00001


def _is_boolean_scalar(value: Any) -> bool:
    """Recognize Python and NumPy booleans without importing NumPy eagerly."""

    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__ == "numpy" and value_type.__name__ in {"bool", "bool_"}
    )


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    """Return one finite float while rejecting booleans and bad coercions."""

    if _is_boolean_scalar(value):
        raise LblrtmInputError(f"{field_name} must be a real number, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(f"{field_name} must be convertible to a finite float") from exc
    if not math.isfinite(normalized):
        raise LblrtmInputError(f"{field_name} must be finite")
    return normalized


def _freeze_metadata_value(value: Any, *, field_name: str) -> Any:
    """Make JSON-like spectrum metadata read-only, recursively."""

    import numpy as np

    if isinstance(value, Mapping):
        frozen_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LblrtmInputError(f"{field_name} mapping keys must be strings")
            frozen_mapping[key] = _freeze_metadata_value(
                item,
                field_name=f"{field_name}.{key}",
            )
        return MappingProxyType(frozen_mapping)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(
                item,
                field_name=f"{field_name}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LblrtmInputError(f"{field_name} must be finite")
        return value
    raise LblrtmInputError(f"{field_name} must contain only JSON-like values")


def _immutable_float64_array(
    value: Any,
    *,
    field_name: str,
) -> _NumpyArray:
    """Return a read-only one-dimensional float64 copy."""

    import numpy as np

    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LblrtmInputError(f"{field_name} must be convertible to a float64 array") from exc
    if array.ndim != 1:
        raise LblrtmInputError(f"{field_name} must be one-dimensional")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LblrtmMoleculeDefinition:
    """One molecule of the user atmosphere passed to LBLRTM."""

    molecule_id: int
    canonical_name: str
    profile_field: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.molecule_id, bool)
            or not isinstance(self.molecule_id, int)
            or self.molecule_id < 1
        ):
            raise LblrtmInputError("molecule_id must be a positive integer")
        if not isinstance(self.canonical_name, str) or not self.canonical_name:
            raise LblrtmInputError("canonical_name must be a non-empty string")
        if (
            not isinstance(self.profile_field, str)
            or not self.profile_field.endswith("_ppmv")
            or len(self.profile_field) == len("_ppmv")
        ):
            raise LblrtmInputError(
                "profile_field must be a non-empty molecule name followed by '_ppmv'"
            )
        if not isinstance(self.aliases, tuple):
            raise LblrtmInputError("aliases must be a tuple of strings")
        if any(not isinstance(alias, str) or not alias for alias in self.aliases):
            raise LblrtmInputError("aliases must contain only non-empty strings")
        if len(set(self.aliases)) != len(self.aliases):
            raise LblrtmInputError("aliases must not contain duplicates")


@dataclass(frozen=True, slots=True)
class LblrtmSpectralWindow:
    """One direct LBLRTM calculation interval in vacuum wavenumber."""

    minimum_wavenumber_cm1: float
    maximum_wavenumber_cm1: float

    def __post_init__(self) -> None:
        lower = _coerce_finite_float(
            self.minimum_wavenumber_cm1,
            field_name="minimum_wavenumber_cm1",
        )
        upper = _coerce_finite_float(
            self.maximum_wavenumber_cm1,
            field_name="maximum_wavenumber_cm1",
        )
        if lower <= 0.0:
            raise LblrtmInputError("minimum_wavenumber_cm1 must be greater than zero")
        if upper <= lower:
            raise LblrtmInputError("maximum_wavenumber_cm1 must exceed minimum_wavenumber_cm1")
        object.__setattr__(self, "minimum_wavenumber_cm1", lower)
        object.__setattr__(self, "maximum_wavenumber_cm1", upper)

    @property
    def width_cm1(self) -> float:
        """Return the interval width in reciprocal centimetres."""

        return self.maximum_wavenumber_cm1 - self.minimum_wavenumber_cm1


@dataclass(frozen=True, slots=True)
class LblrtmLineOfSight:
    """Observer geometry for an upward-looking atmospheric path."""

    airmass: float
    observer_altitude_km: float | None = None
    path_end_altitude_km: float | None = None
    reference_latitude_deg: float = 45.0

    def __post_init__(self) -> None:
        airmass = _coerce_finite_float(
            self.airmass,
            field_name="airmass",
        )
        if airmass < 1.0:
            raise LblrtmInputError("airmass must be >= 1")
        object.__setattr__(self, "airmass", airmass)

        for field_name in ("observer_altitude_km", "path_end_altitude_km"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = _coerce_finite_float(
                value,
                field_name=field_name,
            )
            object.__setattr__(self, field_name, normalized)
        if (
            self.observer_altitude_km is not None
            and self.path_end_altitude_km is not None
            and self.path_end_altitude_km <= self.observer_altitude_km
        ):
            raise LblrtmInputError("path_end_altitude_km must exceed observer_altitude_km")

        reference_latitude_deg = _coerce_finite_float(
            self.reference_latitude_deg,
            field_name="reference_latitude_deg",
        )
        if not -90.0 <= reference_latitude_deg <= 90.0:
            raise LblrtmInputError("reference_latitude_deg must lie within [-90, 90]")
        object.__setattr__(
            self,
            "reference_latitude_deg",
            reference_latitude_deg,
        )


@dataclass(frozen=True, slots=True)
class LblrtmRunConfiguration:
    """The scientific settings written into one LBLRTM TAPE5."""

    spectral_window: LblrtmSpectralWindow
    line_of_sight: LblrtmLineOfSight
    continuum_mode: str = "full"
    spectral_output: str = "transmission"
    optical_depth_grid_spacing_cm1: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spectral_window, LblrtmSpectralWindow):
            raise LblrtmInputError("spectral_window must be an LblrtmSpectralWindow")
        if not isinstance(self.line_of_sight, LblrtmLineOfSight):
            raise LblrtmInputError("line_of_sight must be an LblrtmLineOfSight")
        if not isinstance(self.continuum_mode, str) or self.continuum_mode not in _CONTINUUM_MODES:
            choices = ", ".join(sorted(_CONTINUUM_MODES))
            raise LblrtmInputError(f"continuum_mode must be one of: {choices}")
        if (
            not isinstance(self.spectral_output, str)
            or self.spectral_output not in _SPECTRAL_OUTPUTS
        ):
            choices = ", ".join(sorted(_SPECTRAL_OUTPUTS))
            raise LblrtmInputError(f"spectral_output must be one of: {choices}")
        if self.optical_depth_grid_spacing_cm1 is not None:
            spacing = _coerce_finite_float(
                self.optical_depth_grid_spacing_cm1,
                field_name="optical_depth_grid_spacing_cm1",
            )
            if spacing <= 0.0:
                raise LblrtmInputError("optical_depth_grid_spacing_cm1 must be strictly positive")
            object.__setattr__(
                self,
                "optical_depth_grid_spacing_cm1",
                spacing,
            )


@dataclass(frozen=True, slots=True)
class LblrtmRunResult:
    """Files and diagnostics kept from one successful LBLRTM run."""

    tape12_bytes: bytes
    tape6_text: str
    stdout: str
    stderr: str
    return_code: int
    elapsed_seconds: float
    resource_fingerprint: str
    tape5_sha256: str
    tape6_sha256: str
    tape12_sha256: str
    work_directory: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tape12_bytes, bytes) or not self.tape12_bytes:
            raise LblrtmInputError("tape12_bytes must be non-empty bytes")
        for field_name in ("tape6_text", "stdout", "stderr"):
            if not isinstance(getattr(self, field_name), str):
                raise LblrtmInputError(f"{field_name} must be a string")
        if not self.tape6_text:
            raise LblrtmInputError("tape6_text must be non-empty")
        try:
            tape6_bytes = self.tape6_text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise LblrtmInputError("tape6_text must contain only ASCII characters") from exc
        if (
            isinstance(self.return_code, bool)
            or not isinstance(self.return_code, int)
            or self.return_code != 0
        ):
            raise LblrtmInputError(
                "LblrtmRunResult represents a successful run; return_code must be zero"
            )
        elapsed = _coerce_finite_float(
            self.elapsed_seconds,
            field_name="elapsed_seconds",
        )
        if elapsed < 0.0:
            raise LblrtmInputError("elapsed_seconds must be non-negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        for field_name in (
            "resource_fingerprint",
            "tape5_sha256",
            "tape6_sha256",
            "tape12_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise LblrtmInputError(f"{field_name} must contain 64 lowercase hexadecimal digits")
        actual_tape6_sha256 = sha256(tape6_bytes).hexdigest()
        if self.tape6_sha256 != actual_tape6_sha256:
            raise LblrtmInputError("tape6_sha256 does not match tape6_text")
        actual_tape12_sha256 = sha256(self.tape12_bytes).hexdigest()
        if self.tape12_sha256 != actual_tape12_sha256:
            raise LblrtmInputError("tape12_sha256 does not match tape12_bytes")
        if self.work_directory is not None:
            try:
                work_directory = Path(self.work_directory)
            except (TypeError, ValueError) as exc:
                raise LblrtmInputError("work_directory must be path-like or None") from exc
            object.__setattr__(self, "work_directory", work_directory)


@dataclass(frozen=True, slots=True, eq=False)
class LblrtmTransmissionSpectrum:
    """Unconvolved transmission on an increasing nm grid (read-only).

    Transmission values outside ``[0, 1.00001]`` are rejected, never clipped.
    Metadata is read-only and limited to JSON-like values.
    """

    wavelength_nm: _NumpyArray
    transmission: _NumpyArray
    wavelength_medium: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    __hash__ = None

    def __post_init__(self) -> None:
        import numpy as np

        wavelength = _immutable_float64_array(
            self.wavelength_nm,
            field_name="wavelength_nm",
        )
        transmission = _immutable_float64_array(
            self.transmission,
            field_name="transmission",
        )
        if wavelength.size < 2:
            raise LblrtmInputError("a transmission spectrum must contain at least two samples")
        if wavelength.shape != transmission.shape:
            raise LblrtmInputError("wavelength_nm and transmission must have identical shapes")
        if not np.all(np.isfinite(wavelength)):
            raise LblrtmInputError("wavelength_nm must contain only finite values")
        if not np.all(wavelength > 0.0):
            raise LblrtmInputError("wavelength_nm must be strictly positive")
        if not np.all(np.diff(wavelength) > 0.0):
            raise LblrtmInputError("wavelength_nm must be strictly increasing")
        if not np.all(np.isfinite(transmission)):
            raise LblrtmInputError("transmission must contain only finite values")
        if np.any(transmission < 0.0):
            raise LblrtmInputError("transmission must not be negative")
        if np.any(transmission > _MAXIMUM_TRANSMISSION):
            raise LblrtmInputError("transmission must not exceed 1.00001")
        if (
            not isinstance(self.wavelength_medium, str)
            or self.wavelength_medium not in _WAVELENGTH_MEDIA
        ):
            choices = ", ".join(sorted(_WAVELENGTH_MEDIA))
            raise LblrtmInputError(f"wavelength_medium must be one of: {choices}")
        if not isinstance(self.metadata, Mapping):
            raise LblrtmInputError("metadata must be a mapping")

        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata_value(
                self.metadata,
                field_name="metadata",
            ),
        )

    def __eq__(self, other: object) -> bool:
        """Compare scientific arrays and metadata by exact value."""

        import numpy as np

        if not isinstance(other, LblrtmTransmissionSpectrum):
            return NotImplemented
        return (
            self.wavelength_medium == other.wavelength_medium
            and np.array_equal(self.wavelength_nm, other.wavelength_nm)
            and np.array_equal(self.transmission, other.transmission)
            and self.metadata == other.metadata
        )


@dataclass(frozen=True, slots=True, eq=False)
class LblrtmOpticalDepthSpectrum:
    """Unconvolved sight-line optical depth on an increasing nm grid (read-only).

    Values are the finite IEMIT=0 TAPE12 output.  Finite negative values are
    retained unchanged for explicit numerical diagnosis; NaN and infinity are
    rejected.
    """

    wavelength_nm: _NumpyArray
    optical_depth: _NumpyArray
    wavelength_medium: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    __hash__ = None

    def __post_init__(self) -> None:
        import numpy as np

        wavelength = _immutable_float64_array(
            self.wavelength_nm,
            field_name="wavelength_nm",
        )
        optical_depth = _immutable_float64_array(
            self.optical_depth,
            field_name="optical_depth",
        )
        if wavelength.size < 2:
            raise LblrtmInputError("an optical-depth spectrum must contain at least two samples")
        if wavelength.shape != optical_depth.shape:
            raise LblrtmInputError("wavelength_nm and optical_depth must have identical shapes")
        if not np.all(np.isfinite(wavelength)):
            raise LblrtmInputError("wavelength_nm must contain only finite values")
        if not np.all(wavelength > 0.0):
            raise LblrtmInputError("wavelength_nm must be strictly positive")
        if not np.all(np.diff(wavelength) > 0.0):
            raise LblrtmInputError("wavelength_nm must be strictly increasing")
        if not np.all(np.isfinite(optical_depth)):
            raise LblrtmInputError("optical_depth must contain only finite values")
        if (
            not isinstance(self.wavelength_medium, str)
            or self.wavelength_medium not in _WAVELENGTH_MEDIA
        ):
            choices = ", ".join(sorted(_WAVELENGTH_MEDIA))
            raise LblrtmInputError(f"wavelength_medium must be one of: {choices}")
        if not isinstance(self.metadata, Mapping):
            raise LblrtmInputError("metadata must be a mapping")

        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "optical_depth", optical_depth)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata_value(
                self.metadata,
                field_name="metadata",
            ),
        )

    def __eq__(self, other: object) -> bool:
        """Compare scientific arrays and metadata by exact value."""

        import numpy as np

        if not isinstance(other, LblrtmOpticalDepthSpectrum):
            return NotImplemented
        return (
            self.wavelength_medium == other.wavelength_medium
            and np.array_equal(self.wavelength_nm, other.wavelength_nm)
            and np.array_equal(self.optical_depth, other.optical_depth)
            and self.metadata == other.metadata
        )


__all__ = [
    "LblrtmLineOfSight",
    "LblrtmMoleculeDefinition",
    "LblrtmOpticalDepthSpectrum",
    "LblrtmRunConfiguration",
    "LblrtmRunResult",
    "LblrtmSpectralWindow",
    "LblrtmTransmissionSpectrum",
]
