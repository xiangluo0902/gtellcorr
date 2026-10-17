from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gtellcorr.constants import C_KMS
from gtellcorr.data.observation.frames import air_to_vac, vac_to_air
from gtellcorr.data.observation.instrument import instruments
from gtellcorr.data.observation.readers import readers
from gtellcorr.errors import DataError


def test_echelle_load_order(make_neid: Callable[..., Path]) -> None:
    path = make_neid()
    spec = instruments.get("NEID").spectral
    reader = readers.get("echelle_hdu")
    s = reader.load_order(path, 0, spec, "espresso")
    assert s.flux.size == 38  # one NaN flux and one zero-variance pixel dropped
    assert np.all(s.flux == 1.0) and np.allclose(s.err, 0.02)  # (200/2)/median, sqrt(16)/2/100
    assert np.array_equal(s.wave_fit, vac_to_air(s.wave_input))  # NEID: vacuum, topocentric
    assert s.wave_input[0] == 600.0
    s1 = reader.load_order(path, 1, spec, "espresso")
    assert s1.flux.size == 40 and s1.flux.min() < 0  # negative flux kept
    empty = reader.load_order(path, 2, spec, "espresso")
    assert empty.flux.size == 0


def _check_grid(s: object) -> None:
    """The grid holds the spectrum's pixels at the kept positions, unchanged."""
    grid = s.grid  # type: ignore[attr-defined]
    assert grid is not None
    for name, grid_name in (("wave_fit", "wave_fit"), ("wave_input", "wave_input"),
                            ("flux", "flux"), ("err", "err")):  # fmt: skip
        assert np.array_equal(getattr(grid, grid_name)[grid.kept], getattr(s, name)), name


def test_echelle_grid_is_the_delivered_order(make_neid: Callable[..., Path]) -> None:
    path = make_neid()
    spec = instruments.get("NEID").spectral
    reader = readers.get("echelle_hdu")
    s = reader.load_order(path, 0, spec, "espresso")
    _check_grid(s)
    assert s.grid is not None and s.grid.kept.size == 40 and s.grid.kept.sum() == 38
    raw = reader.load_order_raw(path, 0, spec, "espresso")
    assert np.array_equal(s.grid.kept, raw.valid_fit_pixels)
    assert np.array_equal(s.grid.wave_input, raw.wave_input)
    empty = reader.load_order(path, 2, spec, "espresso")
    assert empty.grid is not None and empty.grid.kept.size == 40 and not empty.grid.kept.any()


def test_pepsi_grid_is_the_delivered_order(make_pepsi: Callable[..., Path]) -> None:
    path = make_pepsi()
    reader = readers.get("pepsi_dxt")
    s = reader.load_order(path, 0, instruments.get("PEPSI").spectral, "espresso")
    _check_grid(s)
    assert s.grid is not None and s.grid.kept.size == 60
    assert np.flatnonzero(~s.grid.kept).tolist() == [4, 10]  # masked, zero variance
    spec = instruments.get("PEPSI").spectral.replace(
        order_windows_nm=[[629.0, 630.5], [630.5, 632.0]]
    )
    first = reader.load_order(path, 0, spec, "espresso")
    second = reader.load_order(path, 1, spec, "espresso")
    _check_grid(first)
    _check_grid(second)
    assert first.grid is not None and second.grid is not None
    assert first.grid.kept.size + second.grid.kept.size <= 60
    raw = reader.load_order_raw(path, 0, instruments.get("PEPSI").spectral, "espresso")
    assert raw.grid is not None and raw.grid.kept.size == 60
    assert np.array_equal(raw.grid.flux[raw.grid.kept], raw.flux)  # masked pixel left out


def test_grid_must_match_the_spectrum() -> None:
    from gtellcorr.data.observation.spectrum import InputGrid, OrderSpectrum

    ones = np.ones(3)
    grid = InputGrid(wave_input=ones, wave_fit=ones, flux=ones, err=ones,
                     kept=np.array([True, False, True]))  # fmt: skip
    OrderSpectrum(wave_fit=ones[:2], wave_input=ones[:2], flux=ones[:2], cont=ones[:2],
                  err=ones[:2], grid=grid)  # fmt: skip
    with pytest.raises(ValueError, match="grid keeps 2 pixels"):
        OrderSpectrum(wave_fit=ones, wave_input=ones, flux=ones, cont=ones, err=ones, grid=grid)
    with pytest.raises(ValueError, match=r"InputGrid\.flux"):
        InputGrid(wave_input=ones, wave_fit=ones, flux=ones[:2], err=ones, kept=grid.kept)


def test_echelle_raw_order_status(make_neid: Callable[..., Path]) -> None:
    path = make_neid()
    reader = readers.get("echelle_hdu")
    spec = instruments.get("NEID").spectral
    raw = reader.load_order_raw(path, 0, spec, "espresso")
    assert raw.flux.shape == (40,) and raw.status == "raw_only"
    assert raw.valid_fit_pixels.sum() == 38
    missing = reader.load_order_raw(path, 2, spec, "espresso")
    assert missing.status == "no_valid_wavelength" and np.all(np.isnan(missing.wave_fit))


def test_echelle_undoes_berv(make_espresso: Callable[..., Path]) -> None:
    path = make_espresso()
    spec = instruments.get("ESPRESSO").spectral
    s = readers.get("echelle_hdu").load_order(path, 0, spec, "espresso")
    expected = vac_to_air(air_to_vac(s.wave_input) / (1 + 12.5 / C_KMS))
    assert np.array_equal(s.wave_fit, expected)


def test_echelle_leaves_flagged_pixels_out(make_espresso: Callable[..., Path]) -> None:
    path = make_espresso()
    spec = instruments.get("ESPRESSO").spectral
    reader = readers.get("echelle_hdu")
    s = reader.load_order(path, 0, spec, "espresso")
    assert s.flux.size == 35 and np.all(s.flux == 1.0)  # the zero-flux grid end left out
    assert s.grid is not None and s.grid.kept.size == 40
    assert np.flatnonzero(~s.grid.kept).tolist() == [35, 36, 37, 38, 39]
    assert np.array_equal(s.grid.flux[~s.grid.kept], np.zeros(5))  # delivered values kept
    bad = reader.load_order(path, 1, spec, "espresso")
    assert bad.flux.size == 39 and bad.grid is not None and not bad.grid.kept[7]
    raw = reader.load_order_raw(path, 0, spec, "espresso")
    assert raw.valid_fit_pixels.sum() == 35
    unflagged = reader.load_order(path, 0, spec.replace(quality=None), "espresso")
    assert unflagged.flux.size == 40  # without the flags the zeros enter the fit


def test_echelle_requires_hdu_names(make_neid: Callable[..., Path]) -> None:
    spec = instruments.get("NEID").spectral.replace(wave=None)
    with pytest.raises(DataError, match=r"needs spectral\.wave"):
        readers.get("echelle_hdu").load_order(make_neid(), 0, spec, "espresso")


def test_pepsi_load_order(make_pepsi: Callable[..., Path]) -> None:
    path = make_pepsi()
    spec = instruments.get("PEPSI").spectral
    reader = readers.get("pepsi_dxt")
    s = reader.load_order(path, 0, spec, "espresso")
    assert s.flux.size == 58  # masked pixel and zero-variance pixel dropped
    assert np.all(s.flux == 1.0)
    wave_nm = np.linspace(6300.0, 6310.0, 60) / 10.0  # as stored in the file, in nm
    kept = [i for i in range(60) if i not in (4, 10)]
    assert np.array_equal(s.wave_input, wave_nm[kept])  # the delivered wavelengths
    expected = vac_to_air(air_to_vac(wave_nm) * (1 + -25000.0 / 1000.0 / C_KMS))
    assert np.array_equal(s.wave_fit, expected[kept])
    with pytest.raises(DataError, match="only order 0"):
        reader.load_order(path, 1, spec, "espresso")


def test_pepsi_windows_and_raw(make_pepsi: Callable[..., Path]) -> None:
    path = make_pepsi()
    reader = readers.get("pepsi_dxt")
    spec = instruments.get("PEPSI").spectral.replace(
        order_windows_nm=[[629.0, 630.5], [630.5, 632.0]]
    )
    first = reader.load_order(path, 0, spec, "espresso")
    second = reader.load_order(path, 1, spec, "espresso")
    assert first.flux.size + second.flux.size == 58
    raw = reader.load_order_raw(path, 0, instruments.get("PEPSI").spectral, "espresso")
    assert raw.flux.size == 59 and raw.status == "raw_only"  # only the masked pixel removed
    assert raw.valid_fit_pixels.sum() == 58
    windowed = reader.load_order_raw(path, 1, spec, "espresso")
    assert windowed.flux.size < 59


def test_pepsi_requires_columns(make_pepsi: Callable[..., Path]) -> None:
    spec = instruments.get("PEPSI").spectral.replace(columns={"wave": "Arg"})
    with pytest.raises(DataError, match=r"needs spectral\.columns"):
        readers.get("pepsi_dxt").load_order(make_pepsi(), 0, spec, "espresso")
